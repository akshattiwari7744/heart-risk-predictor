"""
Phase 2 - build leakage-safe, reproducible train/val/test data + fit preprocessors on TRAIN ONLY.
Reads data/interim/brfss_phase1_subset.parquet (made in Phase 1). Never touches the XPT. Trains NO model.

    python src\\build_dataset.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import config as C
from cleaning import clean_frame, recode
from preprocess import build_linear, build_tree, groups


def sha256(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(n):
            h.update(chunk)
    return h.hexdigest()


def raw_checks(d: pd.DataFrame) -> list[str]:
    """Evidence checks on RAW codes (valid-target rows) that back the resolutions in the audit."""
    L: list[str] = []
    cols = set(d.columns)
    L.append("### Age variable")
    L.append(f"- `AGE` present in subset: **{'AGE' in cols}**; `_AGEG5YR` present: **{'_AGEG5YR' in cols}**.")
    if "_AGEG5YR" in cols:
        vc = d["_AGEG5YR"].value_counts(dropna=False).sort_index()
        L.append("- `_AGEG5YR` counts: " + ", ".join(f"{k:g}:{v:,}" if pd.notna(k) else f"NaN:{v:,}" for k, v in vc.items()))
    L.append("\n### PHYSHLTH / MENTHLTH special codes (CDC: 88=none, 77/99=DK/refused; 7 and 9 are real day counts)")
    for c in ("PHYSHLTH", "MENTHLTH"):
        if c in cols:
            s = pd.to_numeric(d[c], errors="coerce")
            L.append(f"- `{c}`: 77:{int((s==77).sum()):,} | 88:{int((s==88).sum()):,} | 99:{int((s==99).sum()):,} | "
                     f"7:{int((s==7).sum()):,} (real 7 days) | 9:{int((s==9).sum()):,} (real 9 days) | NaN:{int(s.isna().sum()):,}")
    if "_BMI5" in cols:
        s = pd.to_numeric(d["_BMI5"], errors="coerce")
        L.append(f"\n### `_BMI5`\n- observed min={s.min():g}, max={s.max():g} (codebook: x100, valid 1200-9999) "
                 f"-> divide by 100; NaN: {int(s.isna().sum()):,}")
    if {"SMOKE100", "SMOKDAY2"} <= cols:
        L.append("\n### SMOKDAY2 skip pattern (CDC `_SMOKER3` logic: SMOKDAY2 used only when SMOKE100=1)")
        t = d.groupby(d["SMOKE100"].fillna(-1))["SMOKDAY2"].agg(n="size", smokday2_nan=lambda x: int(x.isna().sum()))
        t["pct_nan"] = (100 * t["smokday2_nan"] / t["n"]).round(2)
        L.append("```\n" + t.rename_axis("SMOKE100 (-1=NaN)").to_string() + "\n```")
        if "_SMOKER3" in cols:
            s1, s2 = d["SMOKE100"], d["SMOKDAY2"]
            der = pd.Series(9.0, index=d.index)
            der[s1 == 2] = 4
            for code in (1, 2, 3):
                der[(s1 == 1) & (s2 == code)] = code
            mism = int((der != d["_SMOKER3"].fillna(9)).sum())
            L.append(f"- `_SMOKER3` re-derived from SMOKE100/SMOKDAY2 per CDC SAS rule: **{mism:,} mismatches** (0 expected).")
    if {"_TOTINDA", "EXERANY2"} <= cols:
        der = d["EXERANY2"].map({1: 1, 2: 2}).fillna(9)
        L.append(f"\n### `_TOTINDA` vs EXERANY2 rule: **{int((der != d['_TOTINDA'].fillna(9)).sum()):,} mismatches** (0 expected).")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=str(C.SUBSET_PATH))
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--allow-count-mismatch", action="store_true", help="skip the 356,158 / 352,145 assertions (tests only)")
    a = ap.parse_args()

    for d in (C.PROCESSED_DIR, C.ARTIFACTS_DIR, C.DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    subset = a.subset
    raw = pd.read_parquet(subset)
    raw.index.name = "row_id"                       # position in original XPT order
    print(f"Loaded subset: {raw.shape[0]:,} rows x {raw.shape[1]} cols")

    # ---- target + integrity
    t = raw[C.TARGET_SRC]
    valid = t.isin([1, 2])
    if not a.allow_count_mismatch:
        assert len(raw) == C.EXPECTED_ROWS, f"rows {len(raw):,} != {C.EXPECTED_ROWS:,}"
        assert int(valid.sum()) == C.EXPECTED_VALID, f"valid {int(valid.sum()):,} != {C.EXPECTED_VALID:,}"
    d = raw.loc[valid].copy()
    y = (d[C.TARGET_SRC] == 1).astype("int8").to_numpy()
    print(f"Valid-target rows: {len(d):,}; positive rate {100 * y.mean():.2f}%")

    # ---- stateless cleaning (+ leakage guard), then split
    X, ev, src_map = clean_frame(d)
    assert C.TARGET_SRC not in X.columns and not any(c in X.columns for c in C.FORBIDDEN_SOURCES)

    idx = np.arange(len(X))
    tr, tmp = train_test_split(idx, test_size=C.VAL_FRAC + C.TEST_FRAC, stratify=y, random_state=a.seed)
    va, te = train_test_split(tmp, test_size=C.TEST_FRAC / (C.VAL_FRAC + C.TEST_FRAC), stratify=y[tmp], random_state=a.seed)
    assert len(set(tr) & set(va)) == len(set(tr) & set(te)) == len(set(va) & set(te)) == 0
    assert len(tr) + len(va) + len(te) == len(X)
    splits = {"train": tr, "val": va, "test": te}
    rates = {k: float(y[v].mean()) for k, v in splits.items()}

    # ---- fit preprocessors on TRAIN ONLY
    Xtr = X.iloc[tr]
    lin, tree = build_linear(), build_tree()
    lin.fit(Xtr)
    tree.fit(Xtr)
    out_shapes = {}
    for name, ix in splits.items():
        a1, a2 = lin.transform(X.iloc[ix]), tree.transform(X.iloc[ix])
        assert not a1.isna().any().any(), "linear preprocessor left NaNs"
        out_shapes[name] = {"linear": list(a1.shape), "tree": list(a2.shape)}

    g = groups()
    scaled = g["continuous"] + g["ordinal"]
    num = lin.named_transformers_["num"]
    sc, imp = num.named_steps["sc"], num.named_steps["imp"]
    train_med = Xtr[scaled].median()
    back = imp.statistics_[: len(scaled)] * sc.scale_ + sc.mean_
    fit_on_train_ok = bool(np.allclose(back, train_med.to_numpy(), atol=1e-3, equal_nan=True))
    assert fit_on_train_ok, "imputer/scaler statistics do not match TRAIN-only medians"

    # ---- persist data splits (clean, un-scaled) + preprocessors
    for name, ix in splits.items():
        df = X.iloc[ix].copy()
        df.insert(0, "y", y[ix])
        df.insert(0, "row_id", X.index.to_numpy()[ix])
        df["ev_race_group"] = ev["race_group"].to_numpy()[ix]
        if "weight" in ev:
            df["ev_weight"] = ev["weight"].to_numpy()[ix]
        df.to_parquet(C.PROCESSED_DIR / f"{name}.parquet", index=False)
    joblib.dump(lin, C.ARTIFACTS_DIR / "preprocessor_linear.joblib", compress=3)
    joblib.dump(tree, C.ARTIFACTS_DIR / "preprocessor_tree.joblib", compress=3)

    # ---- single-feature leakage screen (train only)
    auc_rows = []
    for c in X.columns:
        v = Xtr[c].fillna(Xtr[c].median()).to_numpy()
        if np.nanstd(v) > 0:
            auc = roc_auc_score(y[tr], v)
            auc_rows.append((c, max(auc, 1 - auc)))
    auc_rows.sort(key=lambda r: -r[1])
    suspicious = [c for c, v in auc_rows if v > 0.85]

    miss = (100 * Xtr.isna().mean()).round(2).sort_values(ascending=False)
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"), "seed": a.seed,
        "subset_file": str(subset), "subset_sha256": sha256(subset),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "split": {k: {"n": int(len(v)), "positive_rate": rates[k]} for k, v in splits.items()},
        "features": {f["out"]: {"source": src_map[f["out"]], "kind": f["kind"], "evidence": f["evidence"]} for f in FEATURES_LIST()},
        "groups": g, "output_shapes": out_shapes, "fit_on_train_only_verified": fit_on_train_ok,
        "train_medians": {k: float(v) for k, v in train_med.items()},
    }
    (C.ARTIFACTS_DIR / "preprocess_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- feature_spec_v1.csv
    rows = []
    for f in C.FEATURES:
        rows.append(dict(model_name=f["out"], source_column=src_map[f["out"]], role="model_input", kind=f["kind"],
                         evidence=f["evidence"], code_map=json.dumps({str(k): (None if (isinstance(v, float) and np.isnan(v)) else v)
                                                                       for k, v in f["codes"].items()}) if f["codes"] else f["rule"],
                         description=f["label"], note=f["note"]))
    for f in C.EVAL_ONLY:
        rows.append(dict(model_name=f["out"], source_column=f["src"], role="eval_only", kind=f["kind"], evidence=f["evidence"],
                         code_map="1-8; 9->NaN", description=f["label"], note="subgroup evaluation only"))
    rows.append(dict(model_name="weight", source_column=C.WEIGHT_SRC, role="eval_only", kind="weight", evidence="CDC",
                     code_map="", description="survey weight", note="secondary weighted metrics only"))
    for k, why in C.FORBIDDEN_SOURCES.items():
        rows.append(dict(model_name="", source_column=k, role="excluded_leakage_or_proxy", kind="", evidence="spec", code_map="",
                         description=why, note=""))
    for k, why in C.OTHER_EXCLUSIONS.items():
        rows.append(dict(model_name="", source_column=k, role="excluded", kind="", evidence="", code_map="", description=why, note=""))
    pd.DataFrame(rows).to_csv(C.DOCS_DIR / "feature_spec_v1.csv", index=False, encoding="utf-8-sig")

    # ---- report
    R = ["# Phase 2 Report - preprocessing & split",
         f"_Generated {meta['created']} (seed {a.seed}). Every number comes from your data; nothing hand-entered._\n",
         "## Split (stratified on target, fixed seed)",
         "| split | rows | positives | positive rate |", "|---|---|---|---|"]
    for k, v in splits.items():
        R.append(f"| {k} | {len(v):,} | {int(y[v].sum()):,} | {100 * rates[k]:.3f}% |")
    R += [f"\nFit-on-train-only check (imputer/scaler statistics == train medians): **{fit_on_train_ok}**",
          f"Max positive-rate gap between splits: {100 * (max(rates.values()) - min(rates.values())):.3f} pp",
          "\n## Model inputs (" + str(len(C.FEATURES)) + ")",
          "| model name | source | evidence | train NaN % |", "|---|---|---|---|"]
    for f in C.FEATURES:
        R.append(f"| {f['out']} | {src_map[f['out']]} | {f['evidence']} | {miss[f['out']]} |")
    R += ["\n## Single-feature leakage screen (train; folded AUC; >0.85 would need investigation)",
          "| feature | AUC |", "|---|---|", *[f"| {c} | {v:.3f} |" for c, v in auc_rows],
          f"\nFlagged (>0.85): {', '.join(suspicious) if suspicious else 'none'}",
          "\n## Raw-code evidence checks (valid-target rows)", *raw_checks(d)]
    ac = C.DOCS_DIR / "all_columns.csv"
    if ac.exists():
        a_df = pd.read_csv(ac)
        r = a_df[a_df["column"].str.upper() == "FRNCHDA_"]
        R.append("\n### FRNCHDA_\n- " + (f"present in all_columns.csv, non-null {r.iloc[0]['non_null_pct']}%." if len(r) else "not listed in all_columns.csv.")
                 + " CDC 2025 calculated-variables doc: French-fry intake per day, derived from FRENCHF1 (diet item). "
                   "Name-scan hit came from the letters 'CHD' inside the name; not related to heart disease. Excluded.")
    R.append("\n## Files written\n- data/processed/{train,val,test}.parquet\n- artifacts/preprocessor_linear.joblib, "
             "preprocessor_tree.joblib, preprocess_meta.json\n- docs/feature_spec_v1.csv, docs/phase2_report.md")
    (C.DOCS_DIR / "phase2_report.md").write_text("\n".join(R) + "\n", encoding="utf-8")
    print("\n".join(R[:12]))
    print(f"\nSuspicious single-feature AUCs: {suspicious or 'none'}\nDone. See docs/phase2_report.md")


def FEATURES_LIST():
    return C.FEATURES


if __name__ == "__main__":
    sys.exit(main())
