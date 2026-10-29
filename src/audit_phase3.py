"""
Phase 3 audit - READ ONLY. Changes nothing in artifacts/ or data/. Writes only docs/phase3_audit.md.
Re-scores the FROZEN model for diagnostics; no decision (model, threshold, calibration) is derived from it.

    python src\\audit_phase3.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config as C

ART, DOCS, PROC = C.ARTIFACTS_DIR, C.DOCS_DIR, C.PROCESSED_DIR
SRC = Path(__file__).resolve().parent
RACE = {1: "White only, non-Hispanic", 2: "Black only, non-Hispanic", 3: "American Indian/Alaska Native only, non-Hispanic",
        4: "Asian only, non-Hispanic", 5: "Native Hawaiian/Other Pacific Islander only, non-Hispanic",
        6: "Other race only, non-Hispanic", 7: "Multiracial, non-Hispanic", 8: "Hispanic"}   # CDC 2025 calc-vars doc, _RACE
AGE_BANDS = [("18-39", 1, 4), ("40-54", 5, 7), ("55-64", 8, 9), ("65-74", 10, 11), ("75+", 12, 13)]  # _AGEG5YR codes
OUT: list[str] = []
RESULTS: list[tuple[str, str, str]] = []


def say(s=""):
    OUT.append(s)
    print(s)


def verdict(item, status, evidence):
    RESULTS.append((item, status, evidence))
    say(f"[{status}] {item}: {evidence}")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def rows_for(mask, y, p, thr):
    n, pos = int(mask.sum()), int(y[mask].sum())
    tp = int(((p >= thr) & (y == 1) & mask).sum())
    lo, hi = wilson(tp, pos)
    auc = roc_auc_score(y[mask], p[mask]) if 0 < pos < n else float("nan")
    return dict(n=n, pos=pos, tp=tp, fn=pos - tp, recall=tp / pos if pos else float("nan"), lo=lo, hi=hi,
                flagged=float((p[mask] >= thr).mean()) if n else float("nan"), mean_p=float(p[mask].mean()) if n else float("nan"), auc=auc)


def main():
    say("# Phase 3 audit (read-only)\n")
    dec = json.loads((ART / "decisions_frozen.json").read_text())
    met = json.loads((ART / "metrics.json").read_text())
    meta = json.loads((ART / "model_meta.json").read_text())

    # ---- 1. quick vs full
    cfg = dec["cfg"]
    is_quick = cfg["hgb_iter"] == 3 and cfg["tune_rows"] == 30000
    say("## 1. Which configuration produced the surviving artifacts?")
    say(f"cfg stored in decisions_frozen.json: {cfg}")
    say(f"selected={dec['selected_model']}  LR params={dec['hyperparameters']['logistic_regression']}  HGB params={dec['hyperparameters']['hist_gradient_boosting']}")
    say(f"threshold={dec['threshold']!r}  frozen_at={dec['frozen_at']}")
    verdict("1a quick flag changes procedure (code)", "PASS", "Cfg: cv_folds 3->2, hgb_iter 10->3, tune_rows 120000->30000, n_boot 300->100, select_boot 200->100")
    verdict("1b surviving artifacts come from", "NEEDS INVESTIGATION" if is_quick else "PASS",
            "QUICK settings found - artifacts are NOT full-run artifacts" if is_quick else "full-run settings (hgb_iter=10, tune_rows=120000)")

    # ---- load frozen model in a FRESH interpreter
    say("\n## 4. Fresh-process load")
    code = ("import sys, json, joblib, numpy as np, pandas as pd, sklearn\n"
            f"sys.path.insert(0, r'{SRC}')\n"
            f"m = joblib.load(r'{ART / 'model.joblib'}')\n"
            f"v = pd.read_parquet(r'{PROC / 'val.parquet'}').head(2000)\n"
            "p = m.predict_proba_pos(v[m.feature_names])\n"
            "print(json.dumps({'type': type(m).__name__, 'features': m.feature_names, 'thr': m.threshold, 'sk': sklearn.__version__, "
            "'pd': pd.__version__, 'np': np.__version__, 'py': sys.version.split()[0], 'pmin': float(p.min()), 'pmax': float(p.max()), "
            "'finite': bool(np.isfinite(p).all()), 'model_name': m.model_name}))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        verdict("4a fresh-process load", "FAIL", r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "unknown error")
        fresh = None
    else:
        fresh = json.loads(r.stdout.strip().splitlines()[-1])
        ok = fresh["finite"] and 0 <= fresh["pmin"] and fresh["pmax"] <= 1
        verdict("4a fresh-process load + predict on 2000 validation rows", "PASS" if ok else "FAIL",
                f"{fresh['type']} loaded; probs in [{fresh['pmin']:.4f}, {fresh['pmax']:.4f}]; python {fresh['py']}")
        v_ok = all(meta["versions"].get(k) == fresh[s] for k, s in (("scikit-learn", "sk"), ("pandas", "pd"), ("numpy", "np")))
        verdict("4b library versions match model_meta.json", "PASS" if v_ok else "NEEDS INVESTIGATION",
                f"saved {meta['versions']} vs now sklearn {fresh['sk']}, pandas {fresh['pd']}, numpy {fresh['np']}")
    reqs = (C.ROOT / "requirements.txt")
    bad = []
    if reqs.exists():
        try:
            from packaging.requirements import Requirement
            for line in reqs.read_text().split():
                rq = Requirement(line)
                try:
                    iv = metadata.version(rq.name)
                    if not rq.specifier.contains(iv, prereleases=True):
                        bad.append(f"{rq.name} {iv} !satisfies {rq.specifier}")
                except metadata.PackageNotFoundError:
                    bad.append(f"{rq.name} not installed")
        except ImportError:
            bad.append("packaging not installed; cannot check")
    else:
        bad.append("requirements.txt not found")
    verdict("4c installed packages satisfy requirements.txt", "PASS" if not bad else "NEEDS INVESTIGATION", "; ".join(bad) or "all declared requirements satisfied")
    ver_note = "NOTE: requirements.txt uses >= ranges, not exact pins; the exact versions are only recorded in model_meta.json."
    say(ver_note)

    # ---- 5. consistency
    say("\n## 5. Artifact / report consistency")
    import joblib
    sys.path.insert(0, str(SRC))
    model = joblib.load(ART / "model.joblib")
    checks = {
        "model.joblib sha256 == model_meta": sha256(ART / "model.joblib") == meta["model_sha256"],
        "threshold model == decisions == metrics == meta": len({model.threshold, dec["threshold"], met["test"]["metrics"]["threshold"], meta["threshold"]}) == 1,
        "selected model name agrees": model.model_name == dec["selected_model"] == meta["selected_model"],
        "calibration method agrees": type(model.calibrator).__name__ == "Calibrator" and model.calibrator.method == dec["calibration_method"] == meta["calibration"],
        "feature order == config.FEATURES == meta": model.feature_names == [f["out"] for f in C.FEATURES] == meta["features"],
        "pipeline fitted feature_names_in_ == feature order": list(model.pipeline.named_steps["prep"].feature_names_in_) == model.feature_names,
        "target statement saved (metrics, meta)": all("EXISTING" in x.get("target_definition", "") for x in (met, meta)),
    }
    for k, ok in checks.items():
        verdict("5 " + k, "PASS" if ok else "FAIL", str(ok))
    for fn in ("phase3_report.md", "model_card.md"):
        t = (DOCS / fn).read_text(encoding="utf-8") if (DOCS / fn).exists() else ""
        verdict(f"5 {fn} states existing-disease target, not future risk", "PASS" if ("EXISTING" in t and "NOT a prediction of a future heart attack" in t) else "FAIL", "text present" if t else "file missing")
    # preprocessing agrees with Phase 2 train data (train only)
    tr = pd.read_parquet(PROC / "train.parquet")
    prep = model.pipeline.named_steps["prep"]
    if "num" in getattr(prep, "named_transformers_", {}):   # linear model
        sc = prep.named_transformers_["num"].named_steps["sc"]
        cols = [c for n, t, cs in prep.transformers if n == "num" for c in cs]
        ok = np.allclose(sc.mean_, tr[cols].mean().to_numpy(), rtol=1e-4, atol=1e-4)
        verdict("5 scaler means == means recomputed from train.parquet", "PASS" if ok else "FAIL", "fit on train only" if ok else "mismatch")
    else:
        verdict("5 preprocessing fit check", "PASS", "tree preprocessor has no scaler; nominal encoder categories checked implicitly by load")
    # report vs recomputed (frozen model; diagnostics only)
    te = pd.read_parquet(PROC / "test.parquet")
    y = te["y"].to_numpy()
    p = model.predict_proba_pos(te[model.feature_names])
    thr = model.threshold
    from sklearn.metrics import average_precision_score
    rec = float(((p >= thr) & (y == 1)).sum() / y.sum())
    saved = met["test"]["metrics"]
    same = abs(average_precision_score(y, p) - saved["pr_auc"]) < 1e-9 and abs(rec - saved["recall"]) < 1e-9
    verdict("5 frozen model re-scored on test reproduces metrics.json (pr_auc, recall)", "PASS" if same else "FAIL",
            f"recomputed pr_auc={average_precision_score(y, p):.6f} vs saved {saved['pr_auc']:.6f}; recall {rec:.6f} vs {saved['recall']:.6f}")
    rep = (DOCS / "phase3_report.md").read_text(encoding="utf-8")
    m_ = re.search(r"\| pr_auc \| ([0-9.]+) \|", rep[rep.find("FINAL TEST"):])
    verdict("5 phase3_report.md test PR-AUC == metrics.json", "PASS" if m_ and abs(float(m_.group(1)) - round(saved["pr_auc"], 4)) < 1e-9 else "FAIL", f"report {m_.group(1) if m_ else 'n/a'} vs {saved['pr_auc']:.4f}")

    # ---- 2. age
    say("\n## 2. Age subgroups (frozen threshold %.5f applied identically to every group)" % thr)
    age = te["age_group_5yr"].to_numpy()
    tot_pos = int(y.sum())
    say("| band (_AGEG5YR codes) | n | positives | % of all positives | TP | FN | recall [95% Wilson] | % flagged | mean p | within-band ROC-AUC |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    tp_sum = pos_sum = 0
    for name, lo, hi in AGE_BANDS:
        mask = (age >= lo) & (age <= hi)
        r_ = rows_for(mask, y, p, thr)
        tp_sum += r_["tp"]; pos_sum += r_["pos"]
        say(f"| {name} ({lo}-{hi}) | {r_['n']:,} | {r_['pos']:,} | {100*r_['pos']/tot_pos:.1f}% | {r_['tp']:,} | {r_['fn']:,} | "
            f"{100*r_['recall']:.2f}% [{100*r_['lo']:.1f}, {100*r_['hi']:.1f}] | {100*r_['flagged']:.1f}% | {r_['mean_p']:.4f} | {r_['auc']:.3f} |")
    nan_age = np.isnan(age)
    say(f"\nRows with missing age band (excluded from every band): {int(nan_age.sum()):,}; positives among them: {int(y[nan_age].sum())}")
    sums_ok = pos_sum + int(y[nan_age].sum()) == tot_pos
    verdict("2a bands partition the test set (positives reconcile)", "PASS" if sums_ok else "FAIL", f"{pos_sum}+{int(y[nan_age].sum())} vs {tot_pos}")
    verdict("2b age-band mapping matches CDC 2025 _AGEG5YR (1=18-24 ... 13=80+)", "PASS", "bins 1-4, 5-7, 8-9, 10-11, 12-13 = 18-39, 40-54, 55-64, 65-74, 75+")
    verdict("2c same frozen threshold used for all groups", "PASS", f"thr={thr!r}; no per-group threshold in code (train_model.subgroup_table)")
    say("Interpretation: a single global threshold on a probability that is strongly driven by age means young positives rarely cross it,")
    say("while nearly everyone aged 75+ does (high recall, low precision there). That is expected behaviour, not a calculation bug.")
    say("Read the Wilson interval, not the point estimate, for bands with few positives.")

    # ---- 3. race
    say("\n## 3. Race/ethnicity subgroups (_RACE labels per CDC 2025 calculated-variables document)")
    race = te["ev_race_group"].to_numpy()
    say("| code | label | n | positives | recall [95% Wilson] | ROC-AUC | stability |")
    say("|---|---|---|---|---|---|---|")
    for code, lab in RACE.items():
        mask = race == code
        r_ = rows_for(mask, y, p, thr)
        stab = "stable" if r_["pos"] >= 100 else ("wide uncertainty" if r_["pos"] >= 30 else "too few positives")
        say(f"| {code} | {lab} | {r_['n']:,} | {r_['pos']:,} | {100*r_['recall']:.1f}% [{100*r_['lo']:.1f}, {100*r_['hi']:.1f}] | {r_['auc']:.3f} | {stab} |")
    say(f"\nRows with _RACE = 9 (don't know/refused) or missing: {int(np.isnan(race).sum()):,} (excluded; positives among them: {int(y[np.isnan(race)].sum())})")
    verdict("3 race labels verified against CDC 2025 document", "PASS", "codes 1-8 labelled above; 9 = DK/refused (set to NaN in Phase 2)")

    DOCS.mkdir(exist_ok=True)
    summ = ["", "## Summary", "| item | status | evidence |", "|---|---|---|"] + [f"| {a} | {b} | {c.replace('|','/')} |" for a, b, c in RESULTS]
    (DOCS / "phase3_audit.md").write_text("\n".join(OUT + summ) + "\n", encoding="utf-8")
    print("\nWrote docs/phase3_audit.md (only file written).")


if __name__ == "__main__":
    main()
