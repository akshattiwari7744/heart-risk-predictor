"""
Phase 3 - model training & evaluation ONLY (no API / frontend).

Target: self-reported EXISTING coronary heart disease / MI (_MICHD). Not future heart-attack prediction.

Test-set discipline: `fit_and_decide(train, val, cfg)` has NO access to test data. Decisions are frozen to
artifacts/decisions_frozen.json BEFORE `evaluate_on_test(...)` is called.

    python src\\train_model.py            # full run
    python src\\train_model.py --quick    # fast smoke run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

import config as C
from modeling import (TARGET_DEFINITION, Calibrator, FinalModel, bootstrap_ci, calibration_slope_intercept,
                      choose_threshold, compute_metrics, oof_calibrated, paired_ap_diff, reliability_table,
                      select_calibrator)
from preprocess import build_linear, build_tree, groups

FEATURES = [f["out"] for f in C.FEATURES]
NOMINAL = groups()["nominal"]


@dataclass
class Cfg:
    seed: int = C.SEED
    cv_folds: int = 3
    hgb_iter: int = 10
    tune_rows: int = 120_000
    n_boot: int = 300
    target_recall: float = 0.80
    select_boot: int = 200
    mode: str = "full"          # "full" | "quick"; recorded in every artifact


def quick_cfg() -> Cfg:
    return Cfg(cv_folds=2, hgb_iter=3, tune_rows=30_000, n_boot=100, select_boot=100, mode="quick")


# ----------------------------------------------------------------------------- pipelines
def lr_pipeline(seed, **kw):
    return Pipeline([("prep", build_linear()), ("clf", LogisticRegression(max_iter=2000, random_state=seed, **kw))])


def hgb_pipeline(seed, **kw):
    clf = HistGradientBoostingClassifier(categorical_features=list(NOMINAL), early_stopping=True,
                                         n_iter_no_change=15, random_state=seed, **kw)
    return Pipeline([("prep", build_tree()), ("clf", clf)])


def _strip(params):
    return {k.split("__", 1)[1]: v for k, v in params.items()}


def _subsample(X, y, n, seed):
    if len(X) <= n:
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return Xs, ys


# ----------------------------------------------------------------------------- fit + decide (NO test access)
def fit_and_decide(train: pd.DataFrame, val: pd.DataFrame, cfg: Cfg):
    t0 = time.time()
    Xtr, ytr = train[FEATURES], train["y"].to_numpy()
    Xva, yva = val[FEATURES], val["y"].to_numpy()
    cv = StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    Xs, ys = _subsample(Xtr, ytr, cfg.tune_rows, cfg.seed)

    # baseline
    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    # logistic regression (grid, CV on train subsample, scored by PR-AUC)
    gs = GridSearchCV(lr_pipeline(cfg.seed), {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
                      scoring="average_precision", cv=cv, n_jobs=1, refit=False, error_score="raise").fit(Xs, ys)
    lr_params = _strip(gs.best_params_)
    lr = lr_pipeline(cfg.seed, **lr_params).fit(Xtr, ytr)
    # HistGradientBoosting (random search, CV on train subsample)
    space = {"clf__learning_rate": [0.03, 0.05, 0.1], "clf__max_leaf_nodes": [15, 31, 63],
             "clf__min_samples_leaf": [20, 50, 100, 200], "clf__l2_regularization": [0.0, 0.1, 1.0, 10.0],
             "clf__max_iter": [200, 400], "clf__class_weight": [None, "balanced"]}
    rs = RandomizedSearchCV(hgb_pipeline(cfg.seed), space, n_iter=cfg.hgb_iter, scoring="average_precision", cv=cv,
                            n_jobs=1, refit=False, random_state=cfg.seed, error_score="raise").fit(Xs, ys)
    hgb_params = _strip(rs.best_params_)
    hgb = hgb_pipeline(cfg.seed, **hgb_params).fit(Xtr, ytr)

    # candidate comparison on VALIDATION (raw scores; threshold per candidate by the same rule)
    cands = {"dummy_prior": dummy, "logistic_regression": lr, "hist_gradient_boosting": hgb}
    pv = {k: m.predict_proba(Xva)[:, 1] for k, m in cands.items()}
    cand_val = {}
    for k, p in pv.items():
        thr, info = choose_threshold(yva, p, cfg.target_recall)
        cand_val[k] = compute_metrics(yva, p, thr)
    cand_val["logistic_regression"]["cv_best_pr_auc"] = float(gs.best_score_)
    cand_val["hist_gradient_boosting"]["cv_best_pr_auc"] = float(rs.best_score_)

    # selection: HGB only if it beats LR on validation PR-AUC with a paired-bootstrap CI excluding 0
    pt, lo, hi = paired_ap_diff(yva, pv["hist_gradient_boosting"], pv["logistic_regression"], cfg.select_boot, cfg.seed)
    selected = "hist_gradient_boosting" if lo > 0 else "logistic_regression"
    sel_pipe, sel_params = (hgb, hgb_params) if selected == "hist_gradient_boosting" else (lr, lr_params)
    why = (f"HGB-LR validation PR-AUC difference {pt:+.4f} (95% paired bootstrap CI {lo:+.4f}..{hi:+.4f}); "
           + ("CI excludes 0 in favour of HGB -> HGB selected." if lo > 0 else
              "CI does not show HGB is better -> simpler logistic regression selected."))

    # calibration + threshold on VALIDATION only
    p_raw = pv[selected]
    method, cal_brier = select_calibrator(p_raw, yva, cfg.seed)
    p_oof = oof_calibrated(p_raw, yva, method, cfg.seed)
    thr, thr_info = choose_threshold(yva, p_oof, cfg.target_recall)
    calibrator = Calibrator(method).fit(p_raw, yva)
    p_cal = calibrator.predict(p_raw)
    final_val = compute_metrics(yva, p_oof, thr)           # out-of-fold calibrated -> unbiased-ish
    final_val.update(thr_info)

    meta = {"target_definition": TARGET_DEFINITION, "training_mode": cfg.mode, "seed": cfg.seed, "selected": selected,
            "hyperparameters": sel_params, "calibration_method": method, "target_recall": cfg.target_recall}
    model = FinalModel(sel_pipe, calibrator, thr, FEATURES, selected, meta)
    decisions = {
        "frozen_at": datetime.now().isoformat(timespec="seconds"), "training_mode": cfg.mode, "selected_model": selected,
        "selection_reason": why, "selection_paired_ap_diff": {"point": pt, "ci_lo": lo, "ci_hi": hi},
        "hyperparameters": {"logistic_regression": lr_params, "hist_gradient_boosting": hgb_params},
        "calibration_method": method, "calibration_oof_brier_on_val": cal_brier,
        "threshold": thr, "threshold_rule": thr_info["rule"], "cfg": asdict(cfg),
    }
    arrays = {"y_val": yva, "p_val_raw": pv, "p_val_selected_cal_oof": p_oof, "p_val_selected_cal": p_cal}
    return model, decisions, {"candidates_val": cand_val, "selected_val": final_val, "arrays": arrays,
                              "train_seconds": time.time() - t0}


# ----------------------------------------------------------------------------- test evaluation (after freeze)
def _age_band(a):
    return pd.cut(a, [0, 4, 7, 9, 11, 13], labels=["18-39", "40-54", "55-64", "65-74", "75+"])


def subgroup_table(df, y, p, thr, min_n=200, min_pos=20):
    out = {}
    specs = {"sex": df["sex_female"].map({0.0: "male", 1.0: "female"}),
             "age_band": _age_band(df["age_group_5yr"]).astype("object")}
    if "ev_race_group" in df:
        specs["race_group_code"] = df["ev_race_group"].map(lambda v: None if pd.isna(v) else str(int(v)))
    for name, col in specs.items():
        rows = {}
        for g in sorted(c for c in col.dropna().unique()):
            m = (col == g).to_numpy()
            if m.sum() < min_n or y[m].sum() < min_pos or (1 - y[m]).sum() < min_pos:
                rows[str(g)] = {"n": int(m.sum()), "note": "too few rows/positives for stable metrics"}
                continue
            r = compute_metrics(y[m], p[m], thr)
            rows[str(g)] = {k: r[k] for k in ("n", "prevalence", "pr_auc", "roc_auc", "precision", "recall", "specificity")}
        out[name] = rows
    return out


def evaluate_on_test(model: FinalModel, test: pd.DataFrame, cfg: Cfg):
    yte = test["y"].to_numpy()
    p = model.predict_proba_pos(test[FEATURES])
    m = compute_metrics(yte, p, model.threshold)
    ci = bootstrap_ci(yte, p, model.threshold, cfg.n_boot, cfg.seed)
    slope, icpt = calibration_slope_intercept(yte, p)
    res = {"metrics": m, "ci95": ci, "calibration_slope": slope, "calibration_intercept": icpt,
           "reliability": reliability_table(yte, p), "subgroups": subgroup_table(test, yte, p, model.threshold),
           "prevalence_baseline_pr_auc": float(yte.mean())}
    if "ev_weight" in test and test["ev_weight"].notna().all():
        w = test["ev_weight"].to_numpy()
        wm = compute_metrics(yte, p, model.threshold, sample_weight=w)
        res["weighted_secondary"] = {"pr_auc": wm["pr_auc"], "roc_auc": wm["roc_auc"], "note": "survey-weighted (_LLCPWT); secondary"}
    return res, {"y_test": yte, "p_test": p}


def run(train, val, test, cfg: Cfg):
    model, dec, v = fit_and_decide(train, val, cfg)
    t, ta = evaluate_on_test(model, test, cfg)
    return {"model": model, "decisions": dec, "val": v, "test": t, "test_arrays": ta}


# ----------------------------------------------------------------------------- IO: plots, docs
def _plots(v, ta, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    y = v["arrays"]["y_val"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for k, p in v["arrays"]["p_val_raw"].items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax[0].plot(rc, pr, label=k)
        fpr, tpr, _ = roc_curve(y, p)
        ax[1].plot(fpr, tpr, label=k)
    ax[0].axhline(y.mean(), ls=":", c="gray", label=f"prevalence {y.mean():.3f}")
    ax[0].set(xlabel="recall", ylabel="precision", title="Validation PR curves")
    ax[1].plot([0, 1], [0, 1], ls=":", c="gray")
    ax[1].set(xlabel="false positive rate", ylabel="true positive rate", title="Validation ROC curves")
    ax[0].legend(); ax[1].legend(); fig.tight_layout(); fig.savefig(figs / "val_pr_roc_candidates.png", dpi=130); plt.close(fig)

    yt, pt = ta["y_test"], ta["p_test"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    pr, rc, _ = precision_recall_curve(yt, pt); ax[0].plot(rc, pr); ax[0].axhline(yt.mean(), ls=":", c="gray")
    ax[0].set(xlabel="recall", ylabel="precision", title="Test PR curve (selected, calibrated)")
    fpr, tpr, _ = roc_curve(yt, pt); ax[1].plot(fpr, tpr); ax[1].plot([0, 1], [0, 1], ls=":", c="gray")
    ax[1].set(xlabel="false positive rate", ylabel="true positive rate", title="Test ROC curve")
    fig.tight_layout(); fig.savefig(figs / "test_pr_roc_selected.png", dpi=130); plt.close(fig)


def _plots2(res, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rel = pd.DataFrame(res["test"]["reliability"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rel["mean_predicted"], rel["observed_rate"], "o-"); mx = max(rel["mean_predicted"].max(), rel["observed_rate"].max())
    ax.plot([0, mx], [0, mx], ls=":", c="gray"); ax.set(xlabel="mean predicted probability", ylabel="observed rate", title="Test calibration (deciles)")
    fig.tight_layout(); fig.savefig(figs / "test_calibration.png", dpi=130); plt.close(fig)
    m = res["test"]["metrics"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4.2)); ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"],
           title=f"Test confusion matrix @ thr {m['threshold']:.4f}")
    fig.tight_layout(); fig.savefig(figs / "test_confusion_matrix.png", dpi=130); plt.close(fig)


def _fmt(m, keys):
    return " | ".join(f"{m[k]:.4f}" if isinstance(m[k], float) else f"{m[k]:,}" for k in keys)


# ----------------------------------------------------------------------------- output locations / overwrite protection
@dataclass(frozen=True)
class OutPaths:
    mode: str
    artifacts: Path
    figures: Path
    docs: Path


def output_paths(mode: str, artifacts_base: Path = None, docs_base: Path = None) -> OutPaths:
    """full -> existing production paths; quick -> <production>/quick (separate tree, never the production files)."""
    if mode not in ("full", "quick"):
        raise ValueError(f"mode must be 'full' or 'quick', got {mode!r}")
    ab = Path(artifacts_base) if artifacts_base else C.ARTIFACTS_DIR
    db = Path(docs_base) if docs_base else C.DOCS_DIR
    if mode == "quick":
        ab, db = ab / "quick", db / "quick"
    return OutPaths(mode, ab, ab / "figures", db)


def assert_safe_to_write(paths: OutPaths, artifacts_base: Path = None, docs_base: Path = None, force: bool = False) -> None:
    """Quick may never target production dirs. Full never silently replaces an existing frozen run."""
    prod_a = Path(artifacts_base) if artifacts_base else C.ARTIFACTS_DIR
    prod_d = Path(docs_base) if docs_base else C.DOCS_DIR
    if paths.mode == "quick":
        if paths.artifacts.resolve() == prod_a.resolve() or paths.docs.resolve() == prod_d.resolve():
            raise RuntimeError("Refusing: quick run is pointed at production output directories.")
        return
    frozen = paths.artifacts / "decisions_frozen.json"
    if frozen.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing full-run artifacts ({frozen}). "
                         "They are frozen. Re-run with --force only if you intend to replace them.")


def write_docs(res, cfg, meta, docs_dir=None):
    v, d, t = res["val"], res["decisions"], res["test"]
    cols = ["pr_auc", "roc_auc", "precision", "recall", "specificity", "f1", "balanced_accuracy"]
    R = ["# Phase 3 Report - model training & evaluation", f"_Generated {meta['created']}. All numbers come from this run._", "",
         f"> **{TARGET_DEFINITION}**", "",
         "## Data", f"Train {meta['n_train']:,} / validation {meta['n_val']:,} / test {meta['n_test']:,} rows (Phase 2 stratified split, seed {cfg.seed}). "
         f"Prevalence: train {100*meta['prev_train']:.2f}%, val {100*meta['prev_val']:.2f}%, test {100*meta['prev_test']:.2f}%.",
         "", "## Categorical handling",
         "- Nominal (`diabetes_status`, `smoker_status`): one-hot in logistic regression; native categorical in HistGradientBoosting. Codes are never treated as magnitudes.",
         "- Ordinal bands (`age_group_5yr`, `gen_health`, `education_level`, `income_level`): ordered numeric (monotone bands); HGB can learn non-linearity, LR gets a linear trend.",
         "- Binary features are 0/1; missing values get imputation + missing-indicator flags (LR) or native NaN (HGB). All fitted on training folds only.",
         "", "## Validation comparison (threshold per candidate = highest precision with recall >= "
         f"{cfg.target_recall}, chosen on validation; optimistic for all candidates equally)",
         "| model | " + " | ".join(cols) + " | CV PR-AUC (train) |", "|---|" + "---|" * (len(cols) + 1)]
    for k, m in v["candidates_val"].items():
        R.append(f"| {k} | {_fmt(m, cols)} | {m.get('cv_best_pr_auc', float('nan')):.4f} |")
    R += [f"\nPR-AUC prevalence baseline (validation): **{v['candidates_val']['dummy_prior']['prevalence']:.4f}**",
          "", "### Confusion matrices (validation, per candidate)", "| model | TN | FP | FN | TP |", "|---|---|---|---|---|"]
    for k, m in v["candidates_val"].items():
        R.append(f"| {k} | {m['tn']:,} | {m['fp']:,} | {m['fn']:,} | {m['tp']:,} |")
    R += ["", "## Decisions (frozen before the test set was scored)", f"- Selected: **{d['selected_model']}**", f"- {d['selection_reason']}",
          f"- Hyperparameters: `{json.dumps(d['hyperparameters'][d['selected_model']])}`",
          f"- Calibration: **{d['calibration_method']}** (out-of-fold validation Brier by method: `{json.dumps(d['calibration_oof_brier_on_val'])}`)",
          f"- Threshold **{d['threshold']:.5f}**; rule: {d['threshold_rule']}",
          "", "## Selected model on validation (out-of-fold calibrated)", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols),
          "| " + _fmt(v["selected_val"], cols) + " |",
          "", "## FINAL TEST RESULTS (scored once, after freeze)", "| metric | value | 95% bootstrap CI |", "|---|---|---|"]
    m = t["metrics"]
    for k in ["pr_auc", "roc_auc", "brier", "precision", "recall", "specificity", "f1", "balanced_accuracy"]:
        lo, hi = t["ci95"][k]
        R.append(f"| {k} | {m[k]:.4f} | {lo:.4f} - {hi:.4f} |")
    R += [f"| F2 / MCC | {m['f2']:.4f} / {m['mcc']:.4f} | - |",
          f"\nTest PR-AUC prevalence baseline: **{t['prevalence_baseline_pr_auc']:.4f}**. Confusion matrix: TN {m['tn']:,}, FP {m['fp']:,}, FN {m['fn']:,}, TP {m['tp']:,}.",
          f"Calibration slope {t['calibration_slope']:.3f}, intercept {t['calibration_intercept']:.3f} (ideal 1 / 0)."]
    if "weighted_secondary" in t:
        w = t["weighted_secondary"]
        R.append(f"Survey-weighted (secondary): PR-AUC {w['pr_auc']:.4f}, ROC-AUC {w['roc_auc']:.4f}.")
    R += ["", "## Subgroups (test, frozen threshold)"]
    for name, rows in t["subgroups"].items():
        R += [f"**{name}**", "| group | n | prevalence | PR-AUC | ROC-AUC | precision | recall | specificity |", "|---|---|---|---|---|---|---|---|"]
        for g, r in rows.items():
            R.append(f"| {g} | {r['n']:,} | " + (" | ".join(f"{r[k]:.4f}" for k in ("prevalence", "pr_auc", "roc_auc", "precision", "recall", "specificity")) if "pr_auc" in r else f"{r['note']} | | | | | ") + " |")
        R.append("")
    R += ["## Figures", "`artifacts/figures/`: val_pr_roc_candidates.png, test_pr_roc_selected.png, test_calibration.png, test_confusion_matrix.png"]
    docs_dir = Path(docs_dir) if docs_dir else C.DOCS_DIR
    docs_dir.mkdir(parents=True, exist_ok=True)
    R.insert(2, f"**Training mode: {cfg.mode.upper()}**" + (" - SMOKE RUN, reduced tuning. NOT final results; do not cite." if cfg.mode == "quick" else ""))
    (docs_dir / "phase3_report.md").write_text("\n".join(R) + "\n", encoding="utf-8")

    card = ["# Model Card - BRFSS 2025 Heart Disease Reporting Model", "",
            "## What this model predicts", TARGET_DEFINITION, "",
            "It estimates the likelihood that a respondent *has reported* coronary heart disease or a heart attack, from "
            "self-reported demographics, lifestyle and chronic-condition answers. Because the data are cross-sectional, it "
            "shows association, not cause or future risk, and must not be used to diagnose, triage, or exclude disease.", "",
            "## Data", "CDC BRFSS 2025 (LLCP2025), adults, US telephone survey; self-reported. "
            f"{meta['n_train']+meta['n_val']+meta['n_test']:,} rows with a valid target.", "",
            "## Model", f"{d['selected_model']}, calibrated with {d['calibration_method']}; decision threshold {d['threshold']:.5f} "
            f"(rule: {d['threshold_rule']}). Inputs ({len(FEATURES)}): " + ", ".join(FEATURES) + ".", "",
            "## Test performance (single evaluation)", f"PR-AUC {m['pr_auc']:.4f} (prevalence baseline {t['prevalence_baseline_pr_auc']:.4f}); ROC-AUC {m['roc_auc']:.4f}; "
            f"recall {m['recall']:.4f}; precision {m['precision']:.4f}; specificity {m['specificity']:.4f}. See docs/phase3_report.md for CIs and subgroups.", "",
            "## Limitations", "- Self-reported labels and inputs; recall/misreporting bias.", "- US survey; may not transfer to other populations (e.g. India).",
            "- Existing-condition target: several inputs (e.g. general health, hypertension) can be consequences of heart disease, not only risk factors.",
            "- Race is excluded from inputs and used only for subgroup evaluation; subgroup performance can differ.",
            "- Not a medical device."]
    if cfg.mode == "quick":
        card.insert(1, "> **QUICK SMOKE RUN - not a final model. Do not cite or deploy.**")
    (docs_dir / "model_card.md").write_text("\n".join(card) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main(argv=None, cfg=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke run; writes ONLY to artifacts/quick and docs/quick; NOT for final numbers")
    ap.add_argument("--force", action="store_true", help="allow a FULL run to replace existing frozen full-run artifacts")
    ap.add_argument("--data-dir", default=str(C.PROCESSED_DIR))
    ap.add_argument("--artifacts-dir", default=str(C.ARTIFACTS_DIR), help="production artifacts base (override for tests)")
    ap.add_argument("--docs-dir", default=str(C.DOCS_DIR), help="production docs base (override for tests)")
    a = ap.parse_args(argv)
    mode = "quick" if a.quick else "full"
    base_cfg = cfg if cfg is not None else (quick_cfg() if a.quick else Cfg())
    cfg = replace(base_cfg, mode=mode)
    P = output_paths(mode, a.artifacts_dir, a.docs_dir)
    assert_safe_to_write(P, a.artifacts_dir, a.docs_dir, a.force)          # BEFORE anything is trained or written
    dd = Path(a.data_dir)
    train, val, test = (pd.read_parquet(dd / f"{n}.parquet") for n in ("train", "val", "test"))
    P.figures.mkdir(parents=True, exist_ok=True)
    P.docs.mkdir(parents=True, exist_ok=True)
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}  (mode={mode})")
    print(f"Outputs -> {P.artifacts} and {P.docs}")
    print("Fitting / tuning on TRAIN, selecting + calibrating + thresholding on VALIDATION ...")
    model, dec, v = fit_and_decide(train, val, cfg)
    (P.artifacts / "decisions_frozen.json").write_text(json.dumps(dec, indent=2), encoding="utf-8")
    print(f"Decisions frozen: {dec['selected_model']} | calibration={dec['calibration_method']} | threshold={dec['threshold']:.5f}")
    print("Scoring TEST once ...")
    t, ta = evaluate_on_test(model, test, cfg)
    res = {"model": model, "decisions": dec, "val": v, "test": t}

    mp = P.artifacts / "model.joblib"
    joblib.dump(model, mp, compress=3)
    meta = {"created": datetime.now().isoformat(timespec="seconds"), "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "prev_train": float(train["y"].mean()), "prev_val": float(val["y"].mean()), "prev_test": float(test["y"].mean())}
    metrics = {"training_mode": mode, "target_definition": TARGET_DEFINITION, "validation_candidates": v["candidates_val"],
               "validation_selected_oof_calibrated": v["selected_val"], "test": t, "decisions": dec, "data": meta}
    (P.artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    (P.artifacts / "model_meta.json").write_text(json.dumps({
        **meta, "training_mode": mode, "output_dir": str(P.artifacts),
        "target_definition": TARGET_DEFINITION, "selected_model": dec["selected_model"], "features": FEATURES,
        "feature_sources": {f["out"]: f["src"] for f in C.FEATURES}, "threshold": model.threshold,
        "calibration": dec["calibration_method"], "seed": cfg.seed, "model_sha256": sha256(mp), "train_seconds": v["train_seconds"],
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__, "joblib": joblib.__version__},
        "load_note": "joblib.load needs src/ on sys.path (class FinalModel lives in modeling.py) and the same library versions."},
        indent=2), encoding="utf-8")
    _plots(v, ta, P.figures)
    _plots2(res, P.figures)
    write_docs(res, cfg, meta, P.docs)
    m = t["metrics"]
    print(f"\nTEST (frozen, {mode} run): PR-AUC {m['pr_auc']:.4f} (baseline {m['prevalence']:.4f}) | ROC-AUC {m['roc_auc']:.4f} | "
          f"recall {m['recall']:.4f} | precision {m['precision']:.4f} | specificity {m['specificity']:.4f}")
    print(f"Wrote model.joblib, metrics.json, model_meta.json, decisions_frozen.json, figures/ under {P.artifacts}; reports under {P.docs}")


if __name__ == "__main__":
    sys.exit(main())
