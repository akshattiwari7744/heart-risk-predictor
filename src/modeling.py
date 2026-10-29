"""Metrics, thresholding, calibration and the saved FinalModel (importable module, needed to unpickle)."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

EPS = 1e-6
TARGET_DEFINITION = ("Target = self-reported EXISTING coronary heart disease or myocardial infarction (BRFSS _MICHD: "
                     "ever told by a health professional). It is NOT a prediction of a future heart attack and the "
                     "output is NOT a diagnosis.")


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _div(a, b):
    return float(a / b) if b else 0.0


# ----------------------------------------------------------------------------- metrics
def compute_metrics(y, p, thr, sample_weight=None) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    prec, rec, spec = _div(tp, tp + fp), _div(tp, tp + fn), _div(tn, tn + fp)
    f1 = _div(2 * prec * rec, prec + rec)
    f2 = _div(5 * prec * rec, 4 * prec + rec)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = float((tp * tn - fp * fn) / den) if den else 0.0
    two = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p, sample_weight=sample_weight)) if two else float("nan"),
        "roc_auc": float(roc_auc_score(y, p, sample_weight=sample_weight)) if two else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
        "threshold": float(thr), "precision": prec, "recall": rec, "specificity": spec, "f1": f1, "f2": f2,
        "balanced_accuracy": (rec + spec) / 2, "mcc": mcc, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def bootstrap_ci(y, p, thr, n_boot, seed, keys=("pr_auc", "roc_auc", "brier", "precision", "recall",
                                                "specificity", "f1", "balanced_accuracy")):
    y, p = np.asarray(y), np.asarray(p, float)
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        m = compute_metrics(y[i], p[i], thr)
        for k in keys:
            acc[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in acc.items()}


def paired_ap_diff(y, p_a, p_b, n_boot, seed):
    """Bootstrap of AP(a) - AP(b) on the SAME resamples. Returns (point, lo, hi)."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a, float), np.asarray(p_b, float)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(average_precision_score(y[i], p_a[i]) - average_precision_score(y[i], p_b[i]))
    point = average_precision_score(y, p_a) - average_precision_score(y, p_b)
    return float(point), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def choose_threshold(y, p, target_recall):
    """Highest precision subject to recall >= target (ties -> highest threshold). Uses VALIDATION data only."""
    y, p = np.asarray(y), np.asarray(p, float)
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    ok = np.where(rec >= target_recall)[0]
    if len(ok) == 0:
        return float(thr.min()), {"rule": "no threshold reaches target recall; flag all positive"}
    best = ok[prec[ok] == prec[ok].max()]
    k = best[-1]
    return float(thr[k]), {"rule": f"highest precision subject to recall >= {target_recall}",
                           "val_precision_at_threshold": float(prec[k]), "val_recall_at_threshold": float(rec[k])}


# ----------------------------------------------------------------------------- calibration
class Calibrator:
    """Version-independent probability calibrator: 'none' | 'sigmoid' (Platt on logit) | 'isotonic'."""

    def __init__(self, method="none"):
        self.method = method
        self.m = None

    def fit(self, p, y):
        p = np.asarray(p, float)
        if self.method == "sigmoid":
            self.m = LogisticRegression(C=1e10, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return self

    def predict(self, p):
        p = np.asarray(p, float)
        if self.method == "sigmoid":
            return self.m.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        if self.method == "isotonic":
            return self.m.predict(p)
        return np.clip(p, 0, 1)


def oof_calibrated(p, y, method, seed, k=5):
    """Out-of-fold calibrated probabilities on validation (avoids optimistic threshold selection)."""
    p, y = np.asarray(p, float), np.asarray(y)
    out = np.zeros(len(p))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(p, y):
        out[te] = Calibrator(method).fit(p[tr], y[tr]).predict(p[te])
    return out


def select_calibrator(p, y, seed):
    scores = {m: float(brier_score_loss(y, oof_calibrated(p, y, m, seed))) for m in ("none", "sigmoid", "isotonic")}
    return min(scores, key=scores.get), scores


def calibration_slope_intercept(y, p):
    lr = LogisticRegression(C=1e10, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def reliability_table(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "n": int(m.sum()), "mean_predicted": float(p[m].mean()), "observed_rate": float(y[m].mean())})
    return rows


# ----------------------------------------------------------------------------- saved model
class FinalModel:
    """prep + classifier pipeline -> calibrator -> frozen threshold. Needs `modeling` importable to unpickle."""

    def __init__(self, pipeline, calibrator, threshold, feature_names, model_name, meta=None):
        self.pipeline, self.calibrator, self.threshold = pipeline, calibrator, float(threshold)
        self.feature_names, self.model_name, self.meta = list(feature_names), model_name, meta or {}

    def predict_proba_pos(self, X):
        raw = self.pipeline.predict_proba(X[self.feature_names])[:, 1]
        return self.calibrator.predict(raw)

    def predict(self, X):
        return (self.predict_proba_pos(X) >= self.threshold).astype(int)
