import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C
import train_model as T
from modeling import FinalModel, choose_threshold, compute_metrics

QUICK = T.Cfg(cv_folds=2, hgb_iter=2, tune_rows=1500, n_boot=20, select_boot=20)


def make_frame(n, seed):
    rng = np.random.default_rng(seed)
    d = {}
    for f in C.FEATURES:
        vals = sorted({v for v in f["codes"].values() if not (isinstance(v, float) and np.isnan(v))}) if f["codes"] else None
        if f["rule"] == "bmi":
            col = rng.normal(28, 5, n)
        elif f["rule"] == "days":
            col = rng.choice([0, 1, 5, 14, 30], n).astype(float)
        else:
            col = rng.choice(vals, n).astype(float)
        col = np.where(rng.random(n) < 0.03, np.nan, col)
        d[f["out"]] = col
    df = pd.DataFrame(d)
    z = 0.4 * (df["age_group_5yr"].fillna(7) - 7) + 1.0 * (df["hypertension_told"].fillna(0)) - 3.0
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    df["ev_race_group"] = rng.integers(1, 5, n).astype(float)
    df["ev_weight"] = rng.random(n) * 100 + 1
    return df


@pytest.fixture(scope="module")
def frames():
    return make_frame(3000, 1), make_frame(1500, 2), make_frame(1500, 3)


def test_threshold_rule_toy():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([.9, .8, .7, .3, .6, .2, .15, .1, .05, .01])
    thr, _ = choose_threshold(y, p, 0.75)
    m = compute_metrics(y, p, thr)
    assert m["recall"] >= 0.75 and thr == pytest.approx(0.7)


def test_metrics_confusion_counts():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([.9, .2, .8, .1, .1])
    m = compute_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == pytest.approx(0.5) and m["specificity"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 2 / 3) / 2)


def test_feature_list_is_leakage_safe():
    assert not (set(T.FEATURES) & {"y", "row_id", "ev_race_group", "ev_weight"})
    assert not ({f["src"] for f in C.FEATURES} & set(C.FORBIDDEN_SOURCES))


def test_nominal_features_not_scaled_and_hgb_categorical():
    lin = T.build_linear()
    by_name = {n: t for n, t, cols in lin.transformers}
    nominal_cols = [c for n, t, cols in lin.transformers if n == "nom" for c in cols]
    scaled_cols = [c for n, t, cols in lin.transformers if n == "num" for c in cols]
    assert set(nominal_cols) == {"diabetes_status", "smoker_status"} and not set(nominal_cols) & set(scaled_cols)
    assert any(isinstance(s[1], OneHotEncoder) for s in by_name["nom"].steps)
    assert any(isinstance(s[1], StandardScaler) for s in by_name["num"].steps)
    assert T.hgb_pipeline(0).named_steps["clf"].categorical_features == ["diabetes_status", "smoker_status"]


def test_decisions_independent_of_test_labels(frames):
    tr, va, te = frames
    a = T.run(tr, va, te, QUICK)
    te2 = te.copy()
    te2["y"] = np.random.default_rng(99).permutation(te2["y"].to_numpy())     # scramble test labels
    b = T.run(tr, va, te2, QUICK)
    assert a["decisions"]["selected_model"] == b["decisions"]["selected_model"]
    assert a["decisions"]["threshold"] == b["decisions"]["threshold"]
    assert a["decisions"]["hyperparameters"] == b["decisions"]["hyperparameters"]
    assert a["val"]["selected_val"]["pr_auc"] == b["val"]["selected_val"]["pr_auc"]
    assert a["test"]["metrics"]["pr_auc"] != b["test"]["metrics"]["pr_auc"]      # only test results change


def test_fit_and_decide_has_no_test_argument():
    import inspect
    assert list(inspect.signature(T.fit_and_decide).parameters) == ["train", "val", "cfg"]


def test_deterministic_and_roundtrip(frames, tmp_path):
    tr, va, te = frames
    m1, d1, _ = T.fit_and_decide(tr, va, QUICK)
    m2, d2, _ = T.fit_and_decide(tr, va, QUICK)
    assert d1["threshold"] == d2["threshold"] and d1["selected_model"] == d2["selected_model"]
    p1 = m1.predict_proba_pos(te)
    joblib.dump(m1, tmp_path / "m.joblib")
    m3 = joblib.load(tmp_path / "m.joblib")
    assert isinstance(m3, FinalModel)
    assert np.allclose(p1, m3.predict_proba_pos(te)) and p1.min() >= 0 and p1.max() <= 1
    assert set(np.unique(m3.predict(te))) <= {0, 1}
