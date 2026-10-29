import numpy as np
import pandas as pd
import pytest

import config as C
from cleaning import UnmappedCodeError, assert_no_leakage, recode, resolve_source
from preprocess import build_linear, groups

feat = {f["out"]: f for f in C.FEATURES}


def test_days_rule_keeps_real_7_and_9():
    s = pd.Series([1, 7, 9, 30, 88, 77, 99, np.nan], dtype=float)
    out = recode(s, feat["phys_unhealthy_days"]).tolist()
    assert out[:4] == [1, 7, 9, 30] and out[4] == 0
    assert np.isnan(out[5]) and np.isnan(out[6]) and np.isnan(out[7])
    with pytest.raises(UnmappedCodeError):       # 0 is not a valid raw code (none is coded 88)
        recode(pd.Series([0.0]), feat["phys_unhealthy_days"])


def test_bmi_scale_and_range_guard():
    assert recode(pd.Series([1202.0, 9843.0]), feat["bmi"]).tolist() == [pytest.approx(12.02), pytest.approx(98.43)]
    with pytest.raises(UnmappedCodeError):
        recode(pd.Series([50.0]), feat["bmi"])


def test_hypertension_rule_matches_cdc_rfhype6():
    out = recode(pd.Series([1, 2, 3, 4, 7, 9], dtype=float), feat["hypertension_told"]).tolist()
    assert out[:4] == [1, 0, 0, 0] and np.isnan(out[4]) and np.isnan(out[5])


def test_unmapped_code_aborts():
    with pytest.raises(UnmappedCodeError):
        recode(pd.Series([1.0, 2.0, 5.0]), feat["copd_ever"])


def test_leakage_guard_and_config_clean():
    with pytest.raises(AssertionError):
        assert_no_leakage(["BMI", "CVDINFR4"])
    srcs = {f["src"] for f in C.FEATURES}
    assert not (srcs & set(C.FORBIDDEN_SOURCES))
    assert 15 <= len(C.FEATURES) <= 25


def test_pattern_resolution():
    assert resolve_source("_RFDRHV*", ["A", "_RFDRHV9"]) == "_RFDRHV9"
    with pytest.raises(KeyError):
        resolve_source("_RFDRHV*", ["A"])


def test_preprocessor_fits_on_train_only():
    rng = np.random.default_rng(0)
    g = groups()
    cols = [c for v in g.values() for c in v]
    tr = pd.DataFrame(rng.integers(1, 4, (200, len(cols))).astype(float), columns=cols)
    te = tr.iloc[:20].copy() + 1000.0          # extreme values must not influence the fit
    lin = build_linear().fit(tr)
    sc = lin.named_transformers_["num"].named_steps["sc"]
    scaled = g["continuous"] + g["ordinal"]
    assert np.allclose(sc.mean_, tr[scaled].mean().to_numpy())
    assert not lin.transform(te).isna().any().any()
