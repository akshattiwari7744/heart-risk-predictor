"""Stateless recoding of raw BRFSS codes. Learns nothing from data, so it is safe to run before splitting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEATURES, EVAL_ONLY, FORBIDDEN_SOURCES, TARGET_SRC, WEIGHT_SRC


class UnmappedCodeError(ValueError):
    pass


def resolve_source(src: str, columns) -> str:
    """Resolve a spec source name (supports trailing '*') to one actual column name."""
    if not src.endswith("*"):
        if src not in columns:
            raise KeyError(f"Required source column not found in data: {src}")
        return src
    hits = sorted(c for c in columns if c.upper().startswith(src[:-1].upper()))
    if len(hits) != 1:
        raise KeyError(f"Pattern {src} matched {len(hits)} columns: {hits} (need exactly 1)")
    return hits[0]


def assert_no_leakage(source_columns) -> None:
    bad = [c for c in source_columns if c in FORBIDDEN_SOURCES]
    if bad:
        raise AssertionError(f"Forbidden columns used as model inputs: {bad}")


def _observed(s: pd.Series):
    return sorted(set(float(v) for v in s.dropna().unique()))


def recode(s: pd.Series, feat: dict, report: dict | None = None) -> pd.Series:
    """Apply one feature rule. Raises UnmappedCodeError if an observed code is not covered."""
    s = pd.to_numeric(s, errors="coerce")
    rule = feat["rule"]
    if rule == "map":
        codes = feat["codes"]
        unk = [v for v in _observed(s) if v not in codes]
        if unk:
            raise UnmappedCodeError(f"{feat['src']}: observed codes not in map: {unk}")
        out = s.map(codes)
    elif rule == "days":      # 1-30 real counts; 88 -> 0 (none); 77/99 -> NaN
        unk = [v for v in _observed(s) if not (1 <= v <= 30 or v in (77, 88, 99))]
        if unk:
            raise UnmappedCodeError(f"{feat['src']}: unexpected values: {unk}")
        out = s.where(~s.isin([77, 99]), np.nan)
        out = out.where(out != 88, 0.0)
    elif rule == "bmi":       # stored x100; codebook range 12.00-99.99
        lo, hi = s.min(), s.max()
        if pd.notna(lo) and (lo < 1200 or hi > 9999):
            raise UnmappedCodeError(f"{feat['src']}: values outside 1200-9999 (min={lo}, max={hi})")
        out = s / 100.0
    else:
        raise ValueError(f"Unknown rule {rule}")
    return out.astype("float32")


def clean_frame(raw: pd.DataFrame):
    """Return (X, eval_df, source_map). X has only model inputs, renamed to out-names."""
    cols = list(raw.columns)
    src_map = {f["out"]: resolve_source(f["src"], cols) for f in FEATURES}
    assert_no_leakage(src_map.values())
    X = pd.DataFrame({f["out"]: recode(raw[src_map[f["out"]]], f) for f in FEATURES}, index=raw.index)
    ev = {}
    for f in EVAL_ONLY:
        ev[f["out"]] = recode(raw[resolve_source(f["src"], cols)], f)
    if WEIGHT_SRC in cols:
        ev["weight"] = pd.to_numeric(raw[WEIGHT_SRC], errors="coerce")
    ev_df = pd.DataFrame(ev, index=raw.index)
    assert TARGET_SRC not in X.columns
    return X, ev_df, src_map
