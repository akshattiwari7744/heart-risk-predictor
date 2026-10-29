"""Fit-on-train-only preprocessors: one for linear models, one for tree models."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from config import FEATURES


def groups():
    g = {"continuous": [], "ordinal": [], "binary": [], "nominal": []}
    for f in FEATURES:
        g[f["kind"]].append(f["out"])
    return g


def build_linear():
    g = groups()
    scaled = g["continuous"] + g["ordinal"]
    # scaler first (NaN-safe), then median impute + missing-indicator flags
    num = Pipeline([("sc", StandardScaler()), ("imp", SimpleImputer(strategy="median", add_indicator=True))])
    bina = SimpleImputer(strategy="most_frequent", add_indicator=True)
    nomi = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=-1)),
                     ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    ct = ColumnTransformer([("num", num, scaled), ("bin", bina, g["binary"]), ("nom", nomi, g["nominal"])],
                           remainder="drop", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def build_tree():
    g = groups()
    keep = g["continuous"] + g["ordinal"] + g["binary"]
    nomi = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=np.nan)
    ct = ColumnTransformer([("keep", "passthrough", keep), ("nom", nomi, g["nominal"])],
                           remainder="drop", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")
