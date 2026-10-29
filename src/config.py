"""
Phase 2 configuration - single source of truth for features, code maps, exclusions.

Evidence tags (no variable is assumed without one):
  CDC25_CALC   : coding verified in CDC "Calculated Variables in the 2025 BRFSS Data File" (v5, Jun 26 2026)
  QUEST_2021_22: coding taken from the 2021/2022 BRFSS questionnaire (2025 value table not retrieved)
  Q2025_NC     : response codes read in the North Carolina 2025 BRFSS questionnaire (core Section 7 is
                 CDC-mandated and identical across states); the PDF prints question text, not SAS names, so the
                 question->variable link is by question text and matches CDC 2022 questionnaire / WI codebook labels.
                 Enforced at runtime: any observed code NOT in the map aborts the build.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUBSET_PATH = ROOT / "data" / "interim" / "brfss_phase1_subset.parquet"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
DOCS_DIR = ROOT / "docs"

SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15
EXPECTED_ROWS = 356_158
EXPECTED_VALID = 352_145

TARGET_SRC = "_MICHD"
NAN = np.nan
YN = {1: 1, 2: 0, 7: NAN, 9: NAN}          # 1 Yes, 2 No, 7 DK, 9 Refused
ID = lambda *ks: {k: k for k in ks}        # noqa: E731  identity map helper


def F(out, src, kind, rule, codes=None, ev="", label="", note=""):
    return dict(out=out, src=src, kind=kind, rule=rule, codes=codes or {}, evidence=ev, label=label, note=note)


# kind: continuous | ordinal | binary | nominal  (drives the imputer/encoder used)
FEATURES = [
    F("age_group_5yr", "_AGEG5YR", "ordinal", "map", {**ID(*range(1, 14)), 14: NAN}, "CDC25_CALC",
      "Age group, 13 five-year bands (1=18-24 ... 13=80+); 14=DK/refused/missing -> NaN",
      "AGE itself is not in the public file; _AGEG5YR is derived from it"),
    F("sex_female", "_SEX", "binary", "map", {1: 0, 2: 1}, "CDC25_CALC", "Calculated sex: 1 male -> 0, 2 female -> 1"),
    F("bmi", "_BMI5", "continuous", "bmi", None, "CDC25_CALC",
      "BMI stored x100 (codebook range 12.00-99.99) -> divide by 100"),
    F("gen_health", "GENHLTH", "ordinal", "map", {**ID(1, 2, 3, 4, 5), 7: NAN, 9: NAN}, "CDC25_CALC",
      "General health 1 excellent .. 5 poor"),
    F("phys_unhealthy_days", "PHYSHLTH", "continuous", "days", None, "CDC25_CALC",
      "Days physical health not good, 0-30; 88=none -> 0; 77/99 -> NaN",
      "literal 7 and 9 are REAL day counts, not DK/refused"),
    F("ment_unhealthy_days", "MENTHLTH", "continuous", "days", None, "CDC25_CALC",
      "Days mental health not good, 0-30; 88=none -> 0; 77/99 -> NaN",
      "literal 7 and 9 are REAL day counts, not DK/refused"),
    F("any_physical_activity", "_TOTINDA", "binary", "map", {1: 1, 2: 0, 9: NAN}, "CDC25_CALC",
      "Any leisure-time activity/exercise past 30 days"),
    F("hypertension_told", "BPHIGH6", "binary", "map", {1: 1, 2: 0, 3: 0, 4: 0, 7: NAN, 9: NAN}, "CDC25_CALC",
      "Told high BP (same rule CDC uses for _RFHYPE6: 2 pregnancy-only/3 no/4 borderline -> 0)"),
    F("high_cholesterol_told", "TOLDHI3", "binary", "map", YN, "CDC25_CALC",
      "Told cholesterol high", "possible skip pattern (asked only if cholesterol checked) - NaN handled by indicator"),
    F("diabetes_status", "DIABETE4", "nominal", "map", {1: 1, 2: 2, 3: 3, 4: 4, 7: NAN, 9: NAN}, "Q2025_NC",
      "1 Yes; 2 Yes but female told only during pregnancy; 3 No; 4 No, pre-diabetes or borderline diabetes; 7/9 -> NaN",
      "nominal (one-hot / native categorical); codes are NOT ordered numbers"),
    F("asthma_ever", "ASTHMA3", "binary", "map", YN, "CDC25_CALC", "Ever told asthma"),
    F("copd_ever", "CHCCOPD3", "binary", "map", YN, "Q2025_NC", "Ever told COPD, emphysema or chronic bronchitis"),
    F("kidney_disease_ever", "CHCKDNY2", "binary", "map", YN, "Q2025_NC", "Ever told kidney disease (not counting kidney stones, bladder infection, incontinence)"),
    F("depression_ever", "ADDEPEV3", "binary", "map", YN, "Q2025_NC", "Ever told depressive disorder (incl. depression, major depression, dysthymia, minor depression)"),
    F("other_cancer_ever", "CHCOCNC1", "binary", "map", YN, "Q2025_NC",
      "Ever told melanoma or any other type of cancer (non-melanoma skin cancer is a separate item, CHCSCNC1)",
      "column name kept from Phase 2 for compatibility"),
    F("smoker_status", "_SMOKER3", "nominal", "map", {**ID(1, 2, 3, 4), 9: NAN}, "CDC25_CALC",
      "1 daily, 2 some days, 3 former, 4 never; 9 -> NaN",
      "replaces SMOKE100+SMOKDAY2 (skip pattern is already resolved inside _SMOKER3)"),
    F("heavy_drinker", "_RFDRHV*", "binary", "map", {1: 0, 2: 1, 9: NAN}, "CDC25_CALC",
      "Heavy drinker flag (column name suffix resolved at runtime, e.g. _RFDRHV9)"),
    F("education_level", "_EDUCAG", "ordinal", "map", {**ID(1, 2, 3, 4), 9: NAN}, "CDC25_CALC", "Education 1-4"),
    F("income_level", "_INCOMG1", "ordinal", "map", {**ID(1, 2, 3, 4, 5, 6, 7), 9: NAN}, "CDC25_CALC", "Income 1-7"),
]

# Kept ONLY for subgroup evaluation / weighted metrics - never model inputs.
EVAL_ONLY = [
    F("race_group", "_RACE", "nominal", "map", {**ID(1, 2, 3, 4, 5, 6, 7, 8), 9: NAN}, "CDC25_CALC",
      "Race/ethnicity 1-8; eval-only (fairness-sensitive)"),
]
WEIGHT_SRC = "_LLCPWT"

# Never allowed as a model input source column (checked by assert_no_leakage).
FORBIDDEN_SOURCES = {
    "_MICHD": "target", "CVDINFR4": "direct source of target", "CVDCRHD4": "direct source of target",
    "CVDSTRK3": "related CV outcome", "BPMEDS1": "treatment proxy", "CHOLMED3": "treatment proxy",
    "CHECKUP1": "healthcare-contact proxy", "PERSDOC3": "healthcare-contact proxy",
    "MEDCOST1": "healthcare-contact proxy", "_HLTHPL2": "healthcare-contact proxy",
}

# Documented non-inputs (for feature_spec_v1.csv)
OTHER_EXCLUSIONS = {
    "AGE": "not present in the public file (confirmed by Phase 1); use _AGEG5YR",
    "_AGE65YR": "coarser duplicate of _AGEG5YR",
    "SEXVAR": "redundant with _SEX",
    "EXERANY2": "source of _TOTINDA (duplicate)",
    "SMOKE100": "component of _SMOKER3",
    "SMOKDAY2": "component of _SMOKER3; structurally missing for never-smokers (skip pattern)",
    "_IMPRACE": "imputed race; not used",
    "MARITAL": "2025 codes not verified; excluded from v1",
    "EMPLOY1": "2025 codes not verified; excluded from v1",
    "FRNCHDA_": "French-fry intake intermediate (from FRENCHF1); unrelated to heart disease - name-scan false "
                "positive ('CHD' inside FRNCHDA_)",
    "_STATE/_PSU/_STSTR/FMONTH/IDATE/IMONTH/IDAY/IYEAR/DISPCODE/SEQNO": "design/administrative fields",
}
