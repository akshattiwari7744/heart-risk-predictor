"""Request/response models. Request fields = the model's 19 inputs, in the exact order of src/config.py FEATURES."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# name -> (kind, low, high, plain-language meaning). Order MUST equal config.FEATURES (verified at startup and in tests).
FIELD_HELP: Dict[str, dict] = {}


def _h(name, kind, lo, hi, text):
    FIELD_HELP[name] = {"kind": kind, "min": lo, "max": hi, "description": text}


_h("age_group_5yr", "ordinal", 1, 13, "Age band: 1=18-24, 2=25-29, 3=30-34, 4=35-39, 5=40-44, 6=45-49, 7=50-54, 8=55-59, 9=60-64, 10=65-69, 11=70-74, 12=75-79, 13=80 or older.")
_h("sex_female", "binary", 0, 1, "Sex as calculated by BRFSS: 0=male, 1=female.")
_h("bmi", "continuous", 12.0, 99.99, "Body mass index in kg/m2 (BRFSS valid range 12.00-99.99).")
_h("gen_health", "ordinal", 1, 5, "Self-rated general health: 1=excellent, 2=very good, 3=good, 4=fair, 5=poor.")
_h("phys_unhealthy_days", "continuous", 0, 30, "Days in the past 30 when physical health was not good (0-30).")
_h("ment_unhealthy_days", "continuous", 0, 30, "Days in the past 30 when mental health was not good (0-30).")
_h("any_physical_activity", "binary", 0, 1, "1=did any leisure-time physical activity or exercise in the past 30 days, 0=none.")
_h("hypertension_told", "binary", 0, 1, "1=told by a health professional that blood pressure is high; 0=not told (includes borderline and pregnancy-only).")
_h("high_cholesterol_told", "binary", 0, 1, "1=told cholesterol is high, 0=not told it is high.")
_h("diabetes_status", "nominal", 1, 4, "Category, not a number: 1=yes; 2=yes, but only during pregnancy (female); 3=no; 4=no, pre-diabetes or borderline.")
_h("asthma_ever", "binary", 0, 1, "1=ever told by a health professional of asthma, 0=no.")
_h("copd_ever", "binary", 0, 1, "1=ever told of COPD, emphysema or chronic bronchitis, 0=no.")
_h("kidney_disease_ever", "binary", 0, 1, "1=ever told of kidney disease (not counting kidney stones, bladder infection or incontinence), 0=no.")
_h("depression_ever", "binary", 0, 1, "1=ever told of a depressive disorder, 0=no.")
_h("other_cancer_ever", "binary", 0, 1, "1=ever told of melanoma or any other cancer (non-melanoma skin cancer is a separate item), 0=no.")
_h("smoker_status", "nominal", 1, 4, "Category, not a number: 1=smokes every day, 2=smokes some days, 3=former smoker, 4=never smoked (BRFSS _SMOKER3).")
_h("heavy_drinker", "binary", 0, 1, "1=heavy drinker (men over 14, women over 7 drinks per week), 0=no.")
_h("education_level", "ordinal", 1, 4, "1=did not graduate high school, 2=graduated high school, 3=attended college/technical school, 4=graduated college/technical school.")
_h("income_level", "ordinal", 1, 7, "Household income band: 1=<$15k, 2=$15-25k, 3=$25-35k, 4=$35-50k, 5=$50-100k, 6=$100-200k, 7=$200k+.")

_EXAMPLE = {"age_group_5yr": 7, "sex_female": 0, "bmi": 27.5, "gen_health": 3, "phys_unhealthy_days": 0, "ment_unhealthy_days": 0,
            "any_physical_activity": 1, "hypertension_told": 0, "high_cholesterol_told": 0, "diabetes_status": 3,
            "asthma_ever": 0, "copd_ever": 0, "kidney_disease_ever": 0, "depression_ever": 0, "other_cancer_ever": 0,
            "smoker_status": 4, "heavy_drinker": 0, "education_level": 3, "income_level": 5}


def _f(name):
    h = FIELD_HELP[name]
    kw = dict(description=h["description"])
    if h["kind"] == "continuous" and name == "bmi":
        return Field(..., ge=h["min"], le=h["max"], allow_inf_nan=False, **kw)
    return Field(..., ge=h["min"], le=h["max"], **kw)


class PredictionRequest(BaseModel):
    """All 19 inputs are required. Strict typing: numbers only (no strings, booleans or null). Unknown fields are rejected."""
    model_config = ConfigDict(extra="forbid", strict=True,
                              json_schema_extra={"examples": [_EXAMPLE], "description": "ILLUSTRATIVE example values only - not a typical or recommended profile."})
    age_group_5yr: int = _f("age_group_5yr")
    sex_female: int = _f("sex_female")
    bmi: float = _f("bmi")
    gen_health: int = _f("gen_health")
    phys_unhealthy_days: int = _f("phys_unhealthy_days")
    ment_unhealthy_days: int = _f("ment_unhealthy_days")
    any_physical_activity: int = _f("any_physical_activity")
    hypertension_told: int = _f("hypertension_told")
    high_cholesterol_told: int = _f("high_cholesterol_told")
    diabetes_status: int = _f("diabetes_status")
    asthma_ever: int = _f("asthma_ever")
    copd_ever: int = _f("copd_ever")
    kidney_disease_ever: int = _f("kidney_disease_ever")
    depression_ever: int = _f("depression_ever")
    other_cancer_ever: int = _f("other_cancer_ever")
    smoker_status: int = _f("smoker_status")
    heavy_drinker: int = _f("heavy_drinker")
    education_level: int = _f("education_level")
    income_level: int = _f("income_level")


FEATURE_ORDER: List[str] = list(PredictionRequest.model_fields)


class ModelBrief(BaseModel):
    model_name: str
    calibration: str
    training_mode: str
    data_source: str = "CDC BRFSS 2025 (United States), self-reported"


class PredictionResponse(BaseModel):
    estimated_probability: float = Field(..., ge=0, le=1, description="Calibrated probability from the frozen model that a respondent with these answers REPORTED existing heart disease in the survey.")
    decision_threshold: float = Field(..., description="Frozen Phase 3 threshold, read from the saved model. Never recomputed.")
    above_threshold: bool = Field(..., description="estimated_probability >= decision_threshold")
    result_label: str = Field(..., description="'above_decision_threshold' or 'below_decision_threshold'")
    interpretation: str
    target: str
    model: ModelBrief
    disclaimer: str
    request_id: str


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' when the model is loaded, otherwise 'degraded'")
    api: str = "running"
    model_loaded: bool
    model_name: Optional[str] = None
    training_mode: Optional[str] = None
    problem_code: Optional[str] = Field(None, description="Short safe code explaining why the model is not loaded; details are only in server logs.")


class ErrorItem(BaseModel):
    field: str
    problem: str
    allowed: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str
    errors: Optional[List[ErrorItem]] = None


class FeatureInfo(BaseModel):
    name: str
    kind: str
    min: float
    max: float
    description: str


class ModelInfoResponse(BaseModel):
    model_name: str
    calibration: str
    training_mode: str
    decision_threshold: float
    threshold_rule: str
    target_recall_on_validation: Optional[float] = None
    features: List[FeatureInfo]
    target: str
    data_source: str = "CDC BRFSS 2025 (United States), self-reported"
    disclaimer: str
    request_id: str
