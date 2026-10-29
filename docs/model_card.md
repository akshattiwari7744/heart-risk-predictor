# Model Card - BRFSS 2025 Heart Disease Reporting Model

## What this model predicts
Target = self-reported EXISTING coronary heart disease or myocardial infarction (BRFSS _MICHD: ever told by a health professional). It is NOT a prediction of a future heart attack and the output is NOT a diagnosis.

It estimates the likelihood that a respondent *has reported* coronary heart disease or a heart attack, from self-reported demographics, lifestyle and chronic-condition answers. Because the data are cross-sectional, it shows association, not cause or future risk, and must not be used to diagnose, triage, or exclude disease.

## Data
CDC BRFSS 2025 (LLCP2025), adults, US telephone survey; self-reported. 352,145 rows with a valid target.

## Model
logistic_regression, calibrated with isotonic; decision threshold 0.09987 (rule: highest precision subject to recall >= 0.8). Inputs (19): age_group_5yr, sex_female, bmi, gen_health, phys_unhealthy_days, ment_unhealthy_days, any_physical_activity, hypertension_told, high_cholesterol_told, diabetes_status, asthma_ever, copd_ever, kidney_disease_ever, depression_ever, other_cancer_ever, smoker_status, heavy_drinker, education_level, income_level.

## Test performance (single evaluation)
PR-AUC 0.3279 (prevalence baseline 0.0900); ROC-AUC 0.8359; recall 0.7892; precision 0.2264; specificity 0.7334. See docs/phase3_report.md for CIs and subgroups.

## Limitations
- Self-reported labels and inputs; recall/misreporting bias.
- US survey; may not transfer to other populations (e.g. India).
- Existing-condition target: several inputs (e.g. general health, hypertension) can be consequences of heart disease, not only risk factors.
- Race is excluded from inputs and used only for subgroup evaluation; subgroup performance can differ.
- Not a medical device.
