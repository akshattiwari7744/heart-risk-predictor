# Phase 3 Report - model training & evaluation
_Generated 2026-10-23T18:53:35. All numbers come from this run._

> **Target = self-reported EXISTING coronary heart disease or myocardial infarction (BRFSS _MICHD: ever told by a health professional). It is NOT a prediction of a future heart attack and the output is NOT a diagnosis.**

## Data
Train 246,501 / validation 52,822 / test 52,822 rows (Phase 2 stratified split, seed 42). Prevalence: train 9.00%, val 9.00%, test 9.00%.

## Categorical handling
- Nominal (`diabetes_status`, `smoker_status`): one-hot in logistic regression; native categorical in HistGradientBoosting. Codes are never treated as magnitudes.
- Ordinal bands (`age_group_5yr`, `gen_health`, `education_level`, `income_level`): ordered numeric (monotone bands); HGB can learn non-linearity, LR gets a linear trend.
- Binary features are 0/1; missing values get imputation + missing-indicator flags (LR) or native NaN (HGB). All fitted on training folds only.

## Validation comparison (threshold per candidate = highest precision with recall >= 0.8, chosen on validation; optimistic for all candidates equally)
| model | pr_auc | roc_auc | precision | recall | specificity | f1 | balanced_accuracy | CV PR-AUC (train) |
|---|---|---|---|---|---|---|---|---|
| dummy_prior | 0.0900 | 0.5000 | 0.0900 | 1.0000 | 0.0000 | 0.1651 | 0.5000 | nan |
| logistic_regression | 0.3526 | 0.8409 | 0.2269 | 0.8010 | 0.7302 | 0.3537 | 0.7656 | 0.3485 |
| hist_gradient_boosting | 0.3546 | 0.8431 | 0.2270 | 0.8003 | 0.7305 | 0.3537 | 0.7654 | 0.3478 |

PR-AUC prevalence baseline (validation): **0.0900**

### Confusion matrices (validation, per candidate)
| model | TN | FP | FN | TP |
|---|---|---|---|---|
| dummy_prior | 0 | 48,069 | 0 | 4,753 |
| logistic_regression | 35,101 | 12,968 | 946 | 3,807 |
| hist_gradient_boosting | 35,114 | 12,955 | 949 | 3,804 |

## Decisions (frozen before the test set was scored)
- Selected: **logistic_regression**
- HGB-LR validation PR-AUC difference +0.0020 (95% paired bootstrap CI -0.0021..+0.0049); CI does not show HGB is better -> simpler logistic regression selected.
- Hyperparameters: `{"C": 1.0, "class_weight": null}`
- Calibration: **isotonic** (out-of-fold validation Brier by method: `{"none": 0.06856353485572333, "sigmoid": 0.0685716347061515, "isotonic": 0.06849514288187103}`)
- Threshold **0.09987**; rule: highest precision subject to recall >= 0.8

## Selected model on validation (out-of-fold calibrated)
| pr_auc | roc_auc | precision | recall | specificity | f1 | balanced_accuracy |
|---|---|---|---|---|---|---|
| 0.3466 | 0.8398 | 0.2265 | 0.8018 | 0.7293 | 0.3532 | 0.7655 |

## FINAL TEST RESULTS (scored once, after freeze)
| metric | value | 95% bootstrap CI |
|---|---|---|
| pr_auc | 0.3279 | 0.3155 - 0.3409 |
| roc_auc | 0.8359 | 0.8303 - 0.8409 |
| brier | 0.0692 | 0.0674 - 0.0709 |
| precision | 0.2264 | 0.2198 - 0.2318 |
| recall | 0.7892 | 0.7774 - 0.8001 |
| specificity | 0.7334 | 0.7295 - 0.7373 |
| f1 | 0.3519 | 0.3437 - 0.3590 |
| balanced_accuracy | 0.7613 | 0.7554 - 0.7672 |
| F2 / MCC | 0.5271 / 0.3223 | - |

Test PR-AUC prevalence baseline: **0.0900**. Confusion matrix: TN 35,253, FP 12,816, FN 1,002, TP 3,751.
Calibration slope 0.958, intercept -0.045 (ideal 1 / 0).
Survey-weighted (secondary): PR-AUC 0.3238, ROC-AUC 0.8569.

## Subgroups (test, frozen threshold)
**sex**
| group | n | prevalence | PR-AUC | ROC-AUC | precision | recall | specificity |
|---|---|---|---|---|---|---|---|
| female | 27,912 | 0.0718 | 0.2729 | 0.8278 | 0.2012 | 0.7059 | 0.7833 |
| male | 24,910 | 0.1104 | 0.3642 | 0.8350 | 0.2450 | 0.8498 | 0.6750 |

**age_band**
| group | n | prevalence | PR-AUC | ROC-AUC | precision | recall | specificity |
|---|---|---|---|---|---|---|---|
| 18-39 | 12,356 | 0.0123 | 0.0638 | 0.7604 | 0.1935 | 0.0789 | 0.9959 |
| 40-54 | 10,202 | 0.0368 | 0.2073 | 0.8269 | 0.1876 | 0.4427 | 0.9268 |
| 55-64 | 8,662 | 0.0895 | 0.3019 | 0.7965 | 0.2127 | 0.7277 | 0.7353 |
| 65-74 | 10,736 | 0.1292 | 0.3226 | 0.7621 | 0.2072 | 0.8277 | 0.5302 |
| 75+ | 9,860 | 0.2013 | 0.3749 | 0.7165 | 0.2516 | 0.9128 | 0.3156 |

**race_group_code**
| group | n | prevalence | PR-AUC | ROC-AUC | precision | recall | specificity |
|---|---|---|---|---|---|---|---|
| 1 | 38,142 | 0.0971 | 0.3317 | 0.8310 | 0.2292 | 0.8020 | 0.7101 |
| 2 | 3,896 | 0.0803 | 0.3375 | 0.8433 | 0.2105 | 0.7923 | 0.7404 |
| 3 | 786 | 0.1120 | 0.4229 | 0.8610 | 0.3120 | 0.8295 | 0.7693 |
| 4 | 1,480 | 0.0351 | 0.1517 | 0.8396 | 0.1245 | 0.5962 | 0.8473 |
| 5 | 332 | 0.0934 | 0.4413 | 0.8821 | 0.2872 | 0.8710 | 0.7774 |
| 6 | 378 | 0.1032 | 0.3134 | 0.7900 | 0.2190 | 0.7692 | 0.6844 |
| 7 | 1,303 | 0.0913 | 0.3750 | 0.8454 | 0.2558 | 0.7395 | 0.7838 |
| 8 | 5,316 | 0.0538 | 0.2520 | 0.8226 | 0.1834 | 0.6329 | 0.8398 |

## Figures
`artifacts/figures/`: val_pr_roc_candidates.png, test_pr_roc_selected.png, test_calibration.png, test_confusion_matrix.png
