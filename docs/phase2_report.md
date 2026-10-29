# Phase 2 Report - preprocessing & split
_Generated 2026-10-23T11:57:02 (seed 42). Every number comes from your data; nothing hand-entered._

## Split (stratified on target, fixed seed)
| split | rows | positives | positive rate |
|---|---|---|---|
| train | 246,501 | 22,181 | 8.998% |
| val | 52,822 | 4,753 | 8.998% |
| test | 52,822 | 4,753 | 8.998% |

Fit-on-train-only check (imputer/scaler statistics == train medians): **True**
Max positive-rate gap between splits: 0.000 pp

## Model inputs (19)
| model name | source | evidence | train NaN % |
|---|---|---|---|
| age_group_5yr | _AGEG5YR | CDC25_CALC | 1.84 |
| sex_female | _SEX | CDC25_CALC | 0.0 |
| bmi | _BMI5 | CDC25_CALC | 9.04 |
| gen_health | GENHLTH | CDC25_CALC | 0.26 |
| phys_unhealthy_days | PHYSHLTH | CDC25_CALC | 2.28 |
| ment_unhealthy_days | MENTHLTH | CDC25_CALC | 1.75 |
| any_physical_activity | _TOTINDA | CDC25_CALC | 0.26 |
| hypertension_told | BPHIGH6 | CDC25_CALC | 1.14 |
| high_cholesterol_told | TOLDHI3 | CDC25_CALC | 12.52 |
| diabetes_status | DIABETE4 | QUEST_2021_22 | 0.16 |
| asthma_ever | ASTHMA3 | CDC25_CALC | 0.28 |
| copd_ever | CHCCOPD3 | CONVENTION | 0.36 |
| kidney_disease_ever | CHCKDNY2 | CONVENTION | 0.33 |
| depression_ever | ADDEPEV3 | CONVENTION | 0.55 |
| other_cancer_ever | CHCOCNC1 | CONVENTION | 0.48 |
| smoker_status | _SMOKER3 | CDC25_CALC | 5.75 |
| heavy_drinker | _RFDRHV9 | CDC25_CALC | 8.03 |
| education_level | _EDUCAG | CDC25_CALC | 0.57 |
| income_level | _INCOMG1 | CDC25_CALC | 19.77 |

## Single-feature leakage screen (train; folded AUC; >0.85 would need investigation)
| feature | AUC |
|---|---|
| age_group_5yr | 0.737 |
| gen_health | 0.705 |
| hypertension_told | 0.675 |
| high_cholesterol_told | 0.643 |
| phys_unhealthy_days | 0.623 |
| diabetes_status | 0.596 |
| income_level | 0.595 |
| smoker_status | 0.590 |
| copd_ever | 0.584 |
| any_physical_activity | 0.574 |
| sex_female | 0.562 |
| kidney_disease_ever | 0.562 |
| other_cancer_ever | 0.561 |
| education_level | 0.553 |
| bmi | 0.540 |
| asthma_ever | 0.526 |
| depression_ever | 0.525 |
| heavy_drinker | 0.509 |
| ment_unhealthy_days | 0.502 |

Flagged (>0.85): none

## Raw-code evidence checks (valid-target rows)
### Age variable
- `AGE` present in subset: **False**; `_AGEG5YR` present: **True**.
- `_AGEG5YR` counts: 1:23,092, 2:17,894, 3:19,208, 4:21,313, 5:22,837, 6:21,858, 7:22,935, 8:25,782, 9:32,242, 10:36,729, 11:35,258, 12:30,580, 13:35,957, 14:6,460

### PHYSHLTH / MENTHLTH special codes (CDC: 88=none, 77/99=DK/refused; 7 and 9 are real day counts)
- `PHYSHLTH`: 77:6,559 | 88:206,249 | 99:1,444 | 7:7,063 (real 7 days) | 9:348 (real 9 days) | NaN:1
- `MENTHLTH`: 77:4,310 | 88:205,957 | 99:1,879 | 7:6,393 (real 7 days) | 9:247 (real 9 days) | NaN:0

### `_BMI5`
- observed min=1202, max=9843 (codebook: x100, valid 1200-9999) -> divide by 100; NaN: 31,920

### SMOKDAY2 skip pattern (CDC `_SMOKER3` logic: SMOKDAY2 used only when SMOKE100=1)
```
                        n  smokday2_nan  pct_nan
SMOKE100 (-1=NaN)                               
-1.0                17961         17961   100.00
 1.0               129254           117     0.09
 2.0               203037        203037   100.00
 7.0                 1460          1460   100.00
 9.0                  433           433   100.00
```
- `_SMOKER3` re-derived from SMOKE100/SMOKDAY2 per CDC SAS rule: **0 mismatches** (0 expected).

### `_TOTINDA` vs EXERANY2 rule: **0 mismatches** (0 expected).

### FRNCHDA_
- present in all_columns.csv, non-null 88.22%. CDC 2025 calculated-variables doc: French-fry intake per day, derived from FRENCHF1 (diet item). Name-scan hit came from the letters 'CHD' inside the name; not related to heart disease. Excluded.

## Files written
- data/processed/{train,val,test}.parquet
- artifacts/preprocessor_linear.joblib, preprocessor_tree.joblib, preprocess_meta.json
- docs/feature_spec_v1.csv, docs/phase2_report.md
