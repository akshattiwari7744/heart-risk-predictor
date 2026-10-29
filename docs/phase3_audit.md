# Phase 3 audit (read-only)

## 1. Which configuration produced the surviving artifacts?
cfg stored in decisions_frozen.json: {'seed': 42, 'cv_folds': 3, 'hgb_iter': 10, 'tune_rows': 120000, 'n_boot': 300, 'target_recall': 0.8, 'select_boot': 200}
selected=logistic_regression  LR params={'C': 1.0, 'class_weight': None}  HGB params={'min_samples_leaf': 50, 'max_leaf_nodes': 31, 'max_iter': 200, 'learning_rate': 0.05, 'l2_regularization': 0.0, 'class_weight': None}
threshold=0.09986684420772303  frozen_at=2026-10-23T18:53:24
[PASS] 1a quick flag changes procedure (code): Cfg: cv_folds 3->2, hgb_iter 10->3, tune_rows 120000->30000, n_boot 300->100, select_boot 200->100
[PASS] 1b surviving artifacts come from: full-run settings (hgb_iter=10, tune_rows=120000)

## 4. Fresh-process load
[PASS] 4a fresh-process load + predict on 2000 validation rows: FinalModel loaded; probs in [0.0000, 1.0000]; python 3.14.2
[PASS] 4b library versions match model_meta.json: saved {'python': '3.14.2', 'scikit-learn': '1.9.1', 'pandas': '3.1.0', 'numpy': '2.5.4', 'joblib': '1.6.0'} vs now sklearn 1.9.1, pandas 3.1.0, numpy 2.5.4
[PASS] 4c installed packages satisfy requirements.txt: all declared requirements satisfied
NOTE: requirements.txt uses >= ranges, not exact pins; the exact versions are only recorded in model_meta.json.

## 5. Artifact / report consistency
[PASS] 5 model.joblib sha256 == model_meta: True
[PASS] 5 threshold model == decisions == metrics == meta: True
[PASS] 5 selected model name agrees: True
[PASS] 5 calibration method agrees: True
[PASS] 5 feature order == config.FEATURES == meta: True
[PASS] 5 pipeline fitted feature_names_in_ == feature order: True
[PASS] 5 target statement saved (metrics, meta): True
[PASS] 5 phase3_report.md states existing-disease target, not future risk: text present
[PASS] 5 model_card.md states existing-disease target, not future risk: text present
[PASS] 5 scaler means == means recomputed from train.parquet: fit on train only
[PASS] 5 frozen model re-scored on test reproduces metrics.json (pr_auc, recall): recomputed pr_auc=0.327866 vs saved 0.327866; recall 0.789186 vs 0.789186
[PASS] 5 phase3_report.md test PR-AUC == metrics.json: report 0.3279 vs 0.3279

## 2. Age subgroups (frozen threshold 0.09987 applied identically to every group)
| band (_AGEG5YR codes) | n | positives | % of all positives | TP | FN | recall [95% Wilson] | % flagged | mean p | within-band ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| 18-39 (1-4) | 12,356 | 152 | 3.2% | 12 | 140 | 7.89% [4.6, 13.3] | 0.5% | 0.0086 | 0.760 |
| 40-54 (5-7) | 10,202 | 375 | 7.9% | 166 | 209 | 44.27% [39.3, 49.3] | 8.7% | 0.0350 | 0.827 |
| 55-64 (8-9) | 8,662 | 775 | 16.3% | 564 | 211 | 72.77% [69.5, 75.8] | 30.6% | 0.0864 | 0.796 |
| 65-74 (10-11) | 10,736 | 1,387 | 29.2% | 1,148 | 239 | 82.77% [80.7, 84.7] | 51.6% | 0.1344 | 0.762 |
| 75+ (12-13) | 9,860 | 1,985 | 41.8% | 1,812 | 173 | 91.28% [90.0, 92.4] | 73.0% | 0.1965 | 0.716 |

Rows with missing age band (excluded from every band): 1,006; positives among them: 79
[PASS] 2a bands partition the test set (positives reconcile): 4674+79 vs 4753
[PASS] 2b age-band mapping matches CDC 2025 _AGEG5YR (1=18-24 ... 13=80+): bins 1-4, 5-7, 8-9, 10-11, 12-13 = 18-39, 40-54, 55-64, 65-74, 75+
[PASS] 2c same frozen threshold used for all groups: thr=0.09986684420772303; no per-group threshold in code (train_model.subgroup_table)
Interpretation: a single global threshold on a probability that is strongly driven by age means young positives rarely cross it,
while nearly everyone aged 75+ does (high recall, low precision there). That is expected behaviour, not a calculation bug.
Read the Wilson interval, not the point estimate, for bands with few positives.

## 3. Race/ethnicity subgroups (_RACE labels per CDC 2025 calculated-variables document)
| code | label | n | positives | recall [95% Wilson] | ROC-AUC | stability |
|---|---|---|---|---|---|---|
| 1 | White only, non-Hispanic | 38,142 | 3,702 | 80.2% [78.9, 81.5] | 0.831 | stable |
| 2 | Black only, non-Hispanic | 3,896 | 313 | 79.2% [74.4, 83.4] | 0.843 | stable |
| 3 | American Indian/Alaska Native only, non-Hispanic | 786 | 88 | 83.0% [73.8, 89.4] | 0.861 | wide uncertainty |
| 4 | Asian only, non-Hispanic | 1,480 | 52 | 59.6% [46.1, 71.8] | 0.840 | wide uncertainty |
| 5 | Native Hawaiian/Other Pacific Islander only, non-Hispanic | 332 | 31 | 87.1% [71.1, 94.9] | 0.882 | wide uncertainty |
| 6 | Other race only, non-Hispanic | 378 | 39 | 76.9% [61.7, 87.4] | 0.790 | wide uncertainty |
| 7 | Multiracial, non-Hispanic | 1,303 | 119 | 73.9% [65.4, 81.0] | 0.845 | stable |
| 8 | Hispanic | 5,316 | 286 | 63.3% [57.6, 68.7] | 0.823 | stable |

Rows with _RACE = 9 (don't know/refused) or missing: 1,189 (excluded; positives among them: 123)
[PASS] 3 race labels verified against CDC 2025 document: codes 1-8 labelled above; 9 = DK/refused (set to NaN in Phase 2)

## Summary
| item | status | evidence |
|---|---|---|
| 1a quick flag changes procedure (code) | PASS | Cfg: cv_folds 3->2, hgb_iter 10->3, tune_rows 120000->30000, n_boot 300->100, select_boot 200->100 |
| 1b surviving artifacts come from | PASS | full-run settings (hgb_iter=10, tune_rows=120000) |
| 4a fresh-process load + predict on 2000 validation rows | PASS | FinalModel loaded; probs in [0.0000, 1.0000]; python 3.14.2 |
| 4b library versions match model_meta.json | PASS | saved {'python': '3.14.2', 'scikit-learn': '1.9.1', 'pandas': '3.1.0', 'numpy': '2.5.4', 'joblib': '1.6.0'} vs now sklearn 1.9.1, pandas 3.1.0, numpy 2.5.4 |
| 4c installed packages satisfy requirements.txt | PASS | all declared requirements satisfied |
| 5 model.joblib sha256 == model_meta | PASS | True |
| 5 threshold model == decisions == metrics == meta | PASS | True |
| 5 selected model name agrees | PASS | True |
| 5 calibration method agrees | PASS | True |
| 5 feature order == config.FEATURES == meta | PASS | True |
| 5 pipeline fitted feature_names_in_ == feature order | PASS | True |
| 5 target statement saved (metrics, meta) | PASS | True |
| 5 phase3_report.md states existing-disease target, not future risk | PASS | text present |
| 5 model_card.md states existing-disease target, not future risk | PASS | text present |
| 5 scaler means == means recomputed from train.parquet | PASS | fit on train only |
| 5 frozen model re-scored on test reproduces metrics.json (pr_auc, recall) | PASS | recomputed pr_auc=0.327866 vs saved 0.327866; recall 0.789186 vs 0.789186 |
| 5 phase3_report.md test PR-AUC == metrics.json | PASS | report 0.3279 vs 0.3279 |
| 2a bands partition the test set (positives reconcile) | PASS | 4674+79 vs 4753 |
| 2b age-band mapping matches CDC 2025 _AGEG5YR (1=18-24 ... 13=80+) | PASS | bins 1-4, 5-7, 8-9, 10-11, 12-13 = 18-39, 40-54, 55-64, 65-74, 75+ |
| 2c same frozen threshold used for all groups | PASS | thr=0.09986684420772303; no per-group threshold in code (train_model.subgroup_table) |
| 3 race labels verified against CDC 2025 document | PASS | codes 1-8 labelled above; 9 = DK/refused (set to NaN in Phase 2) |
