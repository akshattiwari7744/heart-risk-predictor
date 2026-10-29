# Feature Audit - BRFSS 2025 (Phase 1)
_Generated 2026-10-23 11:42 by `src/phase1_inspect.py` in 12s; pandas 3.1.0. All numbers below come from your file; nothing is hand-entered._

> Codes marked DRAFT are conventions, **not** confirmed. Check each against the 2025 codebook and set `codebook_confirmed` in `feature_spec.csv`.

## 1. Dataset integrity
| Check | Observed | Expected | Result |
|---|---|---|---|
| Rows in file | 356,158 | 356,158 | PASS |
| Columns in file | 283 | 283 | PASS |
| Rows with missing/non-1-2 _MICHD | 4,013 | 4,013 | PASS |
| Rows after dropping missing target | 352,145 | 352,145 | PASS |

XPT: `C:\Users\akshat tiwari\Downloads\Heart_Risk_Predictor\LLCP2025.XPT` (read once, read-only). Profiles below use the **352,145** rows with a valid target.

## 2. Target `_MICHD`
| _MICHD value (all rows) | count |
|---|---|
| 1 | 31,687 |
| 2 | 320,458 |
| <NaN> | 4,013 |

Valid rows: **352,145**; positives (=1): **31,687**; unweighted positive rate: **9.00%**.
Weighted (`_LLCPWT`) positive rate, informational only: 6.75%.

Re-derivation check (1 if either source=1; 2 if both=2; else missing) vs `_MICHD`: **0 mismatching rows** (0 expected; non-zero means a rule/code difference - check codebook).

## 3. Variable verification
| Column | Role | Status | Similar names (if missing) | NaN % (valid rows) | Special codes seen |
|---|---|---|---|---|---|
| AGE | feature | MISSING | - | - | - |
| _AGEG5YR | feature_alt | FOUND | - | 0.0% | 7:22935\|9:32242 |
| _AGE65YR | feature_alt | FOUND | - | 0.0% | - |
| _SEX | feature | FOUND | - | 0.0% | - |
| SEXVAR | feature_alt | FOUND | - | 0.0% | - |
| _BMI5 | feature | FOUND | - | 9.06% | - |
| GENHLTH | feature | FOUND | - | 0.0% | 7:650\|9:238 |
| PHYSHLTH | feature | FOUND | - | 0.0% | 7:7063\|9:348\|77:6559\|88:206249\|99:1444 |
| MENTHLTH | feature | FOUND | - | 0.0% | 7:6393\|9:247\|77:4310\|88:205957\|99:1879 |
| _TOTINDA | feature | FOUND | - | 0.0% | 9:904 |
| EXERANY2 | feature_alt | FOUND | - | 0.0% | 7:623\|9:281 |
| BPHIGH6 | feature | FOUND | - | 0.73% | 7:1033\|9:405 |
| TOLDHI3 | feature | FOUND | - | 11.71% | 7:2368\|9:333 |
| DIABETE4 | feature | FOUND | - | 0.0% | 7:497\|9:83 |
| ASTHMA3 | feature | FOUND | - | 0.0% | 7:1003\|9:37 |
| CHCCOPD3 | feature | FOUND | - | 0.0% | 7:1227\|9:60 |
| CHCKDNY2 | feature | FOUND | - | 0.0% | 7:1145\|9:72 |
| ADDEPEV3 | feature | FOUND | - | 0.0% | 7:1540\|9:383 |
| CHCOCNC1 | feature | FOUND | - | 0.0% | 7:1555\|9:148 |
| _SMOKER3 | feature | FOUND | - | 0.0% | 9:20232 |
| SMOKE100 | feature_alt | FOUND | - | 5.1% | 7:1460\|9:433 |
| SMOKDAY2 | feature_alt | FOUND | - | 63.33% | 7:107\|9:154 |
| _RFDRHV9 | feature | FOUND | - | 0.0% | 9:28256 |
| _EDUCAG | feature | FOUND | - | 0.0% | 9:2023 |
| _INCOMG1 | feature | FOUND | - | 0.0% | 7:23316\|9:69586 |
| _RACE | feature_alt | FOUND | - | 0.0% | 7:8622\|9:8036 |
| _IMPRACE | feature_alt | FOUND | - | 0.0% | - |
| MARITAL | feature | FOUND | - | 0.0% | 9:3749 |
| EMPLOY1 | feature | FOUND | - | 0.75% | 7:118031\|9:4187 |
| _MICHD | target | FOUND | - | 0.0% | - |
| CVDINFR4 | excluded_leakage | FOUND | - | 0.0% | 7:316\|9:5 |
| CVDCRHD4 | excluded_leakage | FOUND | - | 0.0% | 7:1047\|9:12 |
| CVDSTRK3 | excluded_related | FOUND | - | 0.0% | 7:819\|9:33 |
| BPMEDS1 | excluded_related | FOUND | - | 58.89% | 7:267\|9:113 |
| CHOLMED3 | excluded_related | FOUND | - | 15.84% | 7:1331\|9:265 |
| CHECKUP1 | excluded_access | FOUND | - | 0.0% | 7:3383\|9:448 |
| PERSDOC3 | excluded_access | FOUND | - | 0.0% | 7:2559\|9:798 |
| MEDCOST1 | excluded_access | FOUND | - | 0.0% | 7:884\|9:252 |
| _HLTHPL2 | excluded_access | FOUND | - | 0.0% | 9:15052 |
| _LLCPWT | weight_secondary | FOUND | - | 0.0% | - |
| _STSTR | excluded_admin | FOUND | - | 0.0% | - |
| _PSU | excluded_admin | FOUND | - | 0.0% | - |
| _STATE | excluded_admin | FOUND | - | 0.0% | 9:7362 |
| FMONTH | excluded_admin | FOUND | - | 0.0% | 7:30464\|9:26663 |
| IDATE | excluded_admin | FOUND | - | 0.0% | - |
| IMONTH | excluded_admin | FOUND | - | 0.0% | - |
| IDAY | excluded_admin | FOUND | - | 0.0% | - |
| IYEAR | excluded_admin | FOUND | - | 0.0% | - |
| DISPCODE | excluded_admin | FOUND | - | 0.0% | - |
| SEQNO | excluded_admin | FOUND | - | 0.0% | - |

## 4. Missing or renamed variables
| Spec name | Role | Possible matches in file |
|---|---|---|
| AGE | feature | none |

## 5. Special-code findings (candidate features)
Numeric codes from a standard BRFSS list that actually occur. Whether they mean 'Don't know/Refused/Not asked' or a real answer must be confirmed per variable in the codebook. For continuous fields (`AGE`, `_BMI5`) these hits are probably genuine values, not codes.

| Column | Distinct values | Sentinel-like codes (code:count) |
|---|---|---|
| _AGEG5YR | 14 | 7:22,935; 9:32,242 |
| GENHLTH | 7 | 7:650; 9:238 |
| PHYSHLTH | 33 | 7:7,063; 9:348; 77:6,559; 88:206,249; 99:1,444 |
| MENTHLTH | 33 | 7:6,393; 9:247; 77:4,310; 88:205,957; 99:1,879 |
| _TOTINDA | 3 | 9:904 |
| EXERANY2 | 4 | 7:623; 9:281 |
| BPHIGH6 | 6 | 7:1,033; 9:405 |
| TOLDHI3 | 4 | 7:2,368; 9:333 |
| DIABETE4 | 6 | 7:497; 9:83 |
| ASTHMA3 | 4 | 7:1,003; 9:37 |
| CHCCOPD3 | 4 | 7:1,227; 9:60 |
| CHCKDNY2 | 4 | 7:1,145; 9:72 |
| ADDEPEV3 | 4 | 7:1,540; 9:383 |
| CHCOCNC1 | 4 | 7:1,555; 9:148 |
| _SMOKER3 | 5 | 9:20,232 |
| SMOKE100 | 4 | 7:1,460; 9:433 |
| SMOKDAY2 | 5 | 7:107; 9:154 |
| _RFDRHV9 | 3 | 9:28,256 |
| _EDUCAG | 5 | 9:2,023 |
| _INCOMG1 | 8 | 7:23,316; 9:69,586 |
| _RACE | 9 | 7:8,622; 9:8,036 |
| MARITAL | 7 | 9:3,749 |
| EMPLOY1 | 9 | 7:118,031; 9:4,187 |
| CVDINFR4 | 4 | 7:316; 9:5 |
| CVDCRHD4 | 4 | 7:1,047; 9:12 |
| CVDSTRK3 | 4 | 7:819; 9:33 |
| BPMEDS1 | 4 | 7:267; 9:113 |
| CHOLMED3 | 4 | 7:1,331; 9:265 |
| CHECKUP1 | 7 | 7:3,383; 9:448 |
| PERSDOC3 | 5 | 7:2,559; 9:798 |
| MEDCOST1 | 4 | 7:884; 9:252 |
| _HLTHPL2 | 3 | 9:15,052 |
| _STATE | 50 | 9:7,362 |
| FMONTH | 12 | 7:30,464; 9:26,663 |

Full value tables: `docs/value_counts.csv` (columns with <= 40 distinct values).

## 6. Leakage audit
| Column | Role | Present | Reason |
|---|---|---|---|
| CVDINFR4 | excluded_leakage | yes | Direct source of the target (_MICHD is derived from it). Must never be a feature. |
| CVDCRHD4 | excluded_leakage | yes | Direct source of the target (_MICHD is derived from it). Must never be a feature. |
| CVDSTRK3 | excluded_related | yes | Related cardiovascular outcome or treatment proxy; excluded from v1. |
| BPMEDS1 | excluded_related | yes | Related cardiovascular outcome or treatment proxy; excluded from v1. |
| CHOLMED3 | excluded_related | yes | Related cardiovascular outcome or treatment proxy; excluded from v1. |
| CHECKUP1 | excluded_access | yes | Healthcare-contact proxy for diagnosis; optional labelled experiment only. |
| PERSDOC3 | excluded_access | yes | Healthcare-contact proxy for diagnosis; optional labelled experiment only. |
| MEDCOST1 | excluded_access | yes | Healthcare-contact proxy for diagnosis; optional labelled experiment only. |
| _HLTHPL2 | excluded_access | yes | Healthcare-contact proxy for diagnosis; optional labelled experiment only. |
| _LLCPWT | weight_secondary | yes | Survey weight; used only for secondary weighted evaluation. |
| _STSTR | excluded_admin | yes | Survey design / administrative field; no valid predictive meaning. |
| _PSU | excluded_admin | yes | Survey design / administrative field; no valid predictive meaning. |
| _STATE | excluded_admin | yes | Survey design / administrative field; no valid predictive meaning. |
| FMONTH | excluded_admin | yes | Survey design / administrative field; no valid predictive meaning. |
| IDATE | excluded_admin | yes | Survey design / administrative field; no valid predictive meaning. |
| IMONTH | excluded_admin | yes | Survey design / administrative field; no valid predictive meaning. |
| IDAY | excluded_admin | yes | Survey design / administrative field; no valid predictive meaning. |
| IYEAR | excluded_admin | yes | Survey design / administrative field; no valid predictive meaning. |
| DISPCODE | excluded_admin | yes | Survey design / administrative field; no valid predictive meaning. |
| SEQNO | excluded_admin | yes | Survey design / administrative field; no valid predictive meaning. |

Hard rule: `_MICHD`, `CVDINFR4`, `CVDCRHD4` are never model inputs. The Phase 2 loader must assert this.

### Name scan of all columns (needs your manual codebook review)
Matching /CVD|ASPR|ANGINA|HEART|CORON|STRK|STROKE|CHD/ and not in the spec: FRNCHDA_.
Matching but already handled in the spec: CVDINFR4, CVDCRHD4, CVDSTRK3, _MICHD.

### Calculated (`_`-prefixed) columns not in the spec (51)
Review these for useful features or hidden leakage; do not add any without reading its definition:
`_METSTAT`, `_URBSTAT`, `_STRWT`, `_RAWRAKE`, `_WT2RAKE`, `_CHISPNC`, `_CRACE1`, `_CLLCPWT`, `_DUALUSE`, `_DUALCOR`, `_LLCPWT2`, `_RFHLTH`, `_PHYS14D`, `_MENT14D`, `_HCVU654`, `_RFHYPE6`, `_CHOLCH3`, `_RFCHOL3`, `_LTASTH1`, `_CASTHM1`, `_ASTHMS1`, `_DRDXAR2`, `_MRACE1`, `_HISPANC`, `_RACEG21`, `_RACEGR3`, `_RACEPRV`, `_AGE80`, `_AGE_G`, `_BMI5CAT`, `_RFBMI5`, `_CHLDCNT`, `_RFSMOK3`, `_CURECI3`, `_RFBING6`, `_DRNKWK3`, `_FLSHOT7`, `_PNEUMO3`, `_MISFRT1`, `_MISVEG1`, `_FRTRES1`, `_VEGRES1`, `_FRUTSU1`, `_VEGESU1`, `_FRTLT1A`, `_VEGLT1A`, `_FRT16A`, `_VEG23A`, `_FRUITE1`, `_VEGETE1`, `_AIDTST4`

## 7. Corrections needed
- Candidate/target `AGE` NOT FOUND. No similar column found.
- `SMOKDAY2` has 63.33% NaN among valid-target rows; decide: keep (+missing flag), skip-pattern category, or drop.
- Check `_BMI5` scaling: observed min=1202, max=9843 (expect values like 1200-7000 if stored x100).
- `PHYSHLTH` special codes seen: {'7': 7063, '9': 348, '77': 6559, '88': 206249, '99': 1444} (confirm 88='none', 77/99=don't know/refused).
- `MENTHLTH` special codes seen: {'7': 6393, '9': 247, '77': 4310, '88': 205957, '99': 1879} (confirm 88='none', 77/99=don't know/refused).
- Manual: fill `codebook_label` and set `codebook_confirmed=YES` per variable after reading the codebook.

## 8. Files written
- `docs/feature_spec.csv`, `docs/feature_audit.md`, `docs/value_counts.csv`, `docs/all_columns.csv`, `docs/phase1_report.txt`
- `data/interim/brfss_phase1_subset.parquet` (356,158 rows x 53 columns; spec columns only, includes excluded/leakage columns for audit - Phase 2 must select by role).
