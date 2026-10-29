# Coding verification - DIABETE4, CHCCOPD3, CHCKDNY2, ADDEPEV3, CHCOCNC1

**Source read:** North Carolina 2025 BRFSS Questionnaire (v3), Core Section 7 (CCHC.07-CCHC.12). Core questions are
set by CDC and asked unchanged by every state (NC SCHS BRFSS questionnaire page). The CDC 2025 national
codebook/questionnaire PDFs could not be retrieved, so this is a state copy of the CDC core, not the CDC file itself.

**Question-to-variable link:** the PDF prints question text only (variable-name column blank). The link is by question
text and matches the CDC 2022 questionnaire (which prints CHCCOPD3, ADDEPEV3, CHCKDNY2 beside the same text and
responses) and Wisconsin 2024 codebook labels (CHCOCNC1 "melanoma or any other types of cancer",
DIABETE4 "Diabetes Diagnosis (Lifetime)", CHCCOPD3 "COPD", CHCKDNY2 "Kidney Disease", ADDEPEV3 "depressive disorder").
The 2025 layout lists DIABETE4 in the file.

| Variable | 2025 question (CCHC.#) | Response codes | Phase 2 mapping | Verdict |
|---|---|---|---|---|
| CHCCOPD3 | (Ever told) COPD, emphysema or chronic bronchitis (.08) | 1 Yes, 2 No, 7 DK/Not sure, 9 Refused | 1->1, 2->0, 7/9->NaN | Verified, no change |
| ADDEPEV3 | (Ever told) depressive disorder incl. depression, major depression, dysthymia, minor depression (.09) | 1,2,7,9 as above | same | Verified, no change |
| CHCKDNY2 | Ever told kidney disease, not counting kidney stones, bladder infection or incontinence (.10) | 1,2,7,9 as above | same | Verified, no change |
| CHCOCNC1 | (Ever told) melanoma or any other types of cancer (.07) | 1,2,7,9 as above | same | Codes verified. Label corrected: includes melanoma; non-melanoma skin cancer is a separate item (CHCSCNC1). |
| DIABETE4 | (Ever told) diabetes (.12) | 1 Yes; 2 Yes, but female told only during pregnancy; 3 No; 4 No, pre-diabetes or borderline diabetes; 7 DK; 9 Refused | 1..4 kept as nominal categories; 7/9->NaN | Verified. Code 3 = No is now confirmed (was inferred). Code 4 = "No, pre-diabetes or borderline". Codes are categories, not magnitudes. |

**Residual uncertainty**
- State copy of the CDC core, not the national 2025 PDF. Mitigation: Phase 2 aborts on any observed code outside these maps, and it completed on your real data, so only these codes occur.
- Question-to-variable link is by text (above), not a printed variable name.

**Corrections made:** documentation only (evidence tags, DIABETE4 and CHCOCNC1 labels in `src/config.py`). No code map changed,
so Phase 2 splits and preprocessors remain valid. `docs/feature_spec_v1.csv` shows the old evidence text until Phase 2 is
re-run (optional; deterministic, same seed gives identical splits).
