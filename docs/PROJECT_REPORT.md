# Heart Risk Predictor: Project Report

> **Educational and research prototype. Not a medical device.** The app estimates how likely a person is to have *already reported* coronary heart disease or a past heart attack, based on a US survey. It does **not** predict a future heart attack, does **not** diagnose anyone, and was not validated for people in India.

## 1. Purpose
Show how a machine-learning model trained on public survey data can be packaged as a web application: a validated input form, a documented API, an honest result display, and a what-if comparison. The emphasis is on correct scope, transparent limitations, and reproducible artifacts rather than on predictive performance.

## 2. Architecture

```
Browser (React + Vite, Vercel)  --HTTPS/JSON-->  FastAPI (Render)  -->  frozen model artifacts (artifacts/)
   form (19 inputs), results,                     POST /predict
   simulator (second /predict call)               GET  /health, /model-info
```

- **Backend (`api/`)**: FastAPI and Pydantic. Strict validation: whole numbers where expected, unknown fields rejected, bodies over 16 KB rejected. The model loads once at startup and is served only if the file hash, library versions, training mode, feature order and self-test all pass. Otherwise `/health` reports `degraded` and `/predict` returns a generic 503.
- **Frontend (`frontend/`)**: React 19, TypeScript, Tailwind CSS v4, bundled with Vite. The questionnaire, validation and API client are in `schema.ts` and `api.ts`. No answers are stored in browser storage.
- **Security defaults**: production mode, no cross-origin access unless `API_CORS_ORIGINS` names the frontend explicitly, uniform error bodies with no stack traces, no logging of health answers, and `Cache-Control: no-store` on predictions.
- **Limits**: no authentication or rate limiting. Keep the API behind a reverse proxy with TLS and rate limits before any public use.

## 3. Data
- **Source**: CDC Behavioral Risk Factor Surveillance System (BRFSS) 2025, file `LLCP2025`. A telephone survey of US adults. All answers are self-reported.
- **Outcome**: `_MICHD`, "ever told by a health professional" that you have coronary heart disease or had a heart attack. Records with a missing outcome were dropped, leaving **352,145** respondents.
- **Prevalence**: 9.0% in every split.
- **Split** (stratified on the outcome, seed 42): train 246,501 / validation 52,822 / test 52,822. The positive rate differs by at most 0.000 percentage points between splits (`docs/phase2_report.md`).
- **Inputs**: 19 survey variables mapped to model features (`docs/feature_spec.csv`, `src/config.py`). Race is *not* an input; it is used only for subgroup evaluation.
- **Cleaning**: codes such as "don't know", "refused" and "not asked" become missing values, with a missing-indicator flag where the model uses one. Day counts of 88 mean "none" and become 0. Nominal codes are one-hot encoded, never treated as numbers.

## 4. Model development
- **Candidates**: a prior-only baseline (dummy), logistic regression, and histogram gradient boosting (HGB). Both real models were tuned on training data and compared on validation data.
- **Selection** (`artifacts/decisions_frozen.json`): HGB beat logistic regression by 0.0020 PR-AUC, with a 95% paired-bootstrap CI of -0.0021 to +0.0049. The interval includes zero, so the simpler logistic regression was chosen.
- **Calibration**: isotonic regression, chosen by out-of-fold Brier score on validation (0.06850, against 0.06856 uncalibrated).
- **Decision threshold**: 0.09987, chosen on validation as the highest precision that still gives recall of at least 0.8. The threshold is frozen in the saved model and is never recomputed by the API.
- **Freeze**: decisions were fixed before the test set was scored, and the test set was scored once.

## 5. Results (test set, frozen threshold, 95% bootstrap CIs)

| Metric | Value | 95% CI |
|---|---|---|
| PR-AUC | 0.3279 (baseline 0.0900) | 0.3155 to 0.3409 |
| ROC-AUC | 0.8359 | 0.8303 to 0.8409 |
| Brier score | 0.0692 | 0.0674 to 0.0709 |
| Recall (sensitivity) | 0.7892 | 0.7774 to 0.8001 |
| Specificity | 0.7334 | 0.7295 to 0.7373 |
| Precision | 0.2264 | 0.2198 to 0.2318 |
| F1 | 0.3519 | 0.3437 to 0.3590 |
| Balanced accuracy | 0.7613 | 0.7554 to 0.7672 |

Confusion matrix (test): TN 35,253, FP 12,816, FN 1,002, TP 3,751. Calibration slope 0.958 and intercept -0.045 (ideal 1 and 0).

**How to read this.** Recall is high by design, so most people flagged (about 77%) do not report the condition. A "below threshold" result is not reassurance: about 21% of people who reported the condition fall below it. Survey-weighted secondary results: PR-AUC 0.3238, ROC-AUC 0.8569.

### Subgroups (test, frozen threshold)
Performance differs substantially across groups:
- **Age**: the model is weak under 40. Prevalence is 1.2% at 18 to 39, with PR-AUC 0.064 and recall 0.08. For 75 and over, recall is 0.91, but specificity is only 0.32.
- **Sex**: recall 0.71 for women and 0.85 for men, with PR-AUC 0.27 and 0.36.
- **Race group** (survey codes): PR-AUC ranges from 0.15 (group 4, n = 1,480) to 0.44 (group 5, n = 332). Small groups have wide uncertainty.

These gaps are reported, not corrected. Any future version should evaluate fairness before use.

## 6. Limitations
1. **Target is existing disease, not future risk.** Cross-sectional data cannot establish that a risk factor causes the condition or predicts a future event.
2. **Self-reported data.** Both inputs and outcome may be misreported, and people who never saw a doctor may be missed.
3. **Reverse causation.** General health, blood pressure and similar inputs can be consequences of heart disease.
4. **Population fit.** US adults only. Not validated for India or other health systems.
5. **Step-shaped outputs.** The isotonic calibrator is piecewise constant, so small input changes can leave the displayed probability unchanged for many profiles. The what-if simulator can therefore show "0.0" for small changes. This is a property of the frozen model and was not altered.
6. **Single split, single model.** Results come from one train/validation/test split with a fixed seed.
7. **No clinical validation.** No clinician reviewed the output, and no prospective study was done.

## 7. Reproducibility
- Pinned environment: `requirements-lock.txt` (Python 3.14.2, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4).
- Artifacts: `artifacts/model.joblib`, with SHA-256 recorded in `artifacts/model_meta.json`. The API refuses to load a file that does not match.
- 120 backend tests cover the API contract, model integrity, the output-isolation guard (tests cannot modify production artifacts), and the lock file. The frontend has 13 tests for validation and the API client.

## 8. Future scope
- **Framingham 10-year future-risk model** (a separate feature). Not included in this project because the model files and training code were not available. See the README, "Missing components".
- Per-person explanation of contributions (for example SHAP values). The current app shows only the model's estimate and does not claim contributing factors.
- Subgroup-aware evaluation and calibration checks for each group before any wider use.
- Authentication, rate limiting, and a proper privacy notice before any public deployment.
- Clinical review by a qualified professional.
