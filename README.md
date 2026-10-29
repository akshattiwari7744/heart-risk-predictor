# Heart Risk Predictor

An educational research prototype. A FastAPI backend serves a frozen, calibrated logistic-regression model trained on the CDC BRFSS 2025 survey. A React + Vite frontend collects the 19 survey answers, calls the API, and shows the estimate and a what-if comparison.

> **What the model estimates:** how likely a person with these answers is to *report existing* coronary heart disease or a past heart attack. It does **not** predict a future heart attack, does **not** diagnose anyone, and was built on **US** self-reported survey data that may not generalise to people in India. Full details: `docs/PROJECT_REPORT.md` and `docs/model_card.md`.

## Repository layout

| Path | What it is |
|---|---|
| `api/` | FastAPI app: `/predict`, `/health`, `/model-info`, fail-closed model loading |
| `src/` | Data pipeline, cleaning, preprocessing and training code. `src/modeling.py` defines `FinalModel`, which the API needs to load the model |
| `artifacts/` | Frozen model, preprocessors, metadata, metrics and figures. **Do not retrain or edit** without intending a new model |
| `data/processed/`, `data/interim/` | Train/validation/test splits and the Phase 1 subset (used by tests and reports) |
| `tests/` | Backend pytest suite (120 tests) |
| `docs/` | `PROJECT_REPORT.md` (architecture, data, models, results, limitations), model card, phase reports, feature specs |
| `frontend/` | React + Vite + TypeScript app, wired to the API. Includes Vitest unit tests in `src/__tests__/` and a browser flow check in `e2e/` |
| `render.yaml` | Render blueprint for the backend |
| `requirements-lock.txt` | Exact pins used to train and test (Python 3.14.2). `requirements.txt` holds the loose ranges |

## Setup: backend

Requires **Python 3.14** (tested on 3.14.6) and the exact pins in `requirements-lock.txt`. The model refuses to load if scikit-learn or the other pinned libraries differ.

```bash
cd Heart_Risk_Predictor
python3.14 -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-lock.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/health`: it should show `"status":"ok"` and `"model_loaded":true`. Swagger UI is at `/docs`.

Settings are environment variables: `API_ENV` (`production` by default, or `development`), `API_CORS_ORIGINS` (comma-separated explicit origins, never `*`), `API_ENABLE_DOCS`, `API_LOG_LEVEL`, `API_MAX_BODY_BYTES`. In development mode, CORS allows `localhost:5173` and `localhost:3000`.

## Setup: frontend

Requires **Node 22** and **pnpm 10**.

```bash
cd frontend
pnpm install --frozen-lockfile
cp .env.example .env.local                # VITE_API_BASE_URL=http://127.0.0.1:8000
pnpm dev                                  # http://localhost:5173
```

With the backend running locally, set `API_ENV=development` (or `API_CORS_ORIGINS=http://localhost:5173`), otherwise the browser will block the requests.

## Testing

Backend (run from the project root, 120 tests):

```bash
python -m pytest -q tests
```

The suite loads the real frozen model read-only. A session guard fails the run if any test changes `artifacts/`, `docs/` or `data/processed/`.

Frontend (run from `frontend/`):

```bash
pnpm test          # 13 unit tests: validation rules, backend contract, API error handling
pnpm typecheck     # TypeScript
pnpm build         # type-check, then production build into dist/
```

Browser flow (optional): start the backend with `API_ENV=production API_CORS_ORIGINS=http://127.0.0.1:5173`, run `pnpm preview --port 5173`, then `npm i -g playwright` and run `NODE_PATH=$(npm root -g) node e2e/flow-check.cjs`. It checks validation, submission, results and a simulator comparison.

## Deployment

Step-by-step instructions, including what only you can do in your own accounts, are in `docs/DEPLOYMENT.md`. The summary is below.

### 1. Backend on Render
1. Push the repository to GitHub. **Commit `artifacts/`**: the API loads `artifacts/model.joblib` from the repository. `.gitignore` excludes `data/`, `*.XPT` and `.venv/`, which the API does not need.
2. In Render, choose **New > Blueprint** and select the repo. `render.yaml` sets the build command (`pip install -r requirements-lock.txt`), the start command (`uvicorn api.main:app --host 0.0.0.0 --port $PORT`), the health check (`/health`) and `PYTHON_VERSION=3.14.6` (the version this project was tested on).
3. Set `API_CORS_ORIGINS` to your Vercel URL, for example `https://your-app.vercel.app`, with no trailing slash. Add other frontend origins, comma-separated.
4. Open `https://<your-service>.onrender.com/health`. The free instance sleeps when idle, so the first request can take about a minute.

Render's docs say `PYTHON_VERSION` must be a full version such as `3.14.6`; it takes precedence over any `.python-version` file. Do not change the pinned library versions without re-freezing the model. The model is refused at startup if scikit-learn or other pinned libraries differ.

### 2. Frontend on Vercel
1. Import the repo and set **Root Directory** to `frontend`.
2. Framework preset: **Vite**. Build command: `pnpm build`. Output directory: `dist`. Install command: `pnpm install --frozen-lockfile`.
3. Add the environment variable `VITE_API_BASE_URL=https://<your-service>.onrender.com` and deploy. It is read at build time, so redeploy after changing it.
4. Copy the Vercel URL into Render's `API_CORS_ORIGINS` (step 3 above).

**Secrets:** this project has no API keys or secrets. Do not add any to the repository.

## Changes made in this integration
- **Frontend wired to the API.** Figma placeholders were replaced with a real flow:
  - The questionnaire asks exactly the 19 model inputs, using the backend's codes and ranges. Client-side validation mirrors the backend, which remains the authority.
  - Results show the calibrated estimate (one decimal place), the threshold, the above/below label, and the backend's own interpretation and disclaimer.
  - The what-if simulator makes a second real `/predict` call and shows the difference in percentage points. It only changes model inputs (physical activity, smoking, heavy drinking).
  - The About page describes the real dataset, evaluation and logging.
- **Build tooling:** `pnpm build` type-checks before bundling. Vitest unit tests were added. `index.html` has a real title and language.
- **Backend:** no code changes. Added `render.yaml`.

## Model behaviour
- **Step-shaped outputs.** The isotonic calibrator is piecewise constant. Small input changes can leave the displayed probability unchanged for many profiles, so the simulator can show 0.0. This is a property of the frozen model.
- **Association, not causation.** The simulator and results say so.

## Missing components and manual steps
- **Framingham 10-year future-risk model: NOT INCLUDED.** Its model files, trained artifacts, and training code were not in the uploads, and could not be recovered from this environment. The app predicts only *reported existing* disease. Nothing has been created to stand in for the Framingham model; no Framingham endpoint, page or claim exists in this project. To add it later, provide the trained model file(s), the code that produced them, the input list, and the evaluation results. It would then be integrated as a separate endpoint and page with its own explanation.
- **Raw survey data: NOT INCLUDED in this ZIP.** `LLCP2025.XPT` (about 800 MB, CDC BRFSS 2025) is not needed to run the app, run the tests, or view the report. It is only needed to rerun `src/build_dataset.py`. If you need it, download the CDC BRFSS 2025 (LLCP2025) file from the CDC website and place it at the project root. It is git-ignored, so keep it out of the repository. `data/raw/` and `notebooks/` were empty in the upload.
- **Deployment accounts.** Creating the Render and Vercel services, setting the environment variables, and GitHub access all require your accounts. These steps were not performed or verified here.
- **Before submission:** confirm the Python 3.14.6 runtime on Render; deploy and check `/health`; open the live site and run one full assessment; and review the disclaimers and the report with your supervisor.

## Limitations
Self-reported US survey data; existing-condition target; subgroup performance varies (weaker under age 40); no authentication or rate limiting on the API; not a medical device. Do not use it for diagnosis or triage.
