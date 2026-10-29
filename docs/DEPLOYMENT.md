# Deployment guide: Render (backend) and Vercel (frontend)

You need accounts on **GitHub**, **Render**, and **Vercel**. Nothing in this project stores secrets or API keys, so no keys are needed. I could not access your accounts, so none of these steps were run by me.

Order matters: deploy the backend first (you need its URL), then the frontend, then set the backend's CORS origin to the frontend URL.

## Part 0: Put the project on GitHub

1. Unzip `Heart_Risk_Predictor_Final.zip` to a folder, for example `Heart_Risk_Predictor`.
2. Create a new **private** repository on GitHub, for example `heart-risk-predictor`. Do not add a README or .gitignore on GitHub (the project has its own).
3. In the project folder, run:
   ```bash
   git init
   git add .
   git commit -m "Initial submission"
   git branch -M main
   git remote add origin https://github.com/<your-user>/heart-risk-predictor.git
   git push -u origin main
   ```
4. **Check before pushing** (run `git status`): `artifacts/` must be listed, and `LLCP2025.XPT`, `.venv/`, `node_modules/` and `.env*` files must **not** be listed. The `.gitignore` already handles this. If `artifacts/` is missing, check `git check-ignore -v artifacts/model.joblib`.

## Part 1: Backend on Render

1. Sign in at render.com and click **New > Blueprint**.
2. Connect your GitHub account if asked, select the `heart-risk-predictor` repository, and choose the `main` branch.
3. Render reads `render.yaml` and shows one web service: `heart-risk-api`. The settings it applies are:
   - Runtime: Python, free plan
   - Build command: `pip install -r requirements-lock.txt`
   - Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - Health check path: `/health`
   - `PYTHON_VERSION=3.14.6`, `API_ENV=production`, `API_ENABLE_DOCS=true`, `API_LOG_LEVEL=INFO`
4. Render will ask for **`API_CORS_ORIGINS`** (it is marked "sync: false"). For now, enter `http://127.0.0.1:5173` as a placeholder. You will change it in Part 3. Click **Apply** or **Deploy**.
5. Wait for the build to finish (several minutes the first time). Check the logs for a line saying the application started, and for no errors about `scikit-learn` or the model.
6. Note your backend URL, for example `https://heart-risk-api.onrender.com`.
7. Test it in a browser:
   - `https://<your-backend>/health` should show `"status":"ok"` and `"model_loaded":true`.
   - `https://<your-backend>/docs` should open Swagger UI (it loads its assets from a public CDN).

**If `/health` shows `"status":"degraded"`:** the model did not load. Check the Render logs. The most likely causes are a Python version other than 3.14.6, or `artifacts/` not committed to GitHub.

**Free plan note:** the service sleeps after about 15 minutes idle. The first request afterwards can take about a minute. The frontend already shows a loading message for this.

## Part 2: Frontend on Vercel

1. Sign in at vercel.com and click **Add New > Project**. Import the `heart-risk-predictor` repository from GitHub.
2. Set these project settings:
   - **Framework Preset:** Vite
   - **Root Directory:** `frontend` (click Edit and choose the `frontend` folder)
   - **Build Command:** `pnpm build` (the default is fine; it runs the type-check and then the Vite build)
   - **Output Directory:** `dist`
   - **Install Command:** `pnpm install --frozen-lockfile`
   - **Node.js version:** 22.x (the project was tested on Node 22)
3. Under **Environment Variables**, add:
   - Name: `VITE_API_BASE_URL`
   - Value: `https://<your-backend>.onrender.com` (no trailing slash)
   - Environments: Production (and Preview if you want preview deployments to work)
4. Click **Deploy**. Note the URL Vercel gives you, for example `https://heart-risk-predictor.vercel.app`.

Vercel reads `VITE_*` variables at **build** time. If you change `VITE_API_BASE_URL` later, you must redeploy.

## Part 3: Connect the two (required)

1. Go back to Render > `heart-risk-api` > **Environment**.
2. Set `API_CORS_ORIGINS` to your exact Vercel URL, with no trailing slash, for example `https://heart-risk-predictor.vercel.app`. To allow more origins, separate them with commas.
3. Save. Render redeploys the service.
4. Open `https://<your-frontend>.vercel.app`, click **Start Assessment**, complete the form and submit. You should see your result. If the browser shows "could not be reached", check item 2 again (the exact URL, with `https://` and no trailing slash) and make sure the backend is awake.

Preview deployments from Vercel have different URLs. If you need them to call the API, add each preview URL to `API_CORS_ORIGINS`, or test only on production.

## Part 4: Final checks on the live site

- [ ] `https://<backend>/health` shows `"status":"ok"`
- [ ] Frontend loads with the Manrope font and no blank page
- [ ] An assessment with blank answers shows validation and sends no request
- [ ] A complete assessment shows a result, the threshold, the disclaimer, and the "not a diagnosis" wording
- [ ] The What-If Simulator compares two results
- [ ] About page and footer disclaimer are visible on phone width

## Things only you can do

1. Create the GitHub, Render and Vercel accounts (or sign in) and grant access to the repository.
2. Push the repository and check that `artifacts/` is included and the raw survey file is not.
3. Create the two services and set the environment variables described above.
4. Confirm the backend's Python version on Render's build log (it should say 3.14.6).
5. Share the frontend URL only after the checks in Part 4 pass.
6. Decide whether to keep `API_ENABLE_DOCS=true` on the public backend. It is harmless, but you can set it to `false` in Render.
7. Review the disclaimers and `docs/PROJECT_REPORT.md` with your supervisor before submitting.

## What I could not verify

- Render and Vercel deployments themselves (no access to your accounts).
- Render's free-plan cold start time.
- That Vercel's project settings screen matches the labels above exactly (Vercel changes its UI from time to time).
