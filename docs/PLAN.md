# SmogCast – Milestone 1 Plan (Reproducible Core ML Pipeline)

## Context
The SmogCast proposal (Rizwan Arshad, Rameel Naveed) describes a full MLOps system for Lahore AQI forecasting. This milestone covers only the **core pipeline**: ingest → validate → version → train (baseline + main) → evaluate → track/register. The TA brief rules out Docker, cloud deployment, REST APIs and monitoring for now. The team wants **free platforms only**, so no paid AWS.

## Is "DagsHub instead of AWS" OK? Yes, the TA allows it
- **DVC remote:** the brief says "DagsHub **or** AWS S3", so DagsHub is allowed. Every DagsHub repo comes with free DVC storage.
- **MLflow:** the brief says it must be "hosted on DagsHub **or** a remote server, not local". Every DagsHub repo also comes with a free hosted MLflow server, which covers this.
- **One correction:** the proposal's MLOps Stack section still says *AWS S3 / ECR / EC2 / DVC with S3 remote*. Mention the switch to DagsHub in the write-up so the TA doesn't flag the mismatch. For later milestones, free stand-ins exist: GitHub Container Registry instead of ECR, and Render, Hugging Face Spaces or Fly.io instead of EC2. Plan for this now, but nothing is needed this milestone.
- **Out of scope for now** (save effort): Prefect, Evidently, FastAPI, Docker, Prometheus/Grafana, CI/CD. Hourly scheduled ingestion is also later; this milestone ingests a **fixed date range** so `dvc repro` is deterministic.

## Things to watch out for (where the proposal and the brief might conflict)
1. **OpenAQ v3 needs a free API key**, and Lahore's sensor coverage has gaps. Fallback: Open-Meteo's **Air Quality API** (free, no key) gives hourly PM2.5 history for Lahore's coordinates. Plan: use OpenAQ as the primary source, fill gaps from Open-Meteo AQ, and log the gap ratio as a validation metric.
2. **MLflow "Staging" stage:** stages are deprecated in MLflow ≥2.9 and replaced by aliases. To get the "Staging" screenshot the TA wants, pin the `mlflow` version to the one DagsHub's server supports (2.x) and call `transition_model_version_stage(..., "Staging")`. Also set the alias `staging`.
3. **PatchTST on CPU:** use the Hugging Face `transformers` `PatchTSTForPrediction` (supports multivariate channels: AQI + weather). Keep it small (context 168h, horizon 48h, a few epochs) so it trains in minutes on a laptop.
4. **Every group member needs commits** (TA requirement). See Ownership below.

## Phases

### Phase 0 – Accounts & setup
- Create a GitHub repo and connect it to DagsHub. Add both members as collaborators.
- Get a free OpenAQ API key and a DagsHub access token.
- Put secrets in `.env` (gitignored) and `.dvc/config.local`. Never commit them.

### Phase 1 – Repo skeleton
```
smogcast/
├── data/ (raw/ interim/ processed/)   # DVC-tracked, gitignored
├── models/  reports/  plots/          # DVC outputs, gitignored
├── src/
│   ├── ingest/ (aqi.py = OpenAQ + Open-Meteo AQ fallback, weather.py, http.py)
│   ├── validate/validate.py
│   ├── features/build_features.py
│   ├── models/ (tabular.py = persistence/ridge/xgboost/lightgbm, patchtst.py, patchtst_pyfunc.py)
│   ├── evaluate/ (metrics.py, evaluate.py = comparison + registry)
│   └── utils/common.py
├── params.yaml   dvc.yaml   requirements.txt   .gitignore   .env.example   README.md
```
- Initialize git and DVC. Add DagsHub as the default DVC remote.
- Set up `.gitignore` to exclude `data/`, `models/`, `.env`, `mlruns/` and `*.pt`/`*.pkl`.
- Fill `params.yaml` with: city coordinates, date range (e.g. 2023-01-01 → 2024-12-31), splits (train ≤ 2024-06, val 2024-07→09, **test 2024-10→12 = smog season**), horizons [24, 48], lags, and hyperparameters for every model.

### Phase 2 – Ingestion + DVC versioning
- `ingest_aqi`: OpenAQ v3 → hourly PM2.5 for Lahore stations → `data/raw/aqi.csv`. Fill gaps from Open-Meteo AQ.
- `ingest_weather`: Open-Meteo archive API → temperature, humidity, wind speed/direction, pressure, precipitation → `data/raw/weather.csv`.
- Add retries and timeouts, and fail loudly on empty responses.
- `dvc push` to DagsHub.

### Phase 3 – Validation
- `validate` stage uses **pandera** (lightweight; Great Expectations is overkill for this milestone and can come in with Evidently later) to check:
  - schema (columns and dtypes)
  - nulls (≤ threshold from params)
  - ranges (PM2.5 0–1000, humidity 0–100, temperature −5–50 °C, wind ≥ 0)
  - no duplicate timestamps, monotonic hourly index
  - stuck-sensor detection (same value for N hours in a row)
- Writes `reports/validation.json` and stops the pipeline (non-zero exit) if a check fails.

### Phase 4 – Features
- `featurize`: merge the two sources, resample hourly, interpolate small gaps, compute AQI from PM2.5 (US EPA breakpoints).
- Add lag features (1, 2, 3, 6, 12, 24, 48h), rolling means, and hour/day-of-week/month cyclic features.
- Build targets AQI(t+24) and AQI(t+48), then split chronologically into `data/processed/{train,val,test}.parquet`.

### Phase 5 – Models
- **Baselines:** Persistence (AQI(t+h) = AQI(t)) and Ridge regression.
- **Comparison model:** XGBoost and LightGBM, one model per horizon.
- **Main model:** PatchTST via HF transformers. Forecasts 48 hours ahead in one pass; evaluate the 24h and 48h points.
- Every training script logs to DagsHub MLflow:
  - params
  - metrics: MAE, RMSE, sMAPE per horizon, plus AQI-category accuracy
  - artifacts: model, forecast-vs-actual plot, feature importance (XGBoost), loss curve (PatchTST)
- Tag each run with the git commit and the DVC data hash.

### Phase 6 – Evaluate & register
- `evaluate`: compare all models on the test split and write `reports/metrics.json` (tracked as DVC metrics, so `dvc metrics show` works).
- Register PatchTST as `smogcast-patchtst` and move it to **Staging**. If PatchTST is beaten by XGBoost, still register it (it is the mandated champion) and record the comparison honestly. Optionally register XGBoost too.

### Phase 7 – dvc.yaml & reproducibility
Stages: `ingest_aqi → ingest_weather → validate → featurize → train_baselines → train_gbm → train_patchtst → evaluate`.
- Each stage declares its `deps`, `params` and `outs`.
- **Test:** clone fresh → `pip install -r requirements.txt` → set `.env` → `dvc repro` runs end to end. `dvc dag` gives the diagram for the report.

### Phase 8 – README + PDF report
- README covers: description, setup, env vars, `dvc pull` / `dvc repro`, folder tree, and links.
- PDF (3–4 pages) includes:
  - GitHub and DagsHub/MLflow links
  - screenshots of ≥2 MLflow runs, the model in Staging and `dvc dag`
  - a few sentences per component

## Ownership
- **Rizwan does all phases** (code, DVC, MLflow, README, report).
- ⚠️ The TA brief says "All group members must have commits." If Rameel has zero commits, marks may be lost. Minimum fix: Rameel makes at least a few real commits from their own GitHub account, e.g. a README section, the report draft, or reviewing and tweaking `params.yaml`.

## First action after approval
Copy this plan into the project repo as `docs/PLAN.md` (committed to git), then start Phase 0/1.

## Verification
- `dvc repro` from a fresh clone succeeds. Running `dvc repro` again reports "Data and pipelines are up to date".
- `dvc status -c` shows the remote in sync after `dvc push`.
- The DagsHub MLflow UI shows runs for all 4+ models with metrics and artifacts, and `smogcast-patchtst` vN is in Staging.
- Corrupting a value in raw data (e.g. PM2.5 = −5) makes the `validate` stage fail.
- `git log` shows commits from both members, and `git ls-files` contains no data or model files.

## Decisions I'll assume unless you say otherwise
- Use pandera instead of Great Expectations for this milestone.
- Use Hugging Face PatchTST instead of neuralforecast, because neuralforecast's PatchTST can't use the weather features.
- Use the date range 2023-01 → 2024-12, with the smog season as the test set.
