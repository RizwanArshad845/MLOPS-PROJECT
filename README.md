# SmogCast – Lahore AQI Forecasting (MLOps)

SmogCast forecasts Lahore's Air Quality Index **24h and 48h ahead** from hourly PM2.5 and weather history.
Milestone 1 is a fully reproducible core ML pipeline: ingestion → validation → features → baseline and main models → evaluation → model registry. It uses DVC for data/pipeline versioning and MLflow for experiment tracking, both hosted on **DagsHub** (free, no AWS).

- **GitHub:** https://github.com/RizwanArshad845/MLOPS-PROJECT
- **DagsHub (DVC remote):** https://dagshub.com/RizwanArshad845/ML_PRO
- **MLflow tracking server:** https://dagshub.com/RizwanArshad845/ML_PRO.mlflow

## Pipeline

```
ingest_aqi ─┐
            ├─> validate ─> featurize ─┬─> train_baselines ─┐
ingest_weather ┘                        ├─> train_gbm ───────┼─> evaluate (+ register in Staging)
                                        └─> train_patchtst ──┘
```

| Stage | What it does |
|---|---|
| `ingest_aqi` | Hourly PM2.5 from OpenAQ v3 ground sensors near Lahore, gaps filled from the Open-Meteo Air Quality API |
| `ingest_weather` | Hourly temperature, humidity, wind, pressure, precipitation from the Open-Meteo archive |
| `validate` | pandera schema/range checks, null fractions, hourly continuity, stuck-sensor detection → `reports/validation.json` (fails the pipeline on error) |
| `featurize` | PM2.5 → US-EPA AQI, lags, rolling stats, cyclic calendar features, 24h/48h targets, chronological split (test = Oct–Dec 2024 smog season) |
| `train_baselines` | Persistence and Ridge regression |
| `train_gbm` | XGBoost and LightGBM (comparison benchmark) |
| `train_patchtst` | **Main model:** multivariate PatchTST transformer (Hugging Face) |
| `evaluate` | Comparison table, `reports/metrics.json`, registers `smogcast-patchtst` in the MLflow Model Registry → **Staging** |

Metrics: MAE, RMSE, sMAPE and AQI-category accuracy per horizon. Every run logs params, metrics, plots and the model to MLflow, tagged with the git commit and data hashes.

## Setup

```bash
git clone https://github.com/RizwanArshad845/MLOPS-PROJECT.git && cd MLOPS-PROJECT
python -m venv .venv && .venv\Scripts\activate      # Windows (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
copy .env.example .env                              # then fill in the keys
```

`.env` needs:
- `OPENAQ_API_KEY` – free key from explore.openaq.org (optional; without it only Open-Meteo PM2.5 is used)
- `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` – from the DagsHub repo's *Remote → Experiments* tab (the password is a DagsHub access token)

DVC remote credentials (stored locally, never committed):
```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <dagshub-username>
dvc remote modify origin --local password <dagshub-token>
```

## How to run

```bash
dvc pull            # get the versioned data/models from DagsHub (optional)
dvc repro           # run / reproduce the whole pipeline from scratch
dvc push            # upload new data + model versions to DagsHub
dvc dag             # pipeline diagram
dvc metrics show    # model comparison
```

All configuration (dates, splits, features, hyperparameters, registry stage) lives in `params.yaml`. Change a value and run `dvc repro`; only the affected stages re-run.

## Folder structure

```
├── data/                 # DVC-managed (git-ignored): raw/ and processed/
├── models/               # DVC-managed trained models
├── reports/              # validation.json, metrics/*.json, metrics.json, comparison.csv
├── src/
│   ├── ingest/           # aqi.py (OpenAQ + Open-Meteo AQ), weather.py, http.py (retrying client)
│   ├── validate/         # validate.py (pandera checks)
│   ├── features/         # build_features.py
│   ├── models/           # tabular.py (persistence/ridge/xgboost/lightgbm), patchtst.py, patchtst_pyfunc.py
│   ├── evaluate/         # metrics.py, evaluate.py (comparison + registry)
│   └── utils/            # common.py (params, MLflow setup, AQI conversion)
├── docs/PLAN.md          # project plan
├── dvc.yaml / dvc.lock   # pipeline definition / locked state
├── params.yaml
└── requirements.txt
```

## Using the registered model

```python
import mlflow
model = mlflow.pyfunc.load_model("models:/smogcast-patchtst/Staging")
forecast = model.predict(last_168_hours_df)   # -> 48 hourly AQI forecasts
```

## Team
Rizwan Arshad (23L-0681), Rameel Naveed (23L-0683)
