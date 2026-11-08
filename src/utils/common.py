"""Shared helpers: params loading, paths, MLflow setup, AQI conversion."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def load_params() -> dict:
    with open(ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def setup_mlflow(experiment: str = "smogcast"):
    """Point MLflow at DagsHub (MLFLOW_TRACKING_URI + username/password env vars).

    Falls back to a local ./mlruns store only so development runs don't crash;
    graded runs must use the DagsHub server.
    """
    import mlflow

    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        get_logger("mlflow").warning("MLFLOW_TRACKING_URI not set - logging to local ./mlruns")
        uri = (ROOT / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return mlflow


def standard_tags() -> dict:
    tags = {"git_commit": git_commit()}
    for name in ("train.parquet", "val.parquet", "test.parquet"):
        p = PROCESSED / name
        if p.exists():
            tags[f"data_md5_{name.split('.')[0]}"] = file_md5(p)
    return tags


# US EPA PM2.5 AQI breakpoints (2024 revision): (C_lo, C_hi, I_lo, I_hi)
_PM25_BREAKPOINTS = [
    (0.0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
]
AQI_CATEGORY_EDGES = [50, 100, 150, 200, 300]  # upper edges of categories 0..4, 5 = hazardous


def pm25_to_aqi(pm25) -> np.ndarray:
    c = np.floor(np.asarray(pm25, dtype=float) * 10) / 10  # EPA truncation to 0.1
    out = np.full(c.shape, np.nan)
    for c_lo, c_hi, i_lo, i_hi in _PM25_BREAKPOINTS:
        m = np.isnan(out) & (c >= 0) & (c <= c_hi + 1e-9)
        out[m] = (i_hi - i_lo) / (c_hi - c_lo) * (c[m] - c_lo) + i_lo
    above = c > _PM25_BREAKPOINTS[-1][1]
    out[above] = 500.0  # cap at index max
    return np.round(out)


def aqi_category(aqi) -> np.ndarray:
    return np.digitize(np.asarray(aqi, dtype=float), AQI_CATEGORY_EDGES, right=True)
