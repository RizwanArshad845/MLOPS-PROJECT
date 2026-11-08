"""Forecast metrics + helpers shared by all training scripts."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.common import PROCESSED, REPORTS, aqi_category, write_json


def load_splits() -> dict[str, pd.DataFrame]:
    return {s: pd.read_parquet(PROCESSED / f"{s}.parquet") for s in ("train", "val", "test")}


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("target_")]


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "smape": float(np.mean(np.where(denom == 0, 0, np.abs(err) / np.where(denom == 0, 1, denom))) * 100),
        "category_acc": float(np.mean(aqi_category(y_true) == aqi_category(y_pred))),
    }


def evaluate_predictions(preds: dict[str, dict[int, tuple]]) -> dict:
    """preds[split][horizon] = (y_true, y_pred) -> flat {split_h24_mae: ...} dict."""
    out = {}
    for split, by_h in preds.items():
        for h, (yt, yp) in by_h.items():
            for k, v in regression_metrics(yt, yp).items():
                out[f"{split}_h{h}_{k}"] = v
    return out


def forecast_plot(index, y_true, y_pred, title: str, path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(index, y_true, label="actual", lw=1)
    ax.plot(index, y_pred, label="forecast", lw=1)
    ax.set_title(title)
    ax.set_ylabel("AQI")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_model_report(model: str, metrics: dict, run_id: str) -> None:
    write_json(REPORTS / "metrics" / f"{model}.json", {"run_id": run_id, **metrics})
