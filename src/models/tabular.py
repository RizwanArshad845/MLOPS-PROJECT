"""Stages train_baselines / train_gbm: one MLflow run per model, one regressor per horizon.

Usage: python -m src.models.tabular persistence ridge
       python -m src.models.tabular xgboost lightgbm
"""
import sys

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate.metrics import (evaluate_predictions, feature_cols, forecast_plot, load_splits,
                                  save_model_report)
from src.utils.common import MODELS, REPORTS, get_logger, load_params, setup_mlflow, standard_tags

log = get_logger("train")


def fit_one(name: str, p: dict, Xtr, ytr, Xva, yva):
    if name == "ridge":
        m = make_pipeline(StandardScaler(), Ridge(alpha=p["ridge"]["alpha"]))
        return m.fit(Xtr, ytr)
    if name == "xgboost":
        hp = dict(p["xgboost"])
        m = xgb.XGBRegressor(**hp, random_state=42, n_jobs=-1)
        return m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    if name == "lightgbm":
        hp = dict(p["lightgbm"])
        es = hp.pop("early_stopping_rounds")
        m = lgb.LGBMRegressor(**hp, subsample_freq=1, random_state=42, verbose=-1)
        return m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(es, verbose=False)])
    raise ValueError(name)


def train(name: str, p: dict, splits: dict[str, pd.DataFrame], mlflow) -> None:
    horizons = p["features"]["horizons"]
    feats = feature_cols(splits["train"])
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({**standard_tags(), "model_family": name,
                         "role": "baseline" if name in ("persistence", "ridge") else "comparison"})
        mlflow.log_params({"model": name, "horizons": horizons, "n_features": len(feats),
                           "n_train": len(splits["train"]), **p.get(name, {})})
        preds = {"val": {}, "test": {}}
        for h in horizons:
            tgt = f"target_{h}h"
            if name == "persistence":
                model = None
                for s in preds:
                    preds[s][h] = (splits[s][tgt], splits[s]["aqi"])
            else:
                model = fit_one(name, p, splits["train"][feats], splits["train"][tgt],
                                splits["val"][feats], splits["val"][tgt])
                for s in preds:
                    preds[s][h] = (splits[s][tgt], model.predict(splits[s][feats]))
                path = MODELS / f"{name}_h{h}.joblib"
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, path)
                mlflow.log_artifact(str(path), artifact_path="model")
                if hasattr(model, "feature_importances_"):
                    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
                    ip = REPORTS / "artifacts" / f"{name}_h{h}_importance.csv"
                    ip.parent.mkdir(parents=True, exist_ok=True)
                    imp.to_csv(ip, header=["importance"])
                    mlflow.log_artifact(str(ip), artifact_path="feature_importance")
            yt, yp = preds["test"][h]
            plot = REPORTS / "plots" / f"{name}_h{h}_test.png"
            forecast_plot(splits["test"].index, yt, yp, f"{name} - {h}h ahead (test, smog season)", plot)
            mlflow.log_artifact(str(plot), artifact_path="plots")
        metrics = evaluate_predictions(preds)
        mlflow.log_metrics(metrics)
        save_model_report(name, metrics, run.info.run_id)
        log.info("%s test: %s", name, {k: round(v, 2) for k, v in metrics.items() if k.startswith("test")})


def main(names: list[str]) -> None:
    p = load_params()
    mlflow = setup_mlflow()
    splits = load_splits()
    for name in names:
        train(name, p, splits, mlflow)


if __name__ == "__main__":
    main(sys.argv[1:])
