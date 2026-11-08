"""Stage evaluate: compare all models on the test split (smog season), write reports/metrics.json,
and register the PatchTST run in the MLflow Model Registry at the configured stage (Staging)."""
import json

import pandas as pd
from mlflow.tracking import MlflowClient

from src.utils.common import REPORTS, get_logger, load_params, setup_mlflow, write_json

log = get_logger("evaluate")
MODELS = ["persistence", "ridge", "xgboost", "lightgbm", "patchtst"]


def register(run_id: str, p: dict, mlflow, summary: dict) -> dict:
    reg = p["registry"]
    mv = mlflow.register_model(f"runs:/{run_id}/model", reg["model_name"])
    client = MlflowClient()
    client.transition_model_version_stage(reg["model_name"], mv.version, reg["stage"],
                                          archive_existing_versions=False)
    try:  # aliases are the MLflow >= 2.9 replacement for stages; set both
        client.set_registered_model_alias(reg["model_name"], reg["stage"].lower(), mv.version)
    except Exception as e:  # older servers
        log.warning("alias not set: %s", e)
    for k, v in summary.items():
        client.set_model_version_tag(reg["model_name"], mv.version, k, str(v))
    log.info("Registered %s v%s -> %s", reg["model_name"], mv.version, reg["stage"])
    return {"name": reg["model_name"], "version": mv.version, "stage": reg["stage"]}


def main() -> None:
    p = load_params()
    reports = {m: json.loads((REPORTS / "metrics" / f"{m}.json").read_text(encoding="utf-8")) for m in MODELS}
    rows = []
    for m, r in reports.items():
        for h in p["features"]["horizons"]:
            rows.append({"model": m, "horizon_h": h,
                         **{k: r[f"test_h{h}_{k}"] for k in ("mae", "rmse", "smape", "category_acc")}})
    table = pd.DataFrame(rows).sort_values(["horizon_h", "mae"])
    print(table.to_string(index=False))
    table.to_csv(REPORTS / "comparison.csv", index=False)

    h_main = max(p["features"]["horizons"])
    best = table[table.horizon_h == h_main].iloc[0]["model"]
    beats_persistence = all(
        reports["patchtst"][f"test_h{h}_mae"] < reports["persistence"][f"test_h{h}_mae"]
        for h in p["features"]["horizons"])
    summary = {"best_model_h48_mae": best, "patchtst_beats_persistence": beats_persistence}

    mlflow = setup_mlflow()
    registered = register(reports["patchtst"]["run_id"], p, mlflow, summary)

    # flat metrics file for `dvc metrics show`
    flat = {f"{m}_h{h}_{k}": round(reports[m][f"test_h{h}_{k}"], 4)
            for m in MODELS for h in p["features"]["horizons"] for k in ("mae", "rmse")}
    write_json(REPORTS / "metrics.json", {**flat, **summary, "registered_version": registered["version"]})


if __name__ == "__main__":
    main()
