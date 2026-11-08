"""Stage train_patchtst: main model. Multivariate PatchTST (HF transformers) with channel attention
so weather channels inform the AQI forecast. Forecasts 48h in one pass; 24h and 48h points are
evaluated on the same forecast origins as the tabular models."""
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import PatchTSTConfig, PatchTSTForPrediction

from src.evaluate.metrics import evaluate_predictions, forecast_plot, load_splits, save_model_report
from src.models.patchtst_pyfunc import CHANNELS, PatchTSTForecaster
from src.utils.common import (MODELS, PROCESSED, REPORTS, ROOT, get_logger, load_params, setup_mlflow,
                              standard_tags, write_json)

log = get_logger("train_patchtst")


def make_windows(arr: np.ndarray, origins: np.ndarray, L: int, H: int):
    past = np.stack([arr[o - L + 1:o + 1] for o in origins])
    fut = np.stack([arr[o + 1:o + H + 1] for o in origins])
    return torch.tensor(past, dtype=torch.float32), torch.tensor(fut, dtype=torch.float32)


def valid_origins(arr: np.ndarray, L: int, H: int) -> np.ndarray:
    ok = ~np.isnan(arr).any(axis=1)
    # window [o-L+1, o+H] fully observed  <=>  rolling count of ok over L+H ending at o+H == L+H
    c = np.concatenate([[0], np.cumsum(ok)])
    o = np.arange(L - 1, len(arr) - H)
    return o[(c[o + H + 1] - c[o - L + 1]) == L + H]


@torch.no_grad()
def predict(model, past: torch.Tensor, bs: int = 256) -> np.ndarray:
    model.eval()
    return torch.cat([model(past_values=past[i:i + bs]).prediction_outputs for i in range(0, len(past), bs)]).numpy()


def main() -> None:
    p = load_params()
    hp = p["patchtst"]
    L, H = hp["context_length"], hp["prediction_length"]
    horizons = p["features"]["horizons"]
    torch.manual_seed(hp["seed"])
    np.random.seed(hp["seed"])

    series = pd.read_parquet(PROCESSED / "series.parquet")[CHANNELS]
    splits = load_splits()
    train_end = pd.Timestamp(p["split"]["train_end"], tz="UTC")
    val_end = pd.Timestamp(p["split"]["val_end"], tz="UTC")

    train_mask = series.index <= train_end
    mean = series[train_mask].mean().to_numpy()
    std = series[train_mask].std().replace(0, 1).to_numpy()
    arr = ((series - mean) / std).to_numpy(float)

    idx = series.index
    origins = valid_origins(arr, L, H)
    o_time = idx[origins]
    gap = pd.Timedelta(hours=H)
    sets = {
        "train": origins[o_time <= train_end - gap][::hp["train_stride"]],
        "val": origins[np.isin(o_time, splits["val"].index)],
        "test": origins[np.isin(o_time, splits["test"].index)],
    }
    log.info("windows: %s", {k: len(v) for k, v in sets.items()})
    data = {k: make_windows(arr, v, L, H) for k, v in sets.items()}

    cfg = PatchTSTConfig(
        num_input_channels=len(CHANNELS), context_length=L, prediction_length=H,
        patch_length=hp["patch_length"], patch_stride=hp["patch_stride"], d_model=hp["d_model"],
        num_attention_heads=hp["num_attention_heads"], num_hidden_layers=hp["num_hidden_layers"],
        ffn_dim=hp["ffn_dim"], dropout=hp["dropout"], channel_attention=True, scaling="std", loss="mse")
    model = PatchTSTForPrediction(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"])

    def aqi_units(pred_or_true):  # channel 0 back to AQI
        return pred_or_true[..., 0] * std[0] + mean[0]

    mlflow = setup_mlflow()
    with mlflow.start_run(run_name="patchtst") as run:
        mlflow.set_tags({**standard_tags(), "model_family": "patchtst", "role": "champion"})
        mlflow.log_params({**hp, "channels": ",".join(CHANNELS), "n_train_windows": len(sets["train"]),
                           "n_params": sum(t.numel() for t in model.parameters())})
        Xtr, Ytr = data["train"]
        best, best_state, history = np.inf, None, []
        for epoch in range(hp["epochs"]):
            model.train()
            perm = torch.randperm(len(Xtr))
            total = 0.0
            for i in range(0, len(perm), hp["batch_size"]):
                b = perm[i:i + hp["batch_size"]]
                loss = model(past_values=Xtr[b], future_values=Ytr[b]).loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(b)
            vp = predict(model, data["val"][0])
            val_mae = float(np.mean(np.abs(aqi_units(vp) - aqi_units(data["val"][1].numpy()))))
            history.append({"epoch": epoch, "train_loss": total / len(Xtr), "val_aqi_mae": val_mae})
            mlflow.log_metrics({"train_loss": total / len(Xtr), "val_aqi_mae": val_mae}, step=epoch)
            log.info("epoch %d train_loss %.4f val_aqi_mae %.2f", epoch, total / len(Xtr), val_mae)
            if val_mae < best:
                best, best_state = val_mae, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)

        preds, test_pred = {}, None
        for s in ("val", "test"):
            yp = aqi_units(predict(model, data[s][0]))
            yt = aqi_units(data[s][1].numpy())
            preds[s] = {h: (yt[:, h - 1], yp[:, h - 1].clip(0, 500)) for h in horizons}
        metrics = evaluate_predictions(preds)
        mlflow.log_metrics({**metrics, "best_val_aqi_mae": best})

        # artifacts: loss curve + forecast plots
        hist = pd.DataFrame(history)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hist["epoch"], hist["train_loss"], label="train loss (scaled MSE)")
        ax2 = ax.twinx()
        ax2.plot(hist["epoch"], hist["val_aqi_mae"], color="tab:orange", label="val AQI MAE")
        ax.set_xlabel("epoch")
        fig.legend(loc="upper right")
        fig.tight_layout()
        curve = REPORTS / "plots" / "patchtst_loss_curve.png"
        curve.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(curve, dpi=110)
        plt.close(fig)
        mlflow.log_artifact(str(curve), artifact_path="plots")
        test_idx = idx[sets["test"]]
        for h in horizons:
            plot = REPORTS / "plots" / f"patchtst_h{h}_test.png"
            forecast_plot(test_idx, *preds["test"][h], f"patchtst - {h}h ahead (test, smog season)", plot)
            mlflow.log_artifact(str(plot), artifact_path="plots")

        # save model locally (DVC output) + log as MLflow pyfunc for the registry
        mdir = MODELS / "patchtst"
        shutil.rmtree(mdir, ignore_errors=True)
        model.save_pretrained(mdir)
        write_json(mdir / "scaler.json", {"channels": CHANNELS, "mean": mean.tolist(), "std": std.tolist()})
        example = series.iloc[-L:].reset_index(drop=True)
        mlflow.pyfunc.log_model(
            artifact_path="model", python_model=PatchTSTForecaster(),
            artifacts={"model_dir": str(mdir), "scaler": str(mdir / "scaler.json")},
            code_paths=[str(ROOT / "src")], input_example=example,
            pip_requirements=["torch", "transformers", "pandas", "numpy", "mlflow"])
        save_model_report("patchtst", metrics, run.info.run_id)
        log.info("patchtst test: %s", {k: round(v, 2) for k, v in metrics.items() if k.startswith("test")})


if __name__ == "__main__":
    main()
