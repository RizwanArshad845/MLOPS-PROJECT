"""MLflow pyfunc wrapper so the registered PatchTST model can be loaded and served with one call:

    model = mlflow.pyfunc.load_model("models:/smogcast-patchtst/Staging")
    model.predict(last_168_hours_df)   # columns = CHANNELS -> DataFrame of 48 hourly AQI forecasts
"""
import json

import mlflow.pyfunc
import numpy as np
import pandas as pd

CHANNELS = ["aqi", "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
            "surface_pressure", "precipitation", "wind_u", "wind_v"]


class PatchTSTForecaster(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        from transformers import PatchTSTForPrediction

        self.torch = torch
        self.model = PatchTSTForPrediction.from_pretrained(context.artifacts["model_dir"]).eval()
        with open(context.artifacts["scaler"], encoding="utf-8") as f:
            sc = json.load(f)
        self.mean, self.std = np.array(sc["mean"]), np.array(sc["std"])
        self.context_length = self.model.config.context_length

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        x = model_input[CHANNELS].to_numpy(float)[-self.context_length:]
        if len(x) < self.context_length or np.isnan(x).any():
            raise ValueError(f"need {self.context_length} complete hourly rows of {CHANNELS}")
        x = (x - self.mean) / self.std
        with self.torch.no_grad():
            out = self.model(past_values=self.torch.tensor(x[None], dtype=self.torch.float32)).prediction_outputs
        aqi = out[0, :, 0].numpy() * self.std[0] + self.mean[0]
        return pd.DataFrame({"horizon_h": np.arange(1, len(aqi) + 1), "aqi_forecast": aqi.clip(0, 500)})
