"""Stage featurize: merge AQI + weather, convert PM2.5 -> AQI, build lag/rolling/calendar
features and 24h/48h targets, then split chronologically into train/val/test.

Outputs:
  data/processed/series.parquet           continuous hourly series (used by PatchTST)
  data/processed/{train,val,test}.parquet tabular features + targets (baselines, GBMs)
"""
import numpy as np
import pandas as pd

from src.utils.common import PROCESSED, RAW, get_logger, load_params, pm25_to_aqi

log = get_logger("featurize")


def build_series(p: dict) -> pd.DataFrame:
    aqi = pd.read_csv(RAW / "aqi.csv", parse_dates=["timestamp"])[["timestamp", "pm25"]]
    weather = pd.read_csv(RAW / "weather.csv", parse_dates=["timestamp"])
    df = aqi.merge(weather, on="timestamp", how="outer").set_index("timestamp").sort_index()
    df = df.asfreq("h")
    gap = p["features"]["max_interp_gap_hours"]
    df = df.interpolate(method="time", limit=gap, limit_area="inside")
    df["aqi"] = pm25_to_aqi(df["pm25"])
    # wind direction as vector components (360 deg == 0 deg)
    rad = np.deg2rad(df.pop("wind_direction_10m"))
    df["wind_u"] = np.sin(rad)
    df["wind_v"] = np.cos(rad)
    return df


def build_tabular(series: pd.DataFrame, p: dict) -> pd.DataFrame:
    f = p["features"]
    df = series.copy()
    for lag in f["lags"]:
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)
    for w in f["rolling_windows"]:
        df[f"aqi_roll_mean_{w}"] = df["aqi"].rolling(w).mean()
        df[f"aqi_roll_std_{w}"] = df["aqi"].rolling(w).std()
    idx = df.index
    for name, val, period in (("hour", idx.hour, 24), ("dow", idx.dayofweek, 7), ("month", idx.month - 1, 12)):
        df[f"{name}_sin"] = np.sin(2 * np.pi * val / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * val / period)
    for h in f["horizons"]:
        df[f"target_{h}h"] = df["aqi"].shift(-h)
    return df.dropna()


def split(df: pd.DataFrame, p: dict) -> dict[str, pd.DataFrame]:
    """Chronological split on forecast-origin time. The last max(horizon) hours of train/val are
    dropped so no training target leaks into the next period."""
    gap = pd.Timedelta(hours=max(p["features"]["horizons"]))
    train_end = pd.Timestamp(p["split"]["train_end"], tz="UTC")
    val_end = pd.Timestamp(p["split"]["val_end"], tz="UTC")
    return {
        "train": df[df.index <= train_end - gap],
        "val": df[(df.index > train_end) & (df.index <= val_end - gap)],
        "test": df[df.index > val_end],
    }


def main() -> None:
    p = load_params()
    series = build_series(p)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    series.to_parquet(PROCESSED / "series.parquet")
    parts = split(build_tabular(series, p), p)
    for name, part in parts.items():
        if part.empty:
            raise RuntimeError(f"{name} split is empty - check split dates in params.yaml")
        part.to_parquet(PROCESSED / f"{name}.parquet")
        log.info("%s: %d rows (%s -> %s)", name, len(part), part.index.min(), part.index.max())


if __name__ == "__main__":
    main()
