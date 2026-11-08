"""Stage ingest_weather: hourly weather history for Lahore from the Open-Meteo archive API (free, no key)."""
import pandas as pd

from src.ingest.http import get_json, session
from src.utils.common import RAW, get_logger, load_params

URL = "https://archive-api.open-meteo.com/v1/archive"
log = get_logger("ingest_weather")


def main() -> None:
    p = load_params()
    loc, d = p["location"], p["data"]
    js = get_json(session(), URL, {
        "latitude": loc["latitude"], "longitude": loc["longitude"],
        "start_date": d["start_date"], "end_date": d["end_date"],
        "hourly": ",".join(d["weather_vars"]), "timezone": "UTC",
    })
    hourly = js.get("hourly")
    if not hourly or not hourly.get("time"):
        raise RuntimeError(f"Open-Meteo returned no hourly data: {js}")
    df = pd.DataFrame(hourly).rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW / "weather.csv", index=False)
    log.info("Saved %d weather rows -> %s", len(df), RAW / "weather.csv")


if __name__ == "__main__":
    main()
