"""Stage ingest_aqi: hourly PM2.5 for Lahore.

Primary source: OpenAQ v3 ground sensors (needs free OPENAQ_API_KEY), averaged across
all PM2.5 sensors within `openaq_radius_m` of the city centre.
Fallback: Open-Meteo Air Quality API (CAMS model, free, no key) fills hours OpenAQ misses.
Output columns: timestamp, pm25, pm25_openaq, pm25_openmeteo, source.
"""
import os
import time

import pandas as pd

from src.ingest.http import get_json, session
from src.utils.common import RAW, REPORTS, get_logger, load_params, write_json

OPENAQ = "https://api.openaq.org/v3"
OPENMETEO_AQ = "https://air-quality-api.open-meteo.com/v1/air-quality"
PM25_PARAMETER_ID = 2
log = get_logger("ingest_aqi")


def fetch_openmeteo(p: dict) -> pd.Series:
    loc, d = p["location"], p["data"]
    js = get_json(session(), OPENMETEO_AQ, {
        "latitude": loc["latitude"], "longitude": loc["longitude"],
        "start_date": d["start_date"], "end_date": d["end_date"],
        "hourly": "pm2_5", "timezone": "UTC",
    })
    h = js.get("hourly") or {}
    if not h.get("time"):
        raise RuntimeError(f"Open-Meteo AQ returned no data: {js}")
    s = pd.Series(h["pm2_5"], index=pd.to_datetime(h["time"], utc=True), name="pm25_openmeteo", dtype=float)
    log.info("Open-Meteo AQ: %d hours (%d non-null)", len(s), s.notna().sum())
    return s


def _openaq_sensor_ids(s, p: dict) -> list[int]:
    loc = p["location"]
    js = get_json(s, f"{OPENAQ}/locations", {
        "coordinates": f"{loc['latitude']},{loc['longitude']}",
        "radius": loc["openaq_radius_m"], "parameters_id": PM25_PARAMETER_ID, "limit": 100,
    })
    ids = [sen["id"] for l in js.get("results", []) for sen in l.get("sensors", [])
           if sen.get("parameter", {}).get("id") == PM25_PARAMETER_ID]
    log.info("OpenAQ: %d PM2.5 sensors near %s", len(ids), loc["city"])
    return ids


def _openaq_sensor_hours(s, sensor_id: int, start: str, end: str) -> pd.Series:
    rows, page = [], 1
    while True:
        js = get_json(s, f"{OPENAQ}/sensors/{sensor_id}/hours", {
            "datetime_from": f"{start}T00:00:00Z", "datetime_to": f"{end}T23:59:59Z",
            "limit": 1000, "page": page,
        })
        res = js.get("results", [])
        rows += [(r["period"]["datetimeFrom"]["utc"], r["value"]) for r in res]
        if len(res) < 1000:
            break
        page += 1
        time.sleep(1.1)  # stay under the 60 req/min free-tier limit
    if not rows:
        return pd.Series(dtype=float)
    ts, vals = zip(*rows)
    return pd.Series(vals, index=pd.to_datetime(ts, utc=True), dtype=float)


def fetch_openaq(p: dict) -> pd.Series | None:
    key = os.getenv("OPENAQ_API_KEY")
    if not key:
        log.warning("OPENAQ_API_KEY not set - using Open-Meteo AQ only")
        return None
    s = session({"X-API-Key": key})
    series = []
    for sid in _openaq_sensor_ids(s, p):
        sr = _openaq_sensor_hours(s, sid, p["data"]["start_date"], p["data"]["end_date"])
        sr = sr[(sr >= 0) & (sr < 2000)]  # drop sensor error codes before averaging
        log.info("  sensor %s: %d hours", sid, len(sr))
        if len(sr):
            series.append(sr.groupby(level=0).mean())
    if not series:
        return None
    return pd.concat(series, axis=1).mean(axis=1).rename("pm25_openaq")


def main() -> None:
    p = load_params()
    om = fetch_openmeteo(p)
    oaq = fetch_openaq(p)
    df = om.to_frame()
    df["pm25_openaq"] = oaq.reindex(df.index) if oaq is not None else float("nan")
    df["pm25"] = df["pm25_openaq"].fillna(df["pm25_openmeteo"])
    df["source"] = df["pm25_openaq"].notna().map({True: "openaq", False: "openmeteo"})
    df.index.name = "timestamp"
    RAW.mkdir(parents=True, exist_ok=True)
    df.reset_index()[["timestamp", "pm25", "pm25_openaq", "pm25_openmeteo", "source"]].to_csv(
        RAW / "aqi.csv", index=False)
    meta = {"rows": len(df), "openaq_coverage": float(df["pm25_openaq"].notna().mean()),
            "openmeteo_coverage": float(df["pm25_openmeteo"].notna().mean())}
    write_json(REPORTS / "aqi_meta.json", meta)
    log.info("Saved aqi.csv %s", meta)


if __name__ == "__main__":
    main()
