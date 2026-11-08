"""Stage validate: schema, null, range, duplicate, continuity and stuck-sensor checks on raw data.

Writes reports/validation.json and exits non-zero if any check fails, which stops `dvc repro`.
"""
import sys

import pandas as pd
import pandera.pandas as pa

from src.utils.common import RAW, REPORTS, get_logger, load_params, write_json

log = get_logger("validate")


def _schema(cols: dict[str, tuple], ranges: dict) -> pa.DataFrameSchema:
    columns = {"timestamp": pa.Column("datetime64[ns, UTC]", unique=True)}
    for name, (dtype, nullable) in cols.items():
        checks = [pa.Check.in_range(*ranges[name])] if name in ranges else []
        columns[name] = pa.Column(dtype, checks=checks, nullable=nullable)
    return pa.DataFrameSchema(columns, strict=False, coerce=True)


def _longest_constant_run(s: pd.Series) -> int:
    s = s.dropna()
    if s.empty:
        return 0
    groups = (s != s.shift()).cumsum()
    return int(s.groupby(groups).size().max())


def check(name: str, df: pd.DataFrame, schema: pa.DataFrameSchema, p: dict) -> dict:
    v = p["validation"]
    res = {"rows": len(df), "checks": {}}
    try:
        schema.validate(df, lazy=True)
        res["checks"]["schema_and_ranges"] = {"passed": True}
    except pa.errors.SchemaErrors as e:
        fc = e.failure_cases
        res["checks"]["schema_and_ranges"] = {
            "passed": False, "n_failures": len(fc),
            "examples": fc.head(10).astype(str).to_dict(orient="records")}

    null_frac = df.drop(columns="timestamp").isna().mean()
    bad = null_frac[null_frac > v["max_null_frac"]]
    # pm25_openaq is allowed to be sparse: it's the optional ground-sensor source
    bad = bad.drop(labels=["pm25_openaq"], errors="ignore")
    res["checks"]["null_fraction"] = {"passed": bad.empty, "per_column": null_frac.round(4).to_dict()}

    ts = pd.to_datetime(df["timestamp"], utc=True)
    expected = pd.date_range(p["data"]["start_date"], p["data"]["end_date"] + " 23:00", freq="h", tz="UTC")
    missing = expected.difference(ts)
    res["checks"]["hourly_continuity"] = {
        "passed": ts.is_monotonic_increasing and len(missing) / len(expected) <= v["max_null_frac"],
        "missing_hours": len(missing), "monotonic": bool(ts.is_monotonic_increasing)}

    if "pm25" in df:
        run = _longest_constant_run(df["pm25_openaq"]) if "pm25_openaq" in df else 0
        res["checks"]["stuck_sensor"] = {"passed": run <= v["max_stuck_hours"], "longest_constant_run_h": run}

    res["passed"] = all(c["passed"] for c in res["checks"].values())
    log.info("%s: %s", name, "PASS" if res["passed"] else "FAIL")
    return res


def main() -> None:
    p = load_params()
    ranges = p["validation"]["ranges"]
    aqi = pd.read_csv(RAW / "aqi.csv", parse_dates=["timestamp"])
    weather = pd.read_csv(RAW / "weather.csv", parse_dates=["timestamp"])

    aqi_schema = _schema({"pm25": (float, True), "pm25_openaq": (float, True),
                          "pm25_openmeteo": (float, True), "source": (str, False)},
                         {"pm25": ranges["pm25"], "pm25_openaq": ranges["pm25"],
                          "pm25_openmeteo": ranges["pm25"]})
    weather_schema = _schema({c: (float, True) for c in p["data"]["weather_vars"]}, ranges)

    report = {"aqi": check("aqi", aqi, aqi_schema, p),
              "weather": check("weather", weather, weather_schema, p)}
    report["passed"] = report["aqi"]["passed"] and report["weather"]["passed"]
    write_json(REPORTS / "validation.json", report)
    if not report["passed"]:
        log.error("Validation failed - see reports/validation.json")
        sys.exit(1)


if __name__ == "__main__":
    main()
