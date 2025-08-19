# pipeline/historic_weather.py
from __future__ import annotations
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from meteostat import Stations, Hourly

from .loads import Loads  # uses  upsert methods

CENTRAL = ZoneInfo("America/Chicago")

def he_interval_from_local_end(local_end: pd.Series) -> pd.Series:
    """
    Map local period-end timestamps to Hour-Ending (1..24, or 25 on fall DST day).
    - HE is the local clock HOUR with 0 mapped to 24
    - The duplicated hour during fall-back gets +1 (can see 1 then 2 for the two 01:00's)
    """
    s = pd.Series(local_end)

    # Base HE from clock hour: 0 -> 24, else hour
    hour = s.dt.hour
    base_he = hour.where(hour != 0, 24)

    # Handle the duplicated hour on fall-back: add occurrence count (0,1,...) within (date,hour)
    date = s.dt.date
    # Make a 1s series and cumsum within (date,hour) groups to get 0,1 for duplicates
    occurrence = (
        pd.Series(1, index=s.index)
        .groupby([date, hour], sort=False)
        .cumsum()
        .sub(1)  # 0 for first occurrence, 1 for second
    )

    return (base_he + occurrence).astype("int64")

def fetch_historic_hourly(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch Meteostat hourly weather near (lat, lon), normalize to Central HE with Interval."""
    station = Stations().nearby(lat, lon).fetch(1).index[0]
    raw = Hourly(station, start, end).fetch()

    # Normalize to a single time series 'idx_utc' which is tz-aware UTC
    if raw.empty:
        return pd.DataFrame(columns=["OperatingDTM","Interval","lat","lon","TempC","PrecipMM","WindMS","Humidity"])

    # Case A: time is in the index (common)
    if isinstance(raw.index, pd.DatetimeIndex):
        idx = raw.index
        if getattr(idx, "tz", None) is None:
            idx_utc = pd.DatetimeIndex(idx).tz_localize("UTC")
        else:
            idx_utc = idx.tz_convert("UTC")
        df = raw.copy()

    # Case B: someone did reset_index(), so time is a column
    elif "time" in raw.columns:
        tcol = pd.to_datetime(raw["time"], errors="coerce")
        # If tz-naive, the Meteostat convention is UTC → localize as UTC
        if getattr(tcol.dt, "tz", None) is None:
            tcol = tcol.dt.tz_localize("UTC")
        idx_utc = tcol.dt.tz_convert("UTC")
        df = raw.copy()

    else:
        # Fallback: coerce whatever to UTC
        tcol = pd.to_datetime(raw.index, errors="coerce")
        idx_utc = pd.DatetimeIndex(tcol).tz_localize("UTC")
        df = raw.copy()

    # Convert to Central local time; Hour-Ending is +1 hour from local-begin
    local_begin = idx_utc.tz_convert(CENTRAL)
    local_end = local_begin + pd.Timedelta(hours=1)


    # Build output
    out = df.copy()
    out["OperatingDTM"] = local_end.tz_localize(None)  # local clock, naive
    out["Interval"] = he_interval_from_local_end(local_end).to_numpy()
    out["lat"] = float(lat)
    out["lon"] = float(lon)

    check = (
    out.assign(date=out["OperatingDTM"].dt.date, hour=out["OperatingDTM"].dt.hour)
       .loc[lambda d: d["date"] == out["OperatingDTM"].dt.date.iloc[0],
            ["OperatingDTM","hour","Interval"]]
       .sort_values("OperatingDTM")
    )
    print(check.head(8))

    # Map Meteostat fields → schema
    # temp (°C), prcp (mm), rhum (%), wspd (km/h) → convert to m/s
    out["TempC"] = out["temp"] if "temp" in out.columns else pd.NA
    out["PrecipMM"] = out["prcp"] if "prcp" in out.columns else pd.NA
    out["Humidity"] = out["rhum"] if "rhum" in out.columns else pd.NA
    out["WindMS"] = (out["wspd"] / 3.6) if "wspd" in out.columns else pd.NA

    return out[["OperatingDTM","Interval","lat","lon","TempC","PrecipMM","WindMS","Humidity"]] \
        .dropna(subset=["OperatingDTM"]) \
        .drop_duplicates(subset=["OperatingDTM","lat","lon"])

def load_historic_hourly_to_duckdb(
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
    db_path: str = "ProjectMain/db/data.duckdb",
) -> int:
    """Fetch + upsert into historical_weather; returns number of rows upserted."""
    df = fetch_historic_hourly(lat, lon, start, end)
    if df.empty:
        return 0
    loader = Loads(db_path)
    try:
        # IMPORTANT: use the historical-specific loader (separate table)
        return loader.load_historical_weather(df)
    finally:
        loader.close()
