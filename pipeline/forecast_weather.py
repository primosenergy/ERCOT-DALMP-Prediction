# pipeline/forecast_weather.py
from __future__ import annotations
import requests
import pandas as pd
from datetime import datetime
from typing import Optional, Callable, Dict, Any
from urllib.parse import urlencode
import json
from pathlib import Path
from .loads import Loads
from .historic_weather import he_interval_from_local_end, CENTRAL  # reuse HE logic

BASE = "https://api.open-meteo.com/v1/forecast"
HOURLY_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "windspeed_10m",
    "cloudcover",  # won't store cloudcover yet, but keep for future
]

def _fetch_open_meteo(lat: float,
                      lon: float,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch hourly forecast from Open-Meteo.
    Times come back in the requested timezone (we request America/Chicago).
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_FIELDS),
        "timezone": "America/Chicago",
        "windspeed_unit": "ms",   # get m/s , don't have to convert
    }
    if start_date:
        params["start_date"] = start_date  # YYYY-MM-DD
    if end_date:
        params["end_date"] = end_date

    url = f"{BASE}?{urlencode(params)}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "hourly" not in data or not data["hourly"]:
        return pd.DataFrame(columns=["OperatingDTM","Interval","lat","lon","TempC","PrecipMM","WindMS","Humidity"])

    h = data["hourly"]
    df = pd.DataFrame(h)

    # Ensure all expected columns present
    for col in ["time", "temperature_2m", "precipitation", "windspeed_10m"]:
        if col not in df.columns:
            df[col] = pd.NA
    if "relative_humidity_2m" not in df.columns:
        df["relative_humidity_2m"] = pd.NA

    # Parse time as America/Chicago local, compute Hour-Ending (+1h)
    t = pd.to_datetime(df["time"], errors="coerce")
    # If tz-naive, localize to Central; else convert
    if getattr(t.dt, "tz", None) is None:
        t = t.dt.tz_localize(CENTRAL)
    else:
        t = t.dt.tz_convert(CENTRAL)

    local_end = t + pd.Timedelta(hours=1)

    out = pd.DataFrame({
        "OperatingDTM": local_end.dt.tz_localize(None),  # naive local clock
        "Interval": he_interval_from_local_end(local_end).to_numpy(),
        "lat": float(lat),
        "lon": float(lon),
        "TempC": pd.to_numeric(df["temperature_2m"], errors="coerce"),
        "PrecipMM": pd.to_numeric(df["precipitation"], errors="coerce"),
        "WindMS": pd.to_numeric(df["windspeed_10m"], errors="coerce"),
        "Humidity": pd.to_numeric(df["relative_humidity_2m"], errors="coerce"),
    })

    # Drop null timestamps; de-dup on key
    out = (
        out.dropna(subset=["OperatingDTM"])
           .drop_duplicates(subset=["OperatingDTM", "lat", "lon"])
    )
    return out

def load_forecast_hourly_to_duckdb(
    lat: float,
    lon: float,
    start_date: Optional[str] = None,  # "YYYY-MM-DD" (optional)
    end_date: Optional[str] = None,    # "YYYY-MM-DD" (optional)
    db_path: str = "ProjectMain/db/data.duckdb",
) -> int:
    """
    Fetch Open-Meteo hourly forecast → normalize to your schema →
    upsert into forecast_weather via Loads.load_forecast_weather().
    """
    df = _fetch_open_meteo(lat, lon, start_date, end_date)
    if df.empty:
        return 0

    loader = Loads(db_path)
    try:
        # Stamp forecast issuance/load time; Loads will use this if df lacks ForecastTime
        return loader.load_forecast_weather(df, source="open-meteo", forecast_time=datetime.now())
    finally:
        loader.close()

######################### -- Looping through Houston and Suburbs -- #######################

def load_forecast_hourly_batch_to_duckdb_from_file(
    mapping_file: str | Path,
    start_date: Optional[str],
    end_date: Optional[str],
    db_path: str = "ProjectMain/db/data.duckdb",
    log_fn: Optional[Callable[[str], None]] = None,
) -> int:
    
    if not Path(db_path).exists():
        raise FileNotFoundError(f"DB not found: {db_path}  (pass app.py’s DB_PATH)")
    """
    Read a {city: {lat,lon}} JSON and load Open-Meteo forecast for each point.
    Returns total rows upserted across all cities.
    """
    p = Path(mapping_file)
    if not p.exists():
        raise FileNotFoundError(f"Mapping file not found: {p}")

    coords_map: Dict[str, Dict[str, Any]] = json.loads(p.read_text(encoding="utf-8"))
    total = 0

    for city, geo in coords_map.items():
        lat = float(geo["lat"]); lon = float(geo["lon"])
        if log_fn:
            log_fn(f"▶ {city}: {lat:.4f}, {lon:.4f} …")

        try:
            added = load_forecast_hourly_to_duckdb(
                lat=lat, lon=lon,
                start_date=start_date, end_date=end_date,
                db_path=db_path,
            )
            total += (added or 0)
            if log_fn:
                log_fn(f"✓ {city}: upserted {added} rows")
        except Exception as e:
            if log_fn:
                log_fn(f"❌ {city} failed: {e}")

    return total