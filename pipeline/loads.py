# pipeline/loads.py
from __future__ import annotations
import io
import time
from typing import Dict, Tuple, List
import duckdb
import pandas as pd
import json
from datetime import datetime, date as _date

###### -- Historic Weather Helper -- ######
def _num_series(df: pd.DataFrame, col: str, length: int) -> pd.Series:
    """
    Return a numeric Series for df[col] (coerced), or an all-NA Series of given length,
    with a *numeric* nullable dtype to avoid object/NA inference issues.
    """
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype("Float64")
    return pd.Series([pd.NA] * length, index=df.index, dtype="Float64")


TABLE_NAME = "LoadForecast"  # change if you prefer another name

class Loads:
    """DuckDB loaders with Windows-friendly retry and a clean close()."""
    def __init__(self, db_path: str, retries: int = 25, wait_s: float = 0.25):
        self.db_path = db_path
        self.con = None

        last_err = None
        for _ in range(retries):
            try:
                self.con = duckdb.connect(db_path)  # READ_WRITE by default
                break
            except duckdb.IOException as e:
                last_err = e
                time.sleep(wait_s)
        if self.con is None:
            raise last_err

        self._init_tables()

    def close(self):
        if self.con is not None:
            try:
                self.con.close()
            except Exception:
                pass
            self.con = None

    def _init_tables(self):
        # Do NOT pre-create LoadForecast here; we create it with _ensure_loadforecast_table()
        self.con.execute("""
        CREATE TABLE IF NOT EXISTS SessionID (
            ts TIMESTAMP DEFAULT now(),
            session_id TEXT
        );
        """)
        self.con.execute("CREATE TABLE IF NOT EXISTS Prices (timestamp TIMESTAMP, lmp DOUBLE);")
        self.con.execute("""
          CREATE TABLE IF NOT EXISTS historical_weather (
            timestamp      TIMESTAMP,
            interval       INTEGER,      -- Hour-Ending 1..24/25
            lat            DOUBLE,
            lon            DOUBLE,
            temperature    DOUBLE,       -- °C
            humidity       DOUBLE,       -- %
            windspeed      DOUBLE,       -- m/s
            precipitation  DOUBLE,       -- mm
            source         TEXT,
            PRIMARY KEY (timestamp, lat, lon)
        );
        """)
        self.con.execute("""
         CREATE TABLE IF NOT EXISTS forecast_weather (
            forecast_time  TIMESTAMP,    -- when the forecast was produced/loaded
            timestamp      TIMESTAMP,    -- Central local Hour-Ending
            interval       INTEGER,
            lat            DOUBLE,
            lon            DOUBLE,
            temperature    DOUBLE,
            humidity       DOUBLE,
            windspeed      DOUBLE,
            precipitation  DOUBLE,
            source         TEXT,
            PRIMARY KEY (forecast_time, timestamp, lat, lon)
        );
        """)

    # -------------------- shared helpers --------------------
    def upsert_session(self, session_id: str):
        self.con.execute("INSERT INTO SessionID(session_id) VALUES (?);", [session_id])

    def _insert_df(self, table: str, df: pd.DataFrame):
        self.con.register("df", df)
        self.con.execute(f'INSERT INTO "{table}" SELECT * FROM df')
        self.con.unregister("df")

    # ---------------- PRICES & WEATHER (unchanged) ----------
    def load_prices(self, csv_bytes: bytes):
        df = pd.read_csv(io.BytesIO(csv_bytes))
        self._insert_df("Prices", df)

    # ------------------- WEATHER (meteostat hourly) --------------------
    def _ensure_weather_tables(self):
        self.con.execute("""
             CREATE TABLE IF NOT EXISTS historical_weather (
                timestamp      TIMESTAMP,
                interval       INTEGER,      -- Hour-Ending 1..24/25
                lat            DOUBLE,
                lon            DOUBLE,
                temperature    DOUBLE,       -- °C
                humidity       DOUBLE,       -- %
                windspeed      DOUBLE,       -- m/s
                precipitation  DOUBLE,       -- mm
                source         TEXT,
                PRIMARY KEY (timestamp, lat, lon)
            );
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS forecast_weather (
                forecast_time TIMESTAMP,                -- load/issuance time if provided; else now()
                timestamp TIMESTAMP,                    -- Central local Hour-Ending
                interval  INTEGER,                      -- 1..24/25
                lat DOUBLE,
                lon DOUBLE,
                temperature DOUBLE,
                humidity DOUBLE,
                windspeed DOUBLE,
                precipitation DOUBLE,
                source TEXT,                            -- e.g., 'open-meteo'
                PRIMARY KEY (forecast_time, timestamp, lat, lon)
            );
        """)

    # ---------- Public loaders ----------
    def load_historical_weather(self, df: pd.DataFrame, source: str = "meteostat"):
        """
        Expects columns:
          OperatingDTM, Interval, lat, lon, TempC, PrecipMM, WindMS
        Optional: Humidity (0-100)
        """
        if df is None or df.empty:
            return 0
        self._ensure_weather_tables()

        # ---- in load_historical_weather ----
        n = len(df)
        shaped = pd.DataFrame({
            "timestamp":     pd.to_datetime(df["OperatingDTM"], errors="coerce"),
            "interval":      pd.to_numeric(df["Interval"], errors="coerce").astype("Int64"),
            "lat":           pd.to_numeric(df["lat"], errors="coerce").astype("Float64"),
            "lon":           pd.to_numeric(df["lon"], errors="coerce").astype("Float64"),
            "temperature":   _num_series(df, "TempC",   n),   # Float64
            "humidity":      _num_series(df, "Humidity", n),  # Float64
            "windspeed":     _num_series(df, "WindMS",  n),   # Float64
            "precipitation": _num_series(df, "PrecipMM", n),  # Float64
            "source":        "meteostat",
        })

# (keep your dropna / drop_duplicates)



        shaped = shaped.dropna(subset=["timestamp", "lat", "lon"])
        shaped = shaped.drop_duplicates(subset=["timestamp", "lat", "lon"])

        self.con.register("s", shaped)
        # Upsert via delete+insert
        self.con.execute("""
            DELETE FROM historical_weather t USING s
            WHERE t.timestamp = s.timestamp AND t.lat = s.lat AND t.lon = s.lon;
        """)
        self.con.execute("""
            INSERT INTO historical_weather
            (timestamp, interval, lat, lon, temperature, humidity, windspeed, precipitation, source)
            SELECT timestamp, interval, lat, lon, temperature, humidity, windspeed, precipitation, source
            FROM s;
        """)
        self.con.unregister("s")
        return len(shaped)

    def load_forecast_weather(self, df: pd.DataFrame, source: str = "open-meteo", forecast_time: datetime | None = None):
        """
        Expects columns:
          OperatingDTM, Interval, lat, lon, TempC, PrecipMM, WindMS
        Optional: Humidity
        If df doesn't include a 'ForecastTime' column, we stamp the load time.
        """
        if df is None or df.empty:
            return 0
        self._ensure_weather_tables()  # <-- add this

        if "ForecastTime" in df.columns:
            f_time = pd.to_datetime(df["ForecastTime"])
        else:
            f_time = pd.Series([forecast_time or datetime.now()] * len(df))

        shaped = pd.DataFrame({
            "forecast_time": f_time,
            "timestamp": pd.to_datetime(df["OperatingDTM"]),
            "interval": pd.to_numeric(df["Interval"]),
            "lat": pd.to_numeric(df["lat"]),
            "lon": pd.to_numeric(df["lon"]),
            "temperature": pd.to_numeric(df.get("TempC")),
            "humidity": pd.to_numeric(df.get("Humidity") if "Humidity" in df.columns else None),
            "windspeed": pd.to_numeric(df.get("WindMS")),
            "precipitation": pd.to_numeric(df.get("PrecipMM")),
            "source": source,
        })

        shaped = shaped.dropna(subset=["forecast_time", "timestamp", "lat", "lon"])
        shaped = shaped.drop_duplicates(subset=["forecast_time", "timestamp", "lat", "lon"])

        self.con.register("s", shaped)
        self.con.execute("""
            DELETE FROM forecast_weather t USING s
            WHERE t.forecast_time = s.forecast_time
              AND t.timestamp = s.timestamp
              AND t.lat = s.lat
              AND t.lon = s.lon;
        """)
        self.con.execute("""
            INSERT INTO forecast_weather
            (forecast_time, timestamp, interval, lat, lon, temperature, humidity, windspeed, precipitation, source)
            SELECT forecast_time, timestamp, interval, lat, lon, temperature, humidity, windspeed, precipitation, source
            FROM s;
        """)
        self.con.unregister("s")
        return len(shaped)

    # ------------------- FORECAST (flat JSON) --------------------
    def _ensure_loadforecast_table(self):
        self.con.execute(f"""
            CREATE TABLE IF NOT EXISTS "{TABLE_NAME}" (
                OperatingDTM TIMESTAMP,
                Interval     INTEGER,
                Month        TEXT,
                LocationType TEXT,
                Location     TEXT,
                DSTFlag      TEXT,
                ForecastMW   DOUBLE
            );
        """)
        # Create an index instead of a strict PK
        self.con.execute(f'CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_key ON "{TABLE_NAME}"(OperatingDTM, Interval, LocationType, Location);')

    def _upsert_loadforecast(self, df: pd.DataFrame):
        """Upsert rows by deleting conflicts on the composite PK, then inserting."""
        self.con.register("staging_view", df)
        self.con.execute(f"""
            DELETE FROM "{TABLE_NAME}" AS t
            USING staging_view s
            WHERE
                t.OperatingDTM = s.OperatingDTM AND
                t.Interval     = s.Interval     AND
                t.LocationType = s.LocationType AND
                t.Location     = s.Location;
        """)
        self.con.execute(f"""
            INSERT INTO "{TABLE_NAME}" (
                OperatingDTM, Interval, Month, LocationType, Location, DSTFlag, ForecastMW
            )
            SELECT OperatingDTM, Interval, Month, LocationType, Location, DSTFlag, ForecastMW
            FROM staging_view;
        """)
        self.con.unregister("staging_view")

    def load_load_fcst_json(self, json_bytes: bytes):
        """Accept flat JSON rows and upsert into LoadForecast."""
        import json

        # 0) Parse
        try:
            obj = json.loads(json_bytes.decode("utf-8"))
        except Exception:
            obj = json.loads(json_bytes)

        # 1) Server error payloads
        if isinstance(obj, dict) and any(k in obj for k in ("Message","Exception","StackTrace","InnerException")):
            msg = obj.get("Message") or obj.get("Exception") or "Server returned error"
            raise RuntimeError(f"Pivot API error: {msg}")

        # 2) Rows (flat list or nested)
        if isinstance(obj, dict) and isinstance(obj.get("data"), dict) and isinstance(obj["data"].get("data"), list):
            rows = obj["data"]["data"]
        elif isinstance(obj, list):
            rows = obj
        else:
            raise ValueError("Unexpected JSON shape: expected list or obj['data']['data']")

        if not rows:
            print("DEBUG: API returned 0 rows; nothing to insert.")
            return

        df = pd.json_normalize(rows)

        # 3) Expected flat columns
        required = ["OperatingDTM","Interval","Month","LocationType","Location","DSTFlag","ForecastMW"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print("DEBUG available columns:", list(df.columns))
            raise ValueError(f"Missing expected columns {missing}. Upstream response doesn't match flat schema.")

        # 4) Select, coerce types
        print(f"DEBUG: incoming rows count just before conversions: {len(df)}; columns: {list(df.columns)}")
        df = df[required].copy()
        # timestamps
        s = pd.to_datetime(df["OperatingDTM"], errors="coerce", utc=True)
        df["OperatingDTM"] = s.dt.tz_convert(None)  # make tz-naive after parsing
        # Fallback for rare tz-naive inputs (won't happen with utc=True, but safe):
        if df["OperatingDTM"].dtype == "datetime64[ns]":  # still naive
            df["OperatingDTM"] = pd.to_datetime(df["OperatingDTM"], errors="coerce")
        # numerics
        df["Interval"]   = pd.to_numeric(df["Interval"], errors="coerce").astype("Int64")
        month_as_str = df["Month"].astype("string")
        # If you want just the month number as text, uncomment next line:
        # month_as_str = month_as_str.str.split("-").str[0]
        df["Month"] = month_as_str  # keep as TEXT to match table
        df["ForecastMW"] = pd.to_numeric(df["ForecastMW"], errors="coerce")
        # strings: sometimes come back as None
        for col in ["LocationType","Location","DSTFlag"]:
            df[col] = df[col].astype("string")

        # 5) Drop rows with NULLs in PK cols or missing ForecastMW
        pk_cols = ["OperatingDTM","Interval","Month","LocationType","Location","DSTFlag"]
        pre = len(df)
        df = df.dropna(subset=pk_cols + ["ForecastMW"])
        post = len(df)
        print(f"DEBUG: rows before PK/drop: {pre}, after dropna on PK+ForecastMW: {post}")

        if df.empty:
            print("DEBUG: all rows dropped due to NULLs in PK or ForecastMW; nothing to insert.")
            return

        # 6) Ensure table exists, then upsert
        self._ensure_loadforecast_table()
        print(f"DEBUG: upserting {len(df)} rows into {TABLE_NAME} …")
        self._upsert_loadforecast(df)
        print(f"DEBUG: upsert complete.")
        
# --------- PRICES (pivot JSON, Day-Ahead hourly) ----------
    def _ensure_prices_table(self):
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS prices_hourly (
                OperatingDTM TIMESTAMP,
                TIME         TEXT,
                Location     TEXT,
                Price        DOUBLE
            );
        """)
        self.con.execute("""
            CREATE INDEX IF NOT EXISTS idx_prices_hourly_key
            ON prices_hourly(OperatingDTM, Time, Location);
        """)

    def _upsert_prices(self, df: pd.DataFrame):
        self.con.register("s", df)
        self.con.execute("""
            DELETE FROM prices_hourly t USING s
            WHERE t.OperatingDTM = s.OperatingDTM
              AND t.Time     = s.Time
              AND t.Location     = s.Location
             ;
        """)
        self.con.execute("""
            INSERT INTO prices_hourly (OperatingDTM, Time, Location, Price)
            SELECT OperatingDTM, Time, Location, Price
            FROM s;
        """)
        self.con.unregister("s")

    def load_prices_pivot_json(self, json_bytes: bytes):
        """Accept pivot JSON rows for rtp_PricesHourlyERCOTView and upsert into prices_hourly."""
        # 0) Parse
        try:
            obj = json.loads(json_bytes.decode("utf-8"))
        except Exception:
            obj = json.loads(json_bytes)

        # 1) Server error payloads
        if isinstance(obj, dict) and any(k in obj for k in ("Message","Exception","StackTrace","InnerException")):
            msg = obj.get("Message") or obj.get("Exception") or "Server returned error"
            raise RuntimeError(f"Pivot API error: {msg}")

        # 2) Rows
        if isinstance(obj, dict) and isinstance(obj.get("data"), dict) and isinstance(obj["data"].get("data"), list):
            rows = obj["data"]["data"]
        elif isinstance(obj, list):
            rows = obj
        else:
            raise ValueError("Unexpected JSON shape for prices: expected list or obj['data']['data']")

        if not rows:
            print("DEBUG: prices API returned 0 rows; nothing to insert.")
            return

        df = pd.json_normalize(rows)

        # Common variants for the aggregated column
        rename_map = {"Avg(Price)": "Price", "avg(Price)": "Price"}
        df = df.rename(columns=rename_map)

        required = ["OperatingDTM","Time","Location","Price"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print("DEBUG available columns:", list(df.columns))
            raise ValueError(f"Missing expected columns {missing} for prices.")

        df = df[required].copy()

        # Types
        s = pd.to_datetime(df["OperatingDTM"], errors="coerce", utc=True)
        df["OperatingDTM"] = s.dt.tz_convert(None)
        if df["OperatingDTM"].dtype == "datetime64[ns]":
            df["OperatingDTM"] = pd.to_datetime(df["OperatingDTM"], errors="coerce")

        df["Time"] = pd.to_numeric(df["Time"], errors="coerce").astype("Int64")
        df["Price"]    = pd.to_numeric(df["Price"], errors="coerce")
        df["Location"] = df["Location"].astype("string").str.lower()
        # SASMID can be null/empty
        if "SASMID" in df.columns:
            df["SASMID"] = df["SASMID"].astype("string")

        # Drop nulls in key or measure
        df = df.dropna(subset=["OperatingDTM","Time","Location","Price"])

        if df.empty:
            print("DEBUG: all price rows dropped due to NULL keys; nothing to insert.")
            return

        self._ensure_prices_table()
        self._upsert_prices(df)