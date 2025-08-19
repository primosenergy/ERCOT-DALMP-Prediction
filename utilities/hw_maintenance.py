# utilities/hw_maintenance.py
from __future__ import annotations

import duckdb
import pandas as pd
from contextlib import contextmanager
from typing import Dict, Any

@contextmanager
def connect_duckdb(db_path: str):
    con = duckdb.connect(db_path)
    try:
        yield con
    finally:
        con.close()

def _detect_ts_col(con: duckdb.DuckDBPyConnection) -> str:
    # figure out whether your table uses 'timestamp' or 'OperatingDTM'
    cols = {r[0].lower() for r in con.execute("DESCRIBE historical_weather").fetchall()}
    if "timestamp" in cols:
        return "timestamp"
    if "operatingdtm" in cols:
        return "OperatingDTM"  # keep original case for readability
    raise RuntimeError("historical_weather is missing a timestamp column (expected 'timestamp' or 'OperatingDTM').")

def _missing_sql(ts_col: str) -> str:
    return f"""
WITH bounds AS (
  SELECT
    lat, lon,
    date_trunc('hour', MIN({ts_col})) AS start_ts,
    date_trunc('hour', MAX({ts_col})) AS end_ts
  FROM historical_weather
  GROUP BY lat, lon
),
expected AS (
  SELECT b.lat, b.lon, gs.ts
  FROM bounds b
  CROSS JOIN LATERAL generate_series(b.start_ts, b.end_ts, INTERVAL 1 HOUR) AS gs(ts)
),
normalized_hw AS (
  SELECT DISTINCT lat, lon, date_trunc('hour', {ts_col}) AS ts_hour
  FROM historical_weather
)
SELECT e.lat, e.lon, e.ts AS missing_timestamp
FROM expected e
LEFT JOIN normalized_hw n
  ON e.lat = n.lat AND e.lon = n.lon AND e.ts = n.ts_hour
WHERE n.ts_hour IS NULL
ORDER BY e.lat, e.lon, e.ts;
"""

def _interp_select_sql(ts_col: str) -> str:
    # produces a SELECT of interpolated rows; use in INSERT or MERGE
    return f"""
WITH bounds AS (
  SELECT
    lat, lon,
    date_trunc('hour', MIN({ts_col})) AS start_ts,
    date_trunc('hour', MAX({ts_col})) AS end_ts
  FROM historical_weather
  GROUP BY lat, lon
),
expected AS (
  SELECT b.lat, b.lon, gs.ts
  FROM bounds b
  CROSS JOIN LATERAL generate_series(b.start_ts, b.end_ts, INTERVAL 1 HOUR) AS gs(ts)
),
normalized_hw AS (
  SELECT DISTINCT lat, lon, date_trunc('hour', {ts_col}) AS ts_hour
  FROM historical_weather
),
missing_only AS (
  SELECT e.lat, e.lon, e.ts AS missing_ts
  FROM expected e
  LEFT JOIN normalized_hw n
    ON e.lat = n.lat AND e.lon = n.lon AND e.ts = n.ts_hour
  WHERE n.ts_hour IS NULL
),
nearest_before AS (
  SELECT
    mo.lat, mo.lon, mo.missing_ts,
    MAX(h.{ts_col}) AS prev_ts
  FROM missing_only mo
  JOIN historical_weather h
    ON h.lat = mo.lat
   AND h.lon = mo.lon
   AND h.{ts_col} < mo.missing_ts
  GROUP BY mo.lat, mo.lon, mo.missing_ts
),
nearest_after AS (
  SELECT
    mo.lat, mo.lon, mo.missing_ts,
    MIN(h.{ts_col}) AS next_ts
  FROM missing_only mo
  JOIN historical_weather h
    ON h.lat = mo.lat
   AND h.lon = mo.lon
   AND h.{ts_col} > mo.missing_ts
  GROUP BY mo.lat, mo.lon, mo.missing_ts
),
interp AS (
  SELECT
    b.lat, b.lon, b.missing_ts,
    'meteostat_interp' AS source,
    CAST(DATEDIFF('second', b.prev_ts, b.missing_ts) AS DOUBLE)
      / NULLIF(CAST(DATEDIFF('second', b.prev_ts, a.next_ts) AS DOUBLE), 0) AS ratio,
    hb.temperature   AS temp_prev,  ha.temperature   AS temp_next,
    hb.humidity      AS hum_prev,   ha.humidity      AS hum_next,
    hb.windspeed     AS wind_prev,  ha.windspeed     AS wind_next,
    hb.precipitation AS prcp_prev,  ha.precipitation AS prcp_next
  FROM nearest_before b
  JOIN nearest_after  a
    ON b.lat = a.lat AND b.lon = a.lon AND b.missing_ts = a.missing_ts
  JOIN historical_weather hb
    ON hb.lat = b.lat AND hb.lon = b.lon AND hb.{ts_col} = b.prev_ts
  JOIN historical_weather ha
    ON ha.lat = a.lat AND ha.lon = a.lon AND ha.{ts_col} = a.next_ts
)
SELECT
  i.missing_ts AS {ts_col},
  CASE WHEN EXTRACT(hour FROM i.missing_ts) = 0 THEN 24 ELSE EXTRACT(hour FROM i.missing_ts) END AS interval,
  i.lat,
  i.lon,
  COALESCE(i.temp_prev + (i.temp_next - i.temp_prev) * i.ratio, i.temp_prev) AS temperature,
  COALESCE(i.hum_prev  + (i.hum_next  - i.hum_prev)  * i.ratio, i.hum_prev)  AS humidity,
  COALESCE(i.wind_prev + (i.wind_next - i.wind_prev) * i.ratio, i.wind_prev) AS windspeed,
  COALESCE(i.prcp_prev + (i.prcp_next - i.prcp_prev) * i.ratio, i.prcp_prev) AS precipitation,
  'meteostat_interp' AS source
FROM interp i
WHERE i.ratio IS NOT NULL
"""

def report_missing(db_path: str, preview_rows: int = 25) -> Dict[str, Any]:
    with connect_duckdb(db_path) as con:
        ts_col = _detect_ts_col(con)
        sql = _missing_sql(ts_col)
        df = con.execute(sql).fetchdf()
        return {
            "timestamp_col": ts_col,
            "missing_count": int(len(df)),
            "preview": df.head(preview_rows),
        }

def backfill_missing(db_path: str, use_merge: bool = True) -> Dict[str, Any]:
    with connect_duckdb(db_path) as con:
        ts_col = _detect_ts_col(con)
        interp_select = _interp_select_sql(ts_col)

        # Count missing before
        missing_before = con.execute(_missing_sql(ts_col)).fetchdf()
        missing_n_before = int(len(missing_before))

        if missing_n_before == 0:
            return {
                "timestamp_col": ts_col,
                "inserted": 0,
                "missing_before": 0,
                "missing_after": 0,
                "note": "No gaps to fill."
            }

        if use_merge:
            # Idempotent upsert
            merge_sql = f"""
            MERGE INTO historical_weather AS t
            USING ({interp_select}) AS s
            ON t.lat = s.lat AND t.lon = s.lon AND t.{ts_col} = s.{ts_col}
            WHEN NOT MATCHED THEN
              INSERT ({ts_col}, interval, lat, lon, temperature, humidity, windspeed, precipitation, source)
              VALUES (s.{ts_col}, s.interval, s.lat, s.lon, s.temperature, s.humidity, s.windspeed, s.precipitation, s.source);
            """
            res = con.execute(merge_sql)
            # DuckDB's MERGE doesn't always return a rowcount; recompute after
        else:
            insert_sql = f"""
            INSERT INTO historical_weather ({ts_col}, interval, lat, lon, temperature, humidity, windspeed, precipitation, source)
            {interp_select}
            """
            res = con.execute(insert_sql)

        # Count missing after
        missing_after = con.execute(_missing_sql(ts_col)).fetchdf()
        missing_n_after = int(len(missing_after))

        # Estimate rows inserted
        inserted_estimate = max(missing_n_before - missing_n_after, 0)

        return {
            "timestamp_col": ts_col,
            "inserted": inserted_estimate,
            "missing_before": missing_n_before,
            "missing_after": missing_n_after,
            "tag": "meteostat_interp"
        }

def run(db_path: str = "ProjectMain/db/data.duckdb",
        preview_rows: int = 25,
        use_merge: bool = True,
        log=None):
    """
    Convenience wrapper: report gaps, then backfill once. Returns a summary dict.
    """
    if log: log(f"[hw] reporting gaps… db={db_path}")
    rep = report_missing(db_path, preview_rows=preview_rows)

    if log: log(f"[hw] backfilling… use_merge={use_merge}")
    back = backfill_missing(db_path, use_merge=use_merge)

    if log:
        log(f"[hw] missing_before={back.get('missing_before')} "
            f"→ missing_after={back.get('missing_after')} | inserted={back.get('inserted')}")

    return {"report": rep, "backfill": back}


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description="Historical weather maintenance (report + backfill)")
    p.add_argument("--db", dest="db", default="ProjectMain/db/data.duckdb",
                   help="Path to DuckDB file")
    p.add_argument("--preview", dest="preview", type=int, default=25,
                   help="Rows to preview in report_missing()")
    p.add_argument("--no-merge", dest="no_merge", action="store_true",
                   help="Disable merge/upsert optimization during backfill")
    args = p.parse_args()

    res = run(
        db_path=args.db,
        preview_rows=args.preview,
        use_merge=not args.no_merge,
        log=print
    )
    print(json.dumps(res, default=str, indent=2))