from __future__ import annotations
import duckdb
import pandas as pd
import numpy as np

def _synth_features() -> pd.DataFrame:
    """Small synthetic dataset so training works before real data arrives."""
    idx = pd.date_range('2025-08-01', periods=7*24, freq='H')
    rng = np.random.default_rng(0)
    temp = 30 + 8*np.sin(np.linspace(0, 6*np.pi, len(idx))) + rng.normal(0, 1, len(idx))
    load = 17000 + 1200*np.sin(np.linspace(0, 4*np.pi, len(idx))) + rng.normal(0, 150, len(idx))
    lmp = 25 + 0.4*(temp-30) + 0.002*(load-17000) + rng.normal(0, 2.5, len(idx))
    df = pd.DataFrame({'timestamp': idx, 'temp': temp, 'load': load, 'lmp': lmp})
    df['lag1'] = df['lmp'].shift(1)
    return df.dropna().reset_index(drop=True)

def build_feature_df(db_path: str) -> pd.DataFrame:
    """Join your real tables into a single feature frame; falls back to synthetic data."""
    try:
        con = duckdb.connect(db_path)
        df = con.execute("""
            WITH prices AS (SELECT timestamp, lmp FROM Prices),
                 ld AS (SELECT timestamp, load FROM LoadForecast),
                 wx AS (
                    SELECT timestamp, avg(temp) AS temp
                    FROM (
                        SELECT timestamp, temp FROM WeatherHist
                        UNION ALL
                        SELECT timestamp, temp FROM WeatherFcst
                    ) GROUP BY 1
                 )
            SELECT p.timestamp, p.lmp, ld.load, wx.temp
            FROM prices p
            LEFT JOIN ld ON ld.timestamp = p.timestamp
            LEFT JOIN wx ON wx.timestamp = p.timestamp
            ORDER BY p.timestamp
        """).fetch_df()
        if df.empty:
            return _synth_features()
        df['lag1'] = df['lmp'].shift(1)
        return df.dropna().reset_index(drop=True)
    except Exception:
        return _synth_features()
