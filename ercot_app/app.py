# app.py
import sys, os
from pathlib import Path
from datetime import date as _date, timedelta as _timedelta, datetime
import secrets, asyncio
import pandas as pd
import duckdb
from nicegui import ui, app as nice_app
from dotenv import load_dotenv, find_dotenv
import json
import re


# --- helper for default Forecast Weather dates --- #
try:
    from zoneinfo import ZoneInfo
    _CENTRAL = ZoneInfo("America/Chicago")
    _today = datetime.now(_CENTRAL).date()
except Exception:
    # Fallback if zoneinfo isn't available for some reason
    _today = datetime.today().date()

_start_default = (_today + _timedelta(days=1)).isoformat()      # tomorrow
_end_default   = (_today + _timedelta(days=7)).isoformat()      # tomorrow + 7 days

# --- repo/python path setup ---
HERE = Path(__file__).resolve().parent            # folder containing app.py
ROOT = HERE.parent                                # parent folder (repo root)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.loads import Loads
from storage.paths import get_db_path

# --- helper for Deep Learning Inserts --- #
MODEL_SCRIPT = ROOT / 'modeling' / 'final_torch2.py'  # <- path to the final script for real
DB_PATH = ROOT / 'ProjectMain' / 'db' / 'data.duckdb'  # reuse existing global


# --- env / secrets ---
load_dotenv(find_dotenv(filename='.env', usecwd=True))
STORAGE_SECRET = os.getenv('NICEGUI_STORAGE_SECRET') or secrets.token_urlsafe(32)

# --- external helpers ---
from pipeline.session import get_and_save_session
from pipeline.historic_weather import load_historic_hourly_to_duckdb  # dropped after initial run
from utilities.hw_maintenance import report_missing, backfill_missing # dropped after initial run
from modeling.final_torch2 import run_deep_learning_forecast


# --- DB path resolution (robust across run locations) ---
def resolve_db_path() -> str:
    """
    Return an absolute path to ProjectMain/db/data.duckdb.
    Tries common layouts relative to this file; fails loudly if not found.
    """
    candidates = [
        HERE / 'ProjectMain' / 'db' / 'data.duckdb',      # app.py in repo root
        ROOT / 'ProjectMain' / 'db' / 'data.duckdb',      # app.py in subfolder; db in repo root/ProjectMain
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # If none exist, fail with a clear message (avoid creating a new DB in the wrong place)
    raise FileNotFoundError(
        f"Could not find data.duckdb. Tried:\n  " +
        "\n  ".join(str(p) for p in candidates) +
        "\nFix the layout or update resolve_db_path()."
    )

DB_PATH = get_db_path() 


# --- utilities ---
def _redact(s: str) -> str:
    for k in ('ERCOT_UID', 'ERCOT_PWD'):
        v = os.getenv(k)
        if v:
            s = s.replace(v, '****')
    return s

def _df_to_jsonable(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    for col in out.select_dtypes(include=['Int64']).columns:
        out[col] = out[col].astype('object').where(out[col].notna(), None)
    for col in out.select_dtypes(include=['boolean']).columns:
        out[col] = out[col].astype(object).where(out[col].notna(), None)
    out = out.where(out.notna(), None)
    return out.to_dict(orient='records')

def preview_latest():
    def _resolve_table(con, wanted: str) -> str | None:
        # Return the exact stored table_name (any case), else None
        row = con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE lower(table_name)=lower(?) LIMIT 1",
            [wanted]
        ).fetchone()
        return row[0] if row else None

    try:
        con = duckdb.connect(str(DB_PATH))

        lf = _resolve_table(con, 'LoadForecast')        # Load forecast table (quoted vs unquoted safe)
        ph = _resolve_table(con, 'prices_hourly')       # Prices
        pr = _resolve_table(con, 'Prices')              # Prices (CSV), created in _init_tables()

        title, df = 'Preview', pd.DataFrame()

        if lf:
            df = con.execute(f"""
                SELECT OperatingDTM, Interval, Month, LocationType, Location, DSTFlag, ForecastMW
                FROM "{lf}"
                ORDER BY OperatingDTM DESC, Interval ASC, Location ASC
                LIMIT 200
            """).fetchdf()
            title = f'{lf} preview (latest 200)'

        elif ph:
            df = con.execute(f"""
                SELECT OperatingDTM, Time, Location, SASMID, Price
                FROM "{ph}"
                ORDER BY OperatingDTM DESC, Location ASC
                LIMIT 200
            """).fetchdf()
            title = f'{ph} preview (latest 200)'

        elif pr:
            df = con.execute(f"""
                SELECT *
                FROM "{pr}"
                ORDER BY 1 DESC
                LIMIT 200
            """).fetchdf()
            title = f'{pr} preview (latest 200)'

        else:
            existing = [r[0] for r in con.execute("SELECT table_name FROM duckdb_tables() ORDER BY 1").fetchall()]
            ui.notify(
                'No "LoadForecast", "prices_hourly", or "Prices" table found. '
                f'Found tables: {existing}',
                type='warning'
            )

        con.close()

        with ui.dialog() as d, ui.card():
            ui.label(title).classes('text-lg font-medium')
            if not df.empty:
                rows = _df_to_jsonable(df)
                cols = [{'name': c, 'label': c, 'field': c} for c in df.columns]
                ui.table(columns=cols, rows=rows).classes('w-[1000px]')
            else:
                ui.label('No rows')
            ui.button('Close', on_click=d.close)
        d.open()

    except Exception as e:
        ui.notify(f'Query failed: {e}', type='negative')

def run_query(sql: str, params: tuple | list = ()):
        con = duckdb.connect(str(DB_PATH))
        try:
            df = con.execute(sql, params).fetchdf()
        finally:
            con.close()
        return df

def make_line_option(df: pd.DataFrame, x_col: str, y_cols: list[str], title: str = "") -> dict:
    """Convert a DataFrame into a simple multi-series ECharts line option."""
    x_vals = df[x_col].astype(str).tolist()
    series = [
        {"type": "line", "name": y, "showSymbol": False, "data": df[y].tolist()}
        for y in y_cols if y in df.columns
    ]
    return {
        "title": {"text": title},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": y_cols},
        "xAxis": {"type": "category", "data": x_vals},
        "yAxis": {"type": "value", "scale": True},
        "grid": {"left": 60, "right": 20, "top": 50, "bottom": 40},
        "series": series,
    }

@ui.page('/')
def main():
    ui.label('ERCOT Day-Ahead Price Prediction Console').classes('text-2xl font-bold w-full max-w-3xl mx-auto')

    # --- logging (main) ---
    log = ui.log(max_lines=400).classes('w-full max-w-3xl mx-auto')
    sid_label = ui.label('SID: (none yet)').classes('text-sm text-gray-600 w-full max-w-3xl mx-auto')

    # Date inputs
    from_input = ui.input('From Operating Date (MM/DD/YYYY)', value=_date.today().strftime('%m/%d/%Y')).classes('w-full max-w-3xl mx-auto')
    to_input   = ui.input('To Operating Date (MM/DD/YYYY)',   value=(_date.today() + _timedelta(days=6)).strftime('%m/%d/%Y')).classes('w-full max-w-3xl mx-auto')

    async def click_session():
        log.push(f"App DB_PATH -> {DB_PATH.resolve()}")
        log.push('▶ Session: requesting...')
        try:
            sid = await asyncio.to_thread(get_and_save_session)
            sid_label.text = f'SID: {sid}'
            log.push('✅ Session: SID saved to ProjectMain/db/session_id.txt')
        except Exception as e:
            log.push(f'❌ Session failed: {e}')

    async def click_load_forecast():
        log.push('▶ Load Forecast: posting...')
        try:
            from pipeline.loadforecast import run_load_forecast, LOG_FILE
            await asyncio.to_thread(run_load_forecast, from_input.value, to_input.value, str(DB_PATH))

            tail = ''
            if LOG_FILE.exists():
                lines = LOG_FILE.read_text(encoding='utf-8').strip().splitlines()
                tail = '\n'.join(lines[-2:])
            if tail:
                log.push('⎯⎯ Sent request ⎯⎯')
                for line in tail.splitlines():
                    log.push(_redact(line))

            log.push('✅ Load Forecast: inserted into DuckDB table LoadForecast')
        except Exception as e:
            log.push(f'❌ Load Forecast failed: {e}')


    async def click_load_prices():
        log.push('▶ Prices: posting...')
        try:
            from pipeline.prices import run_prices, LOG_FILE
            # Change defaults here for different location/owner
            await asyncio.to_thread(run_prices, from_input.value, to_input.value, "hb_", "q", str(DB_PATH))

            tail = ''
            if LOG_FILE.exists():
                lines = LOG_FILE.read_text(encoding='utf-8').strip().splitlines()
                # show the last two lines (URL and BODY) like LF
                last = [ln for ln in lines if ln.startswith('[PRICES]')]
                tail = '\n'.join(last[-2:])
            if tail:
                log.push('⎯⎯ Sent request ⎯⎯')
                for line in tail.splitlines():
                    # light redaction reusing your helper
                    log.push(_redact(line))

            log.push('✅ Prices: inserted into DuckDB table prices_hourly')
        except Exception as e:
            log.push(f'❌ Prices failed: {e}')

    with ui.card().classes('w-full max-w-3xl mx-auto'):
        ui.label('Controls').classes('text-lg font-semibold')
        with ui.row():
            ui.button('Session ID', on_click=lambda: asyncio.create_task(click_session()))
            ui.separator().classes('h-10')
            ui.button('Load Forecast', on_click=lambda: asyncio.create_task(click_load_forecast()))
            ui.button('Load Prices',   on_click=lambda: asyncio.create_task(click_load_prices()))
            ui.button('Preview Data', on_click=preview_latest)
        log

############--- Visualization Area ---############

    with ui.card().classes('w-full max-w-3xl mx-auto'):
        ui.label('Visualizations').classes('text-lg font-semibold')

        # --- Global date inputs for all charts in this card ---
        with ui.row().classes('w-full gap-3 items-end'):
            start_chart_date = ui.input(
                'From Operating Date (MM/DD/YYYY)',
                value=_date.today().strftime('%m/%d/%Y')
            ).classes('w-56')
            end_chart_date = ui.input(
                'To Operating Date (MM/DD/YYYY)',
                value=(_date.today() + _timedelta(days=6)).strftime('%m/%d/%Y')
            ).classes('w-56')
            ui.label('These dates filter all charts below.').classes('opacity-70')

        ################### ---- Visual helpers ---- ###################
        def _parse_range():
            """Return (start_dt, end_dt_inclusive) or None and notify on error."""
            try:
                s = datetime.strptime(start_chart_date.value.strip(), '%m/%d/%Y')
                e = datetime.strptime(end_chart_date.value.strip(), '%m/%d/%Y')
                if e < s:
                    ui.notify('End date must be on/after start date.', type='warning')
                    return None, None
                # inclusive end-of-day
                e = pd.to_datetime(e) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                s = pd.to_datetime(s)
                return s, e
            except Exception:
                ui.notify('Enter dates as MM/DD/YYYY.', type='warning')
                return None, None

        ################### -- TAB Layout -- ################
        with ui.tabs() as viz_tabs:
            t1 = ui.tab('Load Forecast')
            t2 = ui.tab('Prices')
            t3 = ui.tab('Historic Weather')
            t4 = ui.tab('Forecast Weather')

        with ui.tab_panels(viz_tabs, value=t1).classes('w-full'):
            # --- Load Forecast chart ---
            with ui.tab_panel(t1):
                ui.label('Load Forecast by Zone').classes('text-md font-medium')
                lf_chart = ui.echart(options={}).classes('w-full h-[360px] border rounded-xl')

                def update_lf_chart():
                    s, e = _parse_range()
                    if s is None:
                        return

                    sql = """
                        SELECT OperatingDTM, Interval,
                            wz_southcentral, wz_east, wz_west, wz_northcentral, wz_farwest,
                            wz_north, wz_southern, wz_coast,
                            lz_north, lz_west, lz_south, lz_houston
                        FROM vw_loadforecast_by_zone
                        WHERE OperatingDTM >= ? AND OperatingDTM <= ?
                        ORDER BY OperatingDTM, Interval
                    """
                    df = run_query(sql, (s, e))
                    if df.empty:
                        # mutate in place + refresh
                        lf_chart.options.clear()
                        lf_chart.options.update({
                            "title": {"text": "Load Forecast (no data)"},
                            "xAxis": {"type": "category", "data": []},
                            "yAxis": {"type": "value"},
                            "series": []
                        })
                        lf_chart.update()
                        return

                    import pandas as pd
                    df["OperatingDTM"] = pd.to_datetime(df["OperatingDTM"], errors="coerce")
                    df["Interval"] = pd.to_numeric(df["Interval"], errors="coerce").fillna(0).astype(int)
                    df["x"] = df["OperatingDTM"].dt.strftime("%Y-%m-%d") + " | I" + df["Interval"].astype(str).str.zfill(2)

                    ignore = {"OperatingDTM", "Interval", "x"}
                    y_cols = [c for c in df.columns if c not in ignore]

                    for c in y_cols:
                        df[c] = pd.to_numeric(df[c], errors="coerce")

                    df_agg = df.groupby("x", as_index=False)[y_cols].sum().sort_values("x")
                    x_vals = df_agg["x"].tolist()

                    new_opts = {
                        "title": {"text": ""},
                        "tooltip": {"trigger": "axis"},
                        "legend": {"data": y_cols, "type": "scroll"},
                        "xAxis": {"type": "category", "data": x_vals},
                        "yAxis": {"type": "value", "scale": True},
                        "grid": {"left": 60, "right": 20, "top": 50, "bottom": 60},
                        "dataZoom": [{"type": "slider"}, {"type": "inside"}],
                        "series": [
                            {
                                "type": "line",
                                "name": c,
                                "showSymbol": False,
                                "data": df_agg[c].where(pd.notna(df_agg[c]), None).tolist(),
                            }
                            for c in y_cols
                        ],
                    }

                    # IMPORTANT!!! mutate options dict, then refresh
                    lf_chart.options.clear()               
                    lf_chart.options.update(new_opts)
                    lf_chart.update()
                ui.button('Refresh', on_click=update_lf_chart).classes('mt-2')

            # --- Prices chart ---
            with ui.tab_panel(t2):
                ui.label('ERCOT Day Ahead Prices by Hub').classes('text-md font-medium')
                pr_chart = ui.echart(options={}).classes('w-full h-[360px] border rounded-xl')
                def update_pr_chart():
                    s, e = _parse_range()
                    if s is None:
                        return

                    sql = """
                        SELECT OperatingDTM,  Interval,  hb_hubavg,  hb_west,  hb_north,  hb_south, hb_houston,  hb_pan
                        FROM vw_prices_by_hub
                        WHERE OperatingDTM >= ? AND OperatingDTM <= ?
                        ORDER BY OperatingDTM,  Interval
                    """
                    df = run_query(sql, (s, e))
                    if df.empty:
                        # mutate in place + refresh
                        pr_chart.options.clear()
                        pr_chart.options.update({
                            "title": {"text": "Load Forecast (no data)"},
                            "xAxis": {"type": "category", "data": []},
                            "yAxis": {"type": "value"},
                            "series": []
                        })
                        pr_chart.update()
                        return

                    import pandas as pd
                    df["OperatingDTM"] = pd.to_datetime(df["OperatingDTM"], errors="coerce")
                    df["Interval"] = pd.to_numeric(df["Interval"], errors="coerce").fillna(0).astype(int)
                    df["x"] = df["OperatingDTM"].dt.strftime("%Y-%m-%d") + " | I" + df["Interval"].astype(str).str.zfill(2)

                    ignore = {"OperatingDTM", "Interval", "x"}
                    y_cols = [c for c in df.columns if c not in ignore]

                    for c in y_cols:
                        df[c] = pd.to_numeric(df[c], errors="coerce")

                    df_agg = df.groupby("x", as_index=False)[y_cols].sum().sort_values("x")
                    x_vals = df_agg["x"].tolist()

                    new_opts = {
                        "title": {"text": ""},
                        "tooltip": {"trigger": "axis"},
                        "legend": {"data": y_cols, "type": "scroll"},
                        "xAxis": {"type": "category", "data": x_vals},
                        "yAxis": {"type": "value", "scale": True},
                        "grid": {"left": 60, "right": 20, "top": 50, "bottom": 60},
                        "dataZoom": [{"type": "slider"}, {"type": "inside"}],
                        "series": [
                            {
                                "type": "line",
                                "name": c,
                                "showSymbol": False,
                                "data": df_agg[c].where(pd.notna(df_agg[c]), None).tolist(),
                            }
                            for c in y_cols
                        ],
                    }

                    # IMPORTANT!!! mutate options dict, then refresh
                    pr_chart.options.clear()               
                    pr_chart.options.update(new_opts)
                    pr_chart.update()
                ui.button('Refresh', on_click=update_pr_chart).classes('mt-2')

            # --- Historic Weather chart ---
            with ui.tab_panel(t3):
                ui.label('Historic Weather by Houston Suburb').classes('text-md font-medium')
                hw_chart = ui.echart(options={}).classes('w-full h-[360px] border rounded-xl')

                def update_hw_chart():
                    s, e = _parse_range()
                    if s is None:
                        return

                    sql = """
                        SELECT OperatingDTM, interval,
                            hist_temp_the_woodlands_tx,
                             hist_hum_the_woodlands_tx,  hist_wind_the_woodlands_tx,
                             hist_precip_the_woodlands_tx,  hist_temp_katy_tx,  hist_hum_katy_tx,
                             hist_wind_katy_tx,
                             hist_precip_katy_tx ,    
                            hist_temp_friendswood_tx ,            
                            hist_hum_friendswood_tx   ,           
                            hist_wind_friendswood_tx   ,      
                            hist_precip_friendswood_tx  ,   
                            hist_temp_baytown_tx         ,     
                            hist_hum_baytown_tx           ,    
                            hist_wind_baytown_tx           ,
                            hist_precip_baytown_tx         ,
                            hist_temp_houston_tx            ,   
                            hist_hum_houston_tx             ,
                            hist_wind_houston_tx           ,
                            hist_precip_houston_tx
                        FROM vw_historical_weather_by_city
                        WHERE OperatingDTM >= ? AND OperatingDTM <= ?
                        ORDER BY OperatingDTM, interval
                    """
                    df = run_query(sql, (s, e))
                    if df.empty:
                        # mutate in place + refresh
                        hw_chart.options.clear()
                        hw_chart.options.update({
                            "title": {"text": "Historic Weather (no data)"},
                            "xAxis": {"type": "category", "data": []},
                            "yAxis": {"type": "value"},
                            "series": []
                        })
                        hw_chart.update()
                        return

                    import pandas as pd
                    df["OperatingDTM"] = pd.to_datetime(df["OperatingDTM"], errors="coerce")
                    df["interval"] = pd.to_numeric(df["interval"], errors="coerce").fillna(0).astype(int)
                    df["x"] = df["OperatingDTM"].dt.strftime("%Y-%m-%d") + " | I" + df["interval"].astype(str).str.zfill(2)

                    ignore = {"OperatingDTM", "Interval", "x"}
                    y_cols = [c for c in df.columns if c not in ignore]

                    for c in y_cols:
                        df[c] = pd.to_numeric(df[c], errors="coerce")

                    df_agg = df.groupby("x", as_index=False)[y_cols].sum().sort_values("x")
                    x_vals = df_agg["x"].tolist()

                    new_opts = {
                        "title": {"text": ""},
                        "tooltip": {"trigger": "axis"},
                        "legend": {"data": y_cols, "type": "scroll"},
                        "xAxis": {"type": "category", "data": x_vals},
                        "yAxis": {"type": "value", "scale": True},
                        "grid": {"left": 60, "right": 20, "top": 50, "bottom": 60},
                        "dataZoom": [{"type": "slider"}, {"type": "inside"}],
                        "series": [
                            {
                                "type": "line",
                                "name": c,
                                "showSymbol": False,
                                "data": df_agg[c].where(pd.notna(df_agg[c]), None).tolist(),
                            }
                            for c in y_cols
                        ],
                    }

                    # IMPORTANT!!! mutate options dict, then refresh
                    hw_chart.options.clear()               
                    hw_chart.options.update(new_opts)
                    hw_chart.update()
                ui.button('Refresh', on_click=update_hw_chart).classes('mt-2')

            # --- Forecast Weather chart ---
            with ui.tab_panel(t4):
                ui.label('Forecasted Weather by Houston Suburb').classes('text-md font-medium')
                fw_chart = ui.echart(options={}).classes('w-full h-[360px] border rounded-xl')

                def update_fw_chart():
                    s, e = _parse_range()
                    if s is None:
                        return

                    sql = """
                        SELECT OperatingDTM                 ,                    
                                interval                     ,         
                            temp_the_woodlands_tx        ,       
                            hum_the_woodlands_tx         ,       
                            wind_the_woodlands_tx        ,       
                            precip_the_woodlands_tx      ,       
                            temp_katy_tx                 ,       
                            hum_katy_tx                  ,       
                            wind_katy_tx                 ,       
                            precip_katy_tx               ,       
                            temp_friendswood_tx          ,       
                            hum_friendswood_tx           ,       
                            wind_friendswood_tx          ,       
                            precip_friendswood_tx        ,       
                            temp_baytown_tx              ,       
                            hum_baytown_tx               ,       
                            wind_baytown_tx              ,       
                            precip_baytown_tx            ,       
                            temp_houston_tx              ,       
                            hum_houston_tx               ,       
                            wind_houston_tx              ,       
                            precip_houston_tx              
                        FROM vw_forecast_weather_by_city
                        WHERE OperatingDTM >= ? AND OperatingDTM <= ?
                        ORDER BY OperatingDTM, interval
                    """
                    df = run_query(sql, (s, e))
                    if df.empty:
                        # mutate in place + refresh
                        fw_chart.options.clear()
                        fw_chart.options.update({
                            "title": {"text": "Historic Weather (no data)"},
                            "xAxis": {"type": "category", "data": []},
                            "yAxis": {"type": "value"},
                            "series": []
                        })
                        fw_chart.update()
                        return

                    import pandas as pd
                    df["OperatingDTM"] = pd.to_datetime(df["OperatingDTM"], errors="coerce")
                    df["interval"] = pd.to_numeric(df["interval"], errors="coerce").fillna(0).astype(int)
                    df["x"] = df["OperatingDTM"].dt.strftime("%Y-%m-%d") + " | I" + df["interval"].astype(str).str.zfill(2)

                    ignore = {"OperatingDTM", "Interval", "x"}
                    y_cols = [c for c in df.columns if c not in ignore]

                    for c in y_cols:
                        df[c] = pd.to_numeric(df[c], errors="coerce")

                    df_agg = df.groupby("x", as_index=False)[y_cols].sum().sort_values("x")
                    x_vals = df_agg["x"].tolist()

                    new_opts = {
                        "title": {"text": ""},
                        "tooltip": {"trigger": "axis"},
                        "legend": {"data": y_cols, "type": "scroll"},
                        "xAxis": {"type": "category", "data": x_vals},
                        "yAxis": {"type": "value", "scale": True},
                        "grid": {"left": 60, "right": 20, "top": 50, "bottom": 60},
                        "dataZoom": [{"type": "slider"}, {"type": "inside"}],
                        "series": [
                            {
                                "type": "line",
                                "name": c,
                                "showSymbol": False,
                                "data": df_agg[c].where(pd.notna(df_agg[c]), None).tolist(),
                            }
                            for c in y_cols
                        ],
                    }

                    # IMPORTANT!!! mutate options dict, then refresh
                    fw_chart.options.clear()               
                    fw_chart.options.update(new_opts)
                    fw_chart.update()
                ui.button('Refresh', on_click=update_fw_chart).classes('mt-2')

    # === Historic Weather (Meteostat) — Houston suburbs fan-out ===
    with ui.card().classes('w-full max-w-3xl mx-auto'):
        ui.label('Historic Weather (Meteostat) — Houston suburbs fan-out').classes('text-lg font-semibold')
        with ui.row().classes('items-center gap-3 mx-auto'):
            hist_from = ui.input('From (MM/DD/YYYY)', value=(_date.today() - _timedelta(days=1)).strftime('%m/%d/%Y')).classes('w-full max-w-3xl mx-auto')
            hist_to   = ui.input('To (MM/DD/YYYY)',  value=(_date.today() + _timedelta(days=0)).strftime('%m/%d/%Y')).classes('w-full max-w-3xl mx-auto')
              #  from_input = ui.input('From Operating Date (MM/DD/YYYY)', value=_date.today().strftime('%m/%d/%Y')).classes('w-full max-w-3xl mx-auto')
             #to_input   = ui.input('To Operating Date (MM/DD/YYYY)',   value=(_date.today() + _timedelta(days=6)).strftime('%m/%d/%Y')).classes('w-full max-w-3xl mx-auto')
        log_hist = ui.log().classes('w-full')

        async def click_hist_weather():
            json_path = Path(__file__).resolve().parents[1] / "pipeline" / "houston_suburbs_coords.json"
            coords_map = json.loads(json_path.read_text(encoding="utf-8"))
            try:
                # read suburbs mapping
                
                if not json_path.exists():
                    log_hist.push(f"❌ Mapping file not found: {json_path}")
                    return
                

                # parse dates; make end inclusive (add 1 day)
                start_dt = datetime.strptime(hist_from.value, "%m/%d/%Y")
                end_dt   = datetime.strptime(hist_to.value, "%m/%d/%Y") + _timedelta(days=1)

                # run the whole batch in a worker thread 
                def run_all():
                    total = 0
                    # instantiate once with db_path
                    ld = Loads(db_path=str(DB_PATH))
                    try:
                        for city, geo in coords_map.items():
                            lat = float(geo["lat"]); lon = float(geo["lon"])

                            def city_log(msg, _city=city):
                                try:
                                    log_hist.push(f"[{_city}] {msg}")
                                except Exception:
                                    pass

                            log_hist.push(f"▶ {city}: {lat:.4f}, {lon:.4f} …")
                            from pipeline.historic_weather import load_historic_hourly_to_duckdb
                            added = load_historic_hourly_to_duckdb(
                                lat=lat, lon=lon,
                                start=start_dt, end=end_dt,
                                db_path=str(DB_PATH),
                            )
                            city_log(f"chunk complete: +{added} rows")
                            total += (added or 0)
                            log_hist.push(f"✓ {city}: upserted {added} rows")
                        return total
                    finally:
                        ld.close()
                log_hist.push("▶ Running historic backfill for Houston suburbs…")
                grand_total = await asyncio.to_thread(run_all)
                log_hist.push(f"✅ All suburbs complete. Total rows upserted: {grand_total}")

            except Exception as e:
                log_hist.push(f"❌ Historic Weather failed: {e}")

        ui.button('Load Historic Weather', on_click=lambda: asyncio.create_task(click_hist_weather()))

    
    with ui.card().classes('w-full max-w-3xl mx-auto'):
        ui.label('Hourly Weather Maintenance').classes('text-lg font-semibold')
        log_hw = ui.log().classes('w-full')

        async def run_hw_import():
            try:
                # add ../utilities to import path
                utils_dir = Path(__file__).resolve().parent.parent / "utilities"
                if str(utils_dir) not in sys.path:
                    sys.path.insert(0, str(utils_dir))
                from utilities import hw_maintenance  

                log_hw.push("▶ Running hw_maintenance...")
                # offload to a worker thread so UI stays responsive
                n = await asyncio.to_thread(hw_maintenance.run)  # or hw_maintenance.main()
                log_hw.push(f"✅ Completed. Rows touched: {n if n is not None else 'unknown'}")
            except Exception as e:
                log_hw.push(f"❌ hw_maintenance failed: {e}")

        ui.button('Run HW Maintenance', on_click=lambda: asyncio.create_task(run_hw_import()))

    # === Forecast Weather card ===
    with ui.card().classes('w-full max-w-3xl mx-auto'):
        ui.label('Forecast Weather (Open-Meteo) — Houston suburbs fan-out').classes('text-lg font-semibold')
        with ui.row().classes('items-center gap-3'):
            fc_from = ui.input('Start Date (YYYY-MM-DD)', value=_start_default)
            fc_to   = ui.input('End Date (YYYY-MM-DD)',   value=_end_default)
        log_fc  = ui.log().classes('w-full')

        async def click_fcst_weather():
            log_fc.push('▶ Forecast Weather: fetching…')
            try:
                from pipeline.forecast_weather import load_forecast_hourly_batch_to_duckdb_from_file
                # same mapping path convention as Historic Weather
                json_path = Path(__file__).resolve().parents[1] / "pipeline" / "houston_suburbs_coords.json"
                if not json_path.exists():
                    log_fc.push(f"❌ Mapping file not found: {json_path}")
                    return

                def logger(msg: str):  # per-city log passthrough
                    try:
                        log_fc.push(msg)
                    except Exception:
                        pass

                # run the fan-out in a thread so UI stays responsive
                def run_all():
                    return load_forecast_hourly_batch_to_duckdb_from_file(
                        mapping_file=str(json_path),
                        start_date=fc_from.value or None,
                        end_date=fc_to.value or None,
                        db_path=str(DB_PATH),
                        log_fn=logger,
                    )

                total = await asyncio.to_thread(run_all)
                log_fc.push(f'✅ All suburbs complete. Total rows upserted: {total}')

            except Exception as e:
                log_fc.push(f'❌ Forecast Weather failed: {e}')

        ui.button('Load Forecast Weather', on_click=lambda: asyncio.create_task(click_fcst_weather()))

    
# === Deep Learning Forecast card ===
    with ui.card().classes('w-full max-w-3xl mx-auto'):
        ui.label('Deep Learning Forecast (HB_Houston)').classes('text-lg font-semibold')

        with ui.row().classes('items-end gap-4'):
            dl_date = ui.input(
                label='Delivery Date (yyyy-mm-dd)',
                placeholder=_today.isoformat(),
                validation={'Use yyyy-mm-dd': lambda v: bool(re.fullmatch(r'\d{4}-\d{2}-\d{2}', (v or '').strip()))}
            ).classes('w-60')
            run_btn = ui.button('Run Model', icon='play_arrow', color='primary')

        dl_log = ui.log(max_lines=2000).classes('h-56 w-full')
        ui.separator()

        dl_cols = [
            {'name': 'OperatingDTM', 'label': 'OperatingDTM', 'field': 'OperatingDTM', 'align': 'left'},
            {'name': 'Interval', 'label': 'Interval', 'field': 'Interval', 'align': 'right'},
            {'name': 'hb_houston_pred', 'label': 'hb_houston_pred', 'field': 'hb_houston_pred', 'align': 'right'},
        ]
        dl_table = ui.table(columns=dl_cols, rows=[]).classes('w-full')

        async def run_model():
            date_str = (dl_date.value or '').strip()
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date_str):
                dl_log.push("❌ Invalid date. Please enter yyyy-mm-dd.")
                return

            dl_table.rows = []
            dl_log.clear()
            dl_log.push(f"▶ Running model for {date_str} …")

            # stream logs via callback; run heavy work off the event loop
            def logger(msg: str):
                try:
                    dl_log.push(msg)
                except Exception:
                    pass

            try:
                df = await asyncio.to_thread(
                    run_deep_learning_forecast,
                    date_str,
                    str(DB_PATH)
                    , dl_log.push
                    , 30)
                # normalize table rows
                if not df.empty:
                    df = df[["OperatingDTM", "Interval", "hb_houston_pred"]].copy()
                    if pd.api.types.is_datetime64_any_dtype(df["OperatingDTM"]):
                        df["OperatingDTM"] = pd.to_datetime(df["OperatingDTM"]).dt.strftime("%Y-%m-%d")
                    df["Interval"] = pd.to_numeric(df["Interval"], errors="coerce").fillna(0).astype(int)
                    df["hb_houston_pred"] = pd.to_numeric(df["hb_houston_pred"], errors="coerce").astype(float).round(2).apply(lambda x: f"${x:,.2f}")  # dolla dolla bills
                    dl_table.rows = df.to_dict(orient="records")
                    dl_log.push(f"✅ Forecast ready: {len(df)} rows.")
                else:
                    dl_log.push("⚠️ No rows returned.")
            except Exception as e:
                dl_log.push(f"❌ Run failed: {e}")

        run_btn.on('click', lambda: asyncio.create_task(run_model()))

ui.run(title='ERCOT DA Forecasts', reload=True, storage_secret=STORAGE_SECRET)
