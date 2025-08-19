from nicegui import ui, app
import asyncio, os, sys, secrets
from datetime import date as _date, timedelta as _timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.scrapes_bak import Scrapes
from pipeline.loads import Loads
from pipeline.jobs_bak import Jobs
from pipeline.features import build_feature_df
from modeling.train import fit_with_backtest
from storage.paths import DB_PATH, WEATHER_POINTS
from pipeline.config_bak import ErcotConfig
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv(filename='.env', usecwd=True))
print("UID loaded:", bool(os.getenv("ERCOT_UID")))
print("PWD loaded:", bool(os.getenv("ERCOT_PWD")))
STORAGE_SECRET = os.getenv('NICEGUI_STORAGE_SECRET') or secrets.token_urlsafe(32)

cfg = ErcotConfig.load()
scrapes = Scrapes(base_urls={"ercot": cfg.base_url}, cfg=cfg)


log = ui.log(max_lines=400).classes('w-full h-48')

#loads = Loads(DB_PATH)
#jobs = Jobs(scrapes, loads, WEATHER_POINTS)

# ---- lazy service factory (avoids DB open during reload race) ----
_services = {}  # {'scrapes':..., 'loads':..., 'jobs':...}

def get_jobs() -> Jobs:
    if 'jobs' not in _services:
        cfg = ErcotConfig.load()
        _services['scrapes'] = Scrapes(base_urls={"ercot": cfg.base_url}, cfg=cfg)
        _services['loads'] = Loads(DB_PATH)
        _services['jobs'] = Jobs(_services['scrapes'], _services['loads'], WEATHER_POINTS)
    return _services['jobs']

def shutdown():
    # close DuckDB so the file is released before reload
    ld = _services.get('loads')
    if ld is not None:
        ld.close()
app.on_shutdown(shutdown)
# ------------------------------------------------------------------


async def run_job(coro, label: str):
    log.push(f'▶ {label} started…')
    try:
        result = await coro
        log.push(f'✅ {label} done.')
        return result
    except Exception as e:
        log.push(f'❌ {label} failed: {e}')
        raise

@ui.page('/')
def main():
    ui.label('ERCOT Day-Ahead Forecasting Console').classes('text-2xl font-bold')

    sid_label = ui.label('SID: (none yet)').classes('text-sm text-gray-600')

    # NEW: from/to date inputs (MM/DD/YYYY)
    from_input = ui.input('From Operating Date (MM/DD/YYYY)',
                          value=_date.today().strftime('%m/%d/%Y')).classes('w-56')
    to_input   = ui.input('To Operating Date (MM/DD/YYYY)',
                          value=(_date.today() + _timedelta(days=6)).strftime('%m/%d/%Y')).classes('w-56')


    async def do_session():
        jobs = get_jobs()
        sid = await run_job(jobs.run_session(), 'Session')
        if sid:
            app.storage.user['sid'] = sid
            sid_label.text = f'SID: {sid}'
            # optional: show the redacted URL used for the request
            try:
                ui.notify(f"Session URL: {jobs.scrapes.last_request_url}")
            except Exception:
                pass  


    with ui.card().classes('w-full'):
        ui.label('Ingestion Jobs (Scrapes ➜ Loads)').classes('text-lg font-semibold')
        with ui.row():
            ui.button('1) Session', on_click=lambda: asyncio.create_task(do_session()))
            ui.button('2) Prices (T+1)', on_click=lambda: asyncio.create_task(run_job(get_jobs().run_prices(), 'Prices')))
            ui.button(
                '3) Load Forecast',
                on_click=lambda: asyncio.create_task(
                    run_job(get_jobs().run_load_fcst(from_input.value, to_input.value), 'Load Forecast')
                ),
            )
            ui.button('4) Hist Weather', on_click=lambda: asyncio.create_task(run_job(get_jobs().run_hist_weather(), 'Hist Weather')))
            ui.button('5) Fcst Weather (T+1)', on_click=lambda: asyncio.create_task(run_job(get_jobs().run_fcst_weather(), 'Fcst Weather')))
        log

    with ui.card().classes('w-full'):
        ui.label('Model Training & Backtest').classes('text-lg font-semibold')

        async def start_training():
            log.push('▶ Building features…')
            df = build_feature_df(DB_PATH)
            log.push(f'Features shape: {df.shape}')
            log.push('▶ Training…')
            results = fit_with_backtest(df, target_col='lmp')
            app.storage.user['metrics'] = results['metrics']
            app.storage.user['feat_imp'] = results['feature_importance']
            app.storage.user['trained_at'] = results['generated_at']
            log.push('✅ Training complete.')

        ui.button('Train Models', on_click=lambda: asyncio.create_task(start_training()))

        metrics_card = ui.card().classes('w-full')

        def show_results():
            metrics = app.storage.user.get('metrics', {})
            feat_imp = app.storage.user.get('feat_imp', [])
            trained_at = app.storage.user.get('trained_at', None)

            with metrics_card:
                metrics_card.clear()
                if trained_at:
                    ui.label(f'Latest Results (UTC {trained_at})').classes('font-medium')
                if metrics:
                    ui.table(
                        columns=[{'name':'metric','label':'Metric','field':'metric'},
                                 {'name':'value','label':'Value','field':'value'}],
                        rows=[{'metric':k, 'value':v} for k,v in metrics.items()],
                    ).classes('w-full')
                else:
                    ui.label('No metrics yet.')

                ui.separator()
                ui.label('Feature Importance').classes('font-medium')
                if feat_imp:
                    ui.table(
                        columns=[{'name':'feature','label':'Feature','field':'feature'},
                                 {'name':'gain','label':'Gain','field':'gain'}],
                        rows=feat_imp
                    ).classes('w-full')
                else:
                    ui.label('None yet.')

        ui.button('Show Results', on_click=show_results)

ui.run(title='ERCOT DA Forecasts', reload=True,storage_secret=STORAGE_SECRET)
