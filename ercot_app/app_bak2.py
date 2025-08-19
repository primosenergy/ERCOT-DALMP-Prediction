# ercot_app/app.py
from nicegui import ui, app as nice_app
from fastapi import FastAPI
import asyncio, os, sys, secrets, json
from datetime import date as _date, timedelta as _timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.scrapes import Scrapes
from pipeline.loads import Loads
from pipeline.jobs import Jobs
from storage.paths import DB_PATH, WEATHER_POINTS
from pipeline.config import ErcotConfig
from dotenv import load_dotenv, find_dotenv

# Load .env
load_dotenv(find_dotenv(filename='.env', usecwd=True))
STORAGE_SECRET = os.getenv('NICEGUI_STORAGE_SECRET') or secrets.token_urlsafe(32)

# FastAPI + NiceGUI
fastapi_app = FastAPI()
ui.run_with(fastapi_app)

# Services (lazy so reload is safe)
_services = {}

def get_jobs() -> Jobs:
    if 'jobs' not in _services:
        cfg = ErcotConfig.load()
        _services['scrapes'] = Scrapes(base_urls={"ercot": cfg.base_url}, cfg=cfg)
        _services['loads'] = Loads(DB_PATH)
        _services['jobs'] = Jobs(_services['scrapes'], _services['loads'], WEATHER_POINTS)
    return _services['jobs']

def shutdown():
    ld = _services.get('loads')
    if ld is not None:
        ld.close()
nice_app.on_shutdown(shutdown)

log = ui.log(max_lines=400).classes('w-full h-48')

async def run_in_thread(fn, *args, label: str = "Task"):
    log.push(f'▶ {label} started…')
    try:
        result = await asyncio.to_thread(fn, *args)
        log.push(f'✅ {label} done.')
        return result
    except Exception as e:
        log.push(f'❌ {label} failed: {e}')
        raise

@ui.page('/')
def main():
    ui.label('ERCOT Day-Ahead Forecasting Console').classes('text-2xl font-bold')

    sid_label = ui.label('SID: (none yet)').classes('text-sm text-gray-600')

    # date inputs
    from_input = ui.input('From Operating Date (MM/DD/YYYY)', value=_date.today().strftime('%m/%d/%Y')).classes('w-56')
    to_input   = ui.input('To Operating Date (MM/DD/YYYY)',   value=(_date.today() + _timedelta(days=6)).strftime('%m/%d/%Y')).classes('w-56')

    async def do_session():
        jobs = get_jobs()
        sid = await run_in_thread(jobs.run_session, label='Session')
        if sid:
            nice_app.storage.user['sid'] = sid
            sid_label.text = f'SID: {sid}'
            try:
                ui.notify(f"Session URL: {jobs.scrapes.last_request_url}")
            except Exception:
                pass

    async def do_load_fcst():
        jobs = get_jobs()
        await run_in_thread(jobs.run_load_fcst, from_input.value, to_input.value, label='Load Forecast')
        # show the actual URL used (redacted)
        req_url = getattr(jobs.scrapes, "last_request_url", None)
        if req_url:
            log.push(f"LOAD REQ URL: {req_url}")

    with ui.card().classes('w-full'):
        ui.label('Ingestion Jobs (Scrapes ➜ Loads)').classes('text-lg font-semibold')
        with ui.row():
            ui.button('1) Session', on_click=lambda: asyncio.create_task(do_session()))
            ui.button('2) Prices (T+1)', on_click=lambda: asyncio.create_task(run_in_thread(get_jobs().run_prices, label='Prices')))
            ui.button('3) Load Forecast', on_click=lambda: asyncio.create_task(do_load_fcst()))
        log

ui.run(title='ERCOT DA Forecasts', reload=True, storage_secret=STORAGE_SECRET)
