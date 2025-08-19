# loadforecast.py
from __future__ import annotations
import os, json, requests
from datetime import date, datetime
from urllib.parse import urlencode
from pathlib import Path
from typing import Union
from dotenv import load_dotenv, find_dotenv

# Load env first
load_dotenv(find_dotenv(filename='.env', usecwd=True))

# Required env
BASE_URL = os.getenv('ERCOT_BASE_URL', '').rstrip('/')           
PIVOT_UPDATE_PATH = os.getenv('PIVOT_UPDATE_PATH', 'UpdateData') #  "UpdateData" path
UID = os.getenv('ERCOT_UID', '')
# Optional override of method name (default given)
PIVOT_METHOD_VALUE = os.getenv('PIVOT_METHOD_VALUE', 'DAL_XM.GetPivotReport')

# Log file
LOG_FILE = Path('logs/api.log')
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# SID file from SessionID.py
SID_FILE = Path('ProjectMain/db/session_id.txt')

# DuckDB loader
def _get_db_path() -> str:
    # Prefer  existing storage path if present; otherwise use local default
    try:
        from storage.paths import get_db_path
        return get_db_path
    except Exception:
        Path('ProjectMain/db').mkdir(parents=True, exist_ok=True)
        print("DEBUG DB path used by loader:",get_db_path)
        return str(Path('ProjectMain/db/data.duckdb').resolve())

def _log(line: str) -> None:
    LOG_FILE.write_text((LOG_FILE.read_text() if LOG_FILE.exists() else '') + line + '\n', encoding='utf-8')

def _fmt_mmddyyyy(d: Union[str, date]) -> str:
    if isinstance(d, date):
        return d.strftime('%m/%d/%Y')
    # already MM/DD/YYYY?
    try:
        datetime.strptime(d, '%m/%d/%Y'); return d
    except ValueError:
        pass
    # convert YYYY-MM-DD
    try:
        return datetime.strptime(d, '%Y-%m-%d').strftime('%m/%d/%Y')
    except ValueError:
        return d

def _build_body(from_mmddyyyy: str, to_mmddyyyy: str) -> dict:
    """ pivot request body (range filter on OperatingDTM)."""
    return {
        "reportView": "SevenDayLoadForecastERCOTView",
        "lsfilter": {
            "Filters": [
                {"comparison": "ge", "field": "OperatingDTM", "type": "date", "value": from_mmddyyyy},
                {"comparison": "le", "field": "OperatingDTM", "type": "date", "value": to_mmddyyyy},
            ],
            "SortOrders": [
                {"property": "OperatingDTM", "direction": "ASC", "text": "Operating Date", "data": "root"},
                {"property": "Interval", "direction": "ASC", "text": "Interval", "data": "root"},
            ],
            "Timezone": ""
        },
        "keyColumns": ["OperatingDTM","Interval","Month","LocationType","Location","DSTFlag"],
        "pivotColumns": [],
        "reportDataColumns": [{"columnName":"ForecastMW","columnOperator":"SUM"}],
        "DatabaseConnection": "",
        "market": 2
    }

def run_load_forecast(from_date: Union[str, date], to_date: Union[str, date], db_path: str | None = None) -> int:
    """POST {{baseUrl}}/UpdateData?uid={{uid}}&sid={{sid}}&m=DAL_XM.GetPivotReport with JSON body."""
    if not (BASE_URL and UID):
        raise RuntimeError('Missing ERCOT_BASE_URL or ERCOT_UID in .env')
    if not SID_FILE.exists():
        raise RuntimeError('No session_id.txt found; click Session ID first.')

    sid = SID_FILE.read_text(encoding='utf-8').strip()
    if not sid:
        raise RuntimeError('Empty SID; click Session ID again.')

    # URL & params
    url = f'{BASE_URL}/{PIVOT_UPDATE_PATH.lstrip("/")}'
    params = {
        'uid': UID,
        'sid': sid,
        'm': PIVOT_METHOD_VALUE,
    }

    # Body with range
    body = _build_body(_fmt_mmddyyyy(from_date), _fmt_mmddyyyy(to_date))

    # Log full URL and pretty body (per  request)
    full_url = f'{url}?{urlencode(params)}'
    _log(f'[LOAD_FORECAST][URL]  {full_url}')
    _log(f'[LOAD_FORECAST][BODY] {json.dumps(body, separators=(",",":"))}')

    # POST
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    r = requests.post(url, params=params, json=body, headers=headers, timeout=30)
    r.raise_for_status()
    payload = r.content  # JSON bytes


    # Save raw response
    from datetime import datetime as _dt
    raw_dir = Path('ProjectMain/db/raw'); raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f'load_forecast_{_dt.utcnow().strftime("%Y%m%d_%H%M%S")}.json'
    raw_path.write_bytes(payload)
    _log(f'[LOAD_FORECAST][RAW]  {raw_path}')


    # Insert into DuckDB
    resolved_db_path = str(Path(db_path).resolve()) if db_path else _get_db_path()
    print("DEBUG DB path used by loader:", resolved_db_path)

    from pipeline.loads import Loads
    loader = Loads(resolved_db_path)
    try:
        loader.load_load_fcst_json(payload)
    finally:
        loader.close()
    return 0


