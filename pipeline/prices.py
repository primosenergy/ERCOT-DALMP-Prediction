# pipeline/prices.py
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
PIVOT_UPDATE_PATH = os.getenv('PIVOT_UPDATE_PATH', 'UpdateData') 
UID = os.getenv('ERCOT_UID', '')
# Optional override of method name (default given)
PIVOT_METHOD_VALUE = os.getenv('PIVOT_METHOD_VALUE', 'DAL_XM.GetPivotReport')

# Log file
LOG_FILE = Path('logs/api.log')
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# SID file from SessionID.py
SID_FILE = Path('ProjectMain/db/session_id.txt')

def _get_db_path() -> str:
    """Resolve the DuckDB path, preferring storage.paths.get_db_path()."""
    try:
        # Prefer your shared resolver
        from storage.paths import get_db_path
        p = Path(get_db_path())
    except Exception:
        # Fallback to ProjectMain/db/data.duckdb (create folder if needed)
        p = Path('ProjectMain/db/data.duckdb').resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        print("DEBUG DB path used by loader:", p)

    return str(Path(p).resolve())

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

def _build_body(from_mmddyyyy: str, to_mmddyyyy: str,
                location_prefix: str = "hb_",
                primary_owner: str = "q") -> dict:
    """Pivot request body for Hourly DA LMPs."""
    return {
        "reportView": "rpt_PricesHourlyERCOTView",
        "lsfilter": {
            "Filters": [
                {"comparison": "ge", "field": "OperatingDTM", "type": "date",   "value": from_mmddyyyy},
                {"comparison": "le", "field": "OperatingDTM", "type": "date",   "value": to_mmddyyyy},
                {"comparison": "eq", "field": "Location",     "type": "string", "value": location_prefix},
                {"comparison": "eq", "field": "PrimaryOwner", "type": "string", "value": primary_owner},
            ],
            "SortOrders": [
                {"property": "OperatingDTM", "direction": "ASC", "text": "Operating Date", "data": "root"},
            ],
            "Timezone": ""
        },
        "keyColumns": ["OperatingDTM","Time","Month","Location"],
        "pivotColumns": [],
        "reportDataColumns": [{"columnName":"Price","columnOperator":"AVG"}],
        "DatabaseConnection": "",
        "market": 2  # Day-Ahead
    }

def run_prices(from_date: Union[str, date],
               to_date:   Union[str, date],
               location_prefix: str = "hb_",
               primary_owner:  str = "q",
               db_path: str | None = None) -> int:
    """
    POST {{baseUrl}}/UpdateData?uid={{uid}}&sid={{sid}}&m=DAL_XM.GetPivotReport with JSON body.
    Saves raw, then upserts into DuckDB via Loads.load_prices_pivot_json().
    """
    if not (BASE_URL and UID):
        raise RuntimeError('Missing ERCOT_BASE_URL or ERCOT_UID in .env')
    if not SID_FILE.exists():
        raise RuntimeError('No session_id.txt found; click Session ID first.')

    sid = SID_FILE.read_text(encoding='utf-8').strip()
    if not sid:
        raise RuntimeError('Empty SID; click Session ID again.')

    # URL & params
    url = f'{BASE_URL}/{PIVOT_UPDATE_PATH.lstrip("/")}'
    params = {'uid': UID, 'sid': sid, 'm': PIVOT_METHOD_VALUE}

    # Body
    body = _build_body(_fmt_mmddyyyy(from_date),
                       _fmt_mmddyyyy(to_date),
                       location_prefix=location_prefix,
                       primary_owner=primary_owner)

    # Log full URL + single-line JSON
    full_url = f'{url}?{urlencode(params)}'
    LOG_FILE.write_text((LOG_FILE.read_text() if LOG_FILE.exists() else '') +
                        f'[PRICES][URL]  {full_url}\n' +
                        f'[PRICES][BODY] {json.dumps(body, separators=(",",":"))}\n',
                        encoding='utf-8')

    # POST
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    r = requests.post(url, params=params, json=body, headers=headers, timeout=30)
    r.raise_for_status()
    payload = r.content  # JSON bytes

    # Save raw
    from datetime import datetime as _dt
    raw_dir = Path('ProjectMain/db/rawprices'); raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f'prices_hourly_{_dt.utcnow().strftime("%Y%m%d_%H%M%S")}.json'
    raw_path.write_bytes(payload)

    # Insert into DuckDB
    resolved_db_path = str(Path(db_path).resolve()) if db_path else _get_db_path()
    print("DEBUG DB path used by loader:", resolved_db_path)

    from pipeline.loads import Loads
    loader = Loads(resolved_db_path)
    try:
        loader.load_prices_pivot_json(payload)
    finally:
        loader.close()
    return 0
