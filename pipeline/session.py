# SessionID.py
from __future__ import annotations
import os, requests
from urllib.parse import urlencode
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load env from project root
load_dotenv(find_dotenv(filename='.env', usecwd=True))

BASE_URL = os.getenv('ERCOT_BASE_URL', '').rstrip('/')
SESSION_ENDPOINT = os.getenv('INITIATE_SESSION_ENDPOINT', '/InitiateSession')
UID = os.getenv('ERCOT_UID', '')
PWD = os.getenv('ERCOT_PWD', '')

# Where to save the latest SID (overwrite each click)
SID_FILE = Path('ProjectMain/db/session_id.txt')
SID_FILE.parent.mkdir(parents=True, exist_ok=True)

# Simple file logger
LOG_FILE = Path('logs/api.log')
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

def _log(line: str) -> None:
    LOG_FILE.write_text((LOG_FILE.read_text() if LOG_FILE.exists() else '') + line + '\n', encoding='utf-8')

def get_and_save_session() -> str:
    """GET {base}{/InitiateSession}?uid=..&pwd=..; writes full URL to log; overwrites SID file."""
    if not (BASE_URL and UID and PWD):
        raise RuntimeError('Missing ERCOT_BASE_URL / ERCOT_UID / ERCOT_PWD in .env')

    url = f'{BASE_URL}{SESSION_ENDPOINT}'
    params = {'uid': UID, 'pwd': PWD}

    # Log the full URL (contains credentials, per your request)
    full_url = f"{url}?{urlencode(params)}"
    _log(f'[SESSION][URL] {full_url}')

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()

    sid = r.text.strip()
    if not sid or 'Exception' in sid:
        raise RuntimeError(f'Session error: {sid}')

    SID_FILE.write_text(sid, encoding='utf-8')
    return sid
