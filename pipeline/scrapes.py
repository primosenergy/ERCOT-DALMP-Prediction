# pipeline/scrapes.py
from typing import List, Dict, Tuple, Optional, Union
from datetime import date, datetime
from urllib.parse import urlencode

import requests

from .config import ErcotConfig, PivotReportConfig, make_report_body

class Scrapes:
    """Synchronous scrapers using 'requests'. Keeps URL params explicit and logs a redacted URL."""
    def __init__(self, base_urls: Optional[Dict[str, str]] = None, api_keys: Optional[Dict[str, str]] = None, cfg: Optional[ErcotConfig] = None):
        self.base = base_urls or {}
        self.api_keys = api_keys or {}
        self.cfg = cfg or ErcotConfig.load()
        self.pivot_cfg = PivotReportConfig()
        self.last_request_url: Optional[str] = None  # redacted preview
        self.timeout = self.cfg.timeout

    def _redact(self, s: str) -> str:
        if not s:
            return s
        out = s
        if self.cfg.uid: out = out.replace(self.cfg.uid, "****")
        if self.cfg.pwd: out = out.replace(self.cfg.pwd, "****")
        return out

    # -------- Session (returns raw SID text) --------
    def get_session_id(self) -> str:
        url = f"{self.cfg.base_url}{self.cfg.session_endpoint}"
        method = self.cfg.session_method.upper()

        if not self.cfg.uid or not self.cfg.pwd:
            raise RuntimeError("Missing ERCOT_UID or ERCOT_PWD in environment (.env).")

        params = {"uid": self.cfg.uid, "pwd": self.cfg.pwd}
        qs = urlencode(params)
        full_url = f"{url}?{qs}" if method == "GET" else url
        self.last_request_url = self._redact(full_url)

        if method == "GET":
            r = requests.get(url, headers=self.cfg.headers, params=params, timeout=self.timeout)
        elif method == "POST":
            r = requests.post(url, headers=self.cfg.headers, json=params, timeout=self.timeout)
        else:
            raise ValueError(f"Unsupported session request type: {method}")

        r.raise_for_status()
        sid = r.text.strip()
        if not sid or "Exception" in sid:
            raise RuntimeError(f"Session error: {sid}")
        return sid

    # -------- Pivot (Load Forecast) --------
    @staticmethod
    def _fmt_operating_date(dt: Union[str, date]) -> str:
        if isinstance(dt, date):
            return dt.strftime("%m/%d/%Y")
        try:
            return datetime.strptime(dt, "%Y-%m-%d").strftime("%m/%d/%Y")
        except ValueError:
            return dt  # assume MM/DD/YYYY already

    def get_load_forecast(self, session_id: str, from_operating_date: Union[str, date], to_operating_date: Union[str, date]) -> bytes:
        """POST pivot body; include SID + UserID in URL query (not body). Returns JSON bytes."""
        endpoint = self.pivot_cfg.endpoint
        url = f"{self.cfg.base_url}/{endpoint.lstrip('/')}"
        headers = {"Accept": "application/json", "Content-Type": "application/json"}

        # body
        fmm = self._fmt_operating_date(from_operating_date)
        tmm = self._fmt_operating_date(to_operating_date)
        body = make_report_body(fmm, tmm)

        # URL params
        sid_param  = self.pivot_cfg.sid_param     # default SID
        user_param = self.pivot_cfg.user_id_param # default UserID

        if not self.cfg.uid:
            raise RuntimeError("Missing ERCOT_UID in environment for pivot call.")

        params = {
            sid_param:  session_id,
            user_param: self.cfg.uid,
        }

        # Save a redacted URL for the UI
        qs = urlencode(params)
        full_url = f"{url}?{qs}"
        self.last_request_url = self._redact(full_url)

        # Send
        r = requests.post(url, headers=headers, params=params, json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.content  # JSON bytes
