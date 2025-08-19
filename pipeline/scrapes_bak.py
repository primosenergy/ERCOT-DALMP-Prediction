# pipeline/scrapes.py
from typing import List, Dict, Tuple, Optional, Union
import asyncio
from datetime import date, datetime
import httpx
from urllib.parse import urlencode
from .config_bak import ErcotConfig, PivotReportConfig, make_report_body

class Scrapes:
    """Async scrapers with retry/timeout; session ID uses your real endpoint."""
    def __init__(self, base_urls: Dict[str, str] | None = None, api_keys: Dict[str, str] | None = None, cfg: Optional[ErcotConfig] = None):
        self.base = base_urls or {}
        self.api_keys = api_keys or {}
        self.cfg = cfg or ErcotConfig.load()
        self.pivot_cfg = PivotReportConfig()
        self.last_request_url: Optional[str] = None

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        # simple retries
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self.cfg.timeout) as client:
                    resp = await client.request(method, url, **kwargs)
                resp.raise_for_status()
                return resp
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(1.2 * (attempt + 1))

    async def get_session_id(self) -> str:
        """Call /InitiateSession and return the raw SID text; logs a redacted URL."""
        url = f"{self.cfg.base_url}{self.cfg.session_endpoint}"
        method = self.cfg.session_method.upper()

        if not self.cfg.uid or not self.cfg.pwd:
            raise RuntimeError("Missing ERCOT_UID or ERCOT_PWD in environment (.env).")

        params = {"uid": self.cfg.uid, "pwd": self.cfg.pwd}

        # Build full URL string (for debugging) and redact secrets
        qs = urlencode(params)
        full_url = f"{url}?{qs}" if method == "GET" else url
        safe_url = full_url.replace(self.cfg.uid, "****").replace(self.cfg.pwd, "****")

        # store for UI/debug
        self.last_request_url = safe_url

        if method == "GET":
            resp = await self._request("GET", url, headers=self.cfg.headers, params=params)
        elif method == "POST":
            resp = await self._request("POST", url, headers=self.cfg.headers, json=params)
        else:
            raise ValueError(f"Unsupported session request type: {method}")

        sid = resp.text.strip()  # API returns raw SID or an error blob
        if not sid or "Exception" in sid:
            # Bubble up a clearer error
            raise RuntimeError(f"Session error: {sid}")
        return sid

    # --- placeholder kept for now; we’ll replace once you share those blocks ---
    async def get_prices_tomorrow(self, session_id: str) -> bytes:
        csv = "timestamp,lmp\n2025-08-12 00:00,30.0\n2025-08-12 01:00,28.5\n"
        await asyncio.sleep(0.02)
        return csv.encode()

    # ---------- NEW: pivot report (Load Forecast) ----------
    @staticmethod
    def _fmt_operating_date(dt: Union[str, date]) -> str:
        if isinstance(dt, date):
            return dt.strftime("%m/%d/%Y")
        # allow YYYY-MM-DD or MM/DD/YYYY strings
        try:
            return datetime.strptime(dt, "%Y-%m-%d").strftime("%m/%d/%Y")
        except ValueError:
            return dt  # assume already MM/DD/YYYY

    async def get_load_forecast(self, session_id: str, from_operating_date, to_operating_date) -> bytes:
        url = f"{self.cfg.base_url}/{self.pivot_cfg.endpoint.lstrip('/')}"
        headers = {"Accept": "application/json", "Content-Type": "application/json"}

        fmm = self._fmt_operating_date(from_operating_date)
        tmm = self._fmt_operating_date(to_operating_date)
        body = make_report_body(fmm, tmm)

        sid_param = self.pivot_cfg.sid_param            # e.g., "sid" or "SID"
        user_param = self.pivot_cfg.user_id_param       # e.g., "UserID"
        user_loc   = self.pivot_cfg.user_id_location    # "query" | "body" | "header" | "both"

        if not self.cfg.uid:
            raise RuntimeError("Missing ERCOT_UID in environment (.env) for pivot call.")

        # Always include SID in query
        params = {sid_param: session_id}

        # Include UserID where configured
        if user_loc in ("query", "both"):
            params[user_param] = self.cfg.uid
        if user_loc in ("body", "both"):
            body[user_param] = self.cfg.uid
        if user_loc in ("header", "both"):
            headers[user_param] = self.cfg.uid

        # If you added instrumentation earlier, keep it (helps you see exactly what was sent)
        try:
            self._record_request_debug(
                method="POST", url=url, headers=headers, params=params, json_body=body,
                redact_keys=[sid_param, user_param]
            )
        except Exception:
            pass

        resp = await self._request("POST", url, headers=headers, params=params, json=body)
        return resp.content
    # -------------------------------------------------------

    # keep placeholders; we’ll fill these later

    async def get_hist_weather(self, latlon_list: List[Tuple[float,float]]) -> Dict[Tuple[float,float], bytes]:
        async def fake(loc):
            csv = "timestamp,temp\n2025-08-11 00:00,30.0\n2025-08-11 01:00,29.7\n"
            await asyncio.sleep(0.01)
            return loc, csv.encode()
        results = await asyncio.gather(*[fake(ll) for ll in latlon_list])
        return dict(results)

    async def get_forecast_weather(self, latlon_list: List[Tuple[float,float]], target_day: date) -> Dict[Tuple[float,float], bytes]:
        async def fake(loc):
            csv = "timestamp,temp\n2025-08-12 00:00,31.0\n2025-08-12 01:00,30.5\n"
            await asyncio.sleep(0.01)
            return loc, csv.encode()
        results = await asyncio.gather(*[fake(ll) for ll in latlon_list])
        return dict(results)
