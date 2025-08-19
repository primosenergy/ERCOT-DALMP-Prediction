# pipeline/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, os
from typing import Dict, Optional

CFG_FILE = Path(__file__).parent / "request_config.json"   # optional

def _load_json(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@dataclass(frozen=True)
class ErcotConfig:
    # Core
    base_url: str
    session_endpoint: str
    session_method: str = "GET"           # "GET" or "POST"
    headers: Dict[str, str] = None

    # Creds (use env; do NOT commit these)
    uid: Optional[str] = None
    pwd: Optional[str] = None

    # HTTP
    timeout: float = 20.0

    @staticmethod
    def load() -> "ErcotConfig":
        j = _load_json(CFG_FILE)  # optional; good place for non-secret headers and endpoints

        def env(name: str, default=None):
            v = os.getenv(name)
            return v if v is not None else default

        # Prefer env, fallback to json
        base_url = env("ERCOT_BASE_URL", j.get("baseUrl", "https://replace-me"))
        session_endpoint = env("ERCOT_SESSION_ENDPOINT", j.get("session_endpoint", "/InitiateSession"))
        session_method = (env("ERCOT_SESSION_METHOD", j.get("default_params", {}).get("session_request_type", "GET")) or "GET").upper()
        headers = j.get("default_headers", {"Accept": "*/*", "Connection": "keep-alive"})

        uid = env("ERCOT_UID", None)   # keep secrets in env only
        pwd = env("ERCOT_PWD", None)
        timeout = float(env("HTTP_TIMEOUT", "20"))

        return ErcotConfig(
            base_url=base_url.rstrip("/"),
            session_endpoint=session_endpoint,
            session_method=session_method,
            headers=headers,
            uid=uid,
            pwd=pwd,
            timeout=timeout,
        )

@dataclass(frozen=True)
class PivotReportConfig:
    endpoint: str = "DAL_XM.GetPivotReport"
    request_type: str = "POST"
    report_view: str = "SevenDayLoadForecastERCOTView"
    market: int = 2
    key_columns: tuple = ("OperatingDTM","Interval","Month","LocationType","Location","DSTFlag")
    pivot_columns: tuple = ("Location",)
    data_columns: tuple = ({"columnName": "ForecastMW", "columnOperator": "SUM"},)

    # NEW: configurable param names & location
    sid_param: str = os.getenv("PIVOT_SID_PARAM", "sid")           # change to "SID" if needed
    user_id_param: str = os.getenv("PIVOT_USER_PARAM", "UserID")   # server error referenced "UserID"
    user_id_location: str = os.getenv("PIVOT_USER_LOCATION", "query")  # "query" | "body" | "header" | "both"
    
def make_report_body(from_operating_date_mmddyyyy: str, to_operating_date_mmddyyyy: str) -> dict:
    """Build the pivot body with a date range [from, to]."""
    return {
        "reportView": PivotReportConfig.report_view,
        "lsfilter": {
            "Filters": [
                {"comparison": "ge", "field": "OperatingDTM", "type": "date", "value": from_operating_date_mmddyyyy},
                {"comparison": "le", "field": "OperatingDTM", "type": "date", "value": to_operating_date_mmddyyyy},
            ],
            "SortOrders": [
                {"property":"OperatingDTM","direction":"ASC","text":"Operating Date","data":"root"},
                {"property":"Interval","direction":"ASC","text":"Interval","data":"root"},
            ],
            "Timezone": ""
        },
        "keyColumns": list(PivotReportConfig.key_columns),
        "pivotColumns": list(PivotReportConfig.pivot_columns),
        "reportDataColumns": list(PivotReportConfig.data_columns),
        "DatabaseConnection": "",
        "market": PivotReportConfig.market,
    }