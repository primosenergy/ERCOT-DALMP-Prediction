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
    base_url: str
    session_endpoint: str
    session_method: str = "GET"           # "GET" or "POST"
    headers: Dict[str, str] = None
    uid: Optional[str] = None             # from .env: ERCOT_UID
    pwd: Optional[str] = None             # from .env: ERCOT_PWD
    timeout: float = 20.0

    @staticmethod
    def load() -> "ErcotConfig":
        j = _load_json(CFG_FILE)

        def env(name: str, default=None):
            v = os.getenv(name)
            return v if v is not None else default

        base_url = env("ERCOT_BASE_URL", j.get("baseUrl", "https://replace-me"))
        session_endpoint = env("ERCOT_SESSION_ENDPOINT", j.get("session_endpoint", "/InitiateSession"))
        session_method = (env("ERCOT_SESSION_METHOD", j.get("default_params", {}).get("session_request_type", "GET")) or "GET").upper()
        headers = j.get("default_headers", {"Accept": "*/*", "Connection": "keep-alive"})

        uid = env("ERCOT_UID", None)
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
    # POST to this path on the base_url
    update_path: str = os.getenv("PIVOT_UPDATE_PATH", "UpdateData")

    # query string names
    uid_param: str  = os.getenv("PIVOT_UID_PARAM", "uid")
    sid_param: str  = os.getenv("PIVOT_SID_PARAM", "sid")
    method_param: str = os.getenv("PIVOT_METHOD_PARAM", "m")

    # method value in the query string
    method_value: str = os.getenv("PIVOT_METHOD_VALUE", "DAL_XM.GetPivotReport")

    # body (unchanged defaults; used by make_report_body)
    report_view: str = "SevenDayLoadForecastERCOTView"
    market: int = 2
    key_columns: tuple = ("OperatingDTM","Interval","Month","LocationType","Location","DSTFlag")
    pivot_columns: tuple = ("Location",)
    data_columns: tuple = ({"columnName": "ForecastMW", "columnOperator": "SUM"},)

    
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
