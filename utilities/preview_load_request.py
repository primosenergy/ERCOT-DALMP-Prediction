# utilities/preview_load_request.py
from __future__ import annotations
import argparse, json, os, sys
from datetime import date, datetime, timedelta
from urllib.parse import urlencode
from pathlib import Path

# --- make sure we can import pipeline/* from anywhere ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# --------------------------------------------------------

from dotenv import load_dotenv, find_dotenv
# load .env from the repo root
load_dotenv(dotenv_path=ROOT / '.env') or load_dotenv(find_dotenv(filename='.env', usecwd=True))

# project imports (module-level builder you asked for)
from pipeline.config_bak import ErcotConfig, PivotReportConfig, make_report_body


def _fmt_mmddyyyy(d: str | date) -> str:
    if isinstance(d, date):
        return d.strftime("%m/%d/%Y")
    try:
        datetime.strptime(d, "%m/%d/%Y"); return d
    except ValueError:
        pass
    try:
        return datetime.strptime(d, "%Y-%m-%d").strftime("%m/%d/%Y")
    except ValueError:
        return d

def build_pivot_request(sid: str, from_date: str | date, to_date: str | date):
    cfg = ErcotConfig.load()
    pivot = PivotReportConfig()

    base = cfg.base_url.rstrip("/")
    endpoint = pivot.endpoint.lstrip("/")
    url = f"{base}/{endpoint}"

    method = "POST"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    fmm = _fmt_mmddyyyy(from_date)
    tmm = _fmt_mmddyyyy(to_date)
    body = make_report_body(fmm, tmm)

    sid_param = os.getenv("PIVOT_SID_PARAM", "sid")  # set to SID in .env if needed
    qs_full = urlencode({sid_param: sid})
    qs_safe = urlencode({sid_param: "****"})

    return {
        "method": method,
        "full_url": f"{url}?{qs_full}",
        "safe_url": f"{url}?{qs_safe}",
        "headers": headers,
        "body": body,
        "debug": {
            "base_url": base, "endpoint": endpoint, "sid_param": sid_param, "from": fmm, "to": tmm,
        },
    }

def main():
    ap = argparse.ArgumentParser(description="DRY RUN: preview Load Forecast (pivot) request without sending it.")
    ap.add_argument("--sid", default="REPLACE_ME", help="Session ID to include as query param")
    ap.add_argument("--from", dest="from_date", default=date.today().strftime("%m/%d/%Y"),
                    help="From Operating Date (MM/DD/YYYY or YYYY-MM-DD). Default: today")
    ap.add_argument("--to", dest="to_date", default=(date.today() + timedelta(days=6)).strftime("%m/%d/%Y"),
                    help="To Operating Date (MM/DD/YYYY or YYYY-MM-DD). Default: today+6")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print the JSON body")
    args = ap.parse_args()

    req = build_pivot_request(args.sid, args.from_date, args.to_date)

    print("\n=== LOAD FORECAST (PIVOT) — DRY RUN ===")
    print(f"Method : {req['method']}")
    print(f"URL    : {req['full_url']}")
    print(f"URL(*) : {req['safe_url']}   # redacted")
    print("Headers:")
    for k, v in req["headers"].items():
        print(f"  {k}: {v}")
    print("Body:")
    print(json.dumps(req["body"], indent=2 if args.pretty else None))
    print("\nDebug:")
    for k, v in req["debug"].items():
        print(f"  {k}: {v}")

    curl_body = json.dumps(req["body"]).replace('"', '\\"')
    print("\nCurl (redacted SID):")
    print(f"""curl -X {req['method']} "{req['safe_url']}" -H "Accept: application/json" -H "Content-Type: application/json" --data-raw "{curl_body}" """)

if __name__ == "__main__":
    main()
