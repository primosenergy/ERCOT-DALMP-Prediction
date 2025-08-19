from pathlib import Path

def get_db_path() -> Path:
    """Always point to the ROOT ProjectMain/db/data.duckdb (not the ercot_app copy)."""
    repo_root = Path(__file__).resolve().parents[1]     # <repo>/
    p = repo_root / "ProjectMain" / "db" / "data.duckdb"
    return p