# bootstrap_scaffold.py
import argparse
from pathlib import Path

# Relative to where you run this script
ROOT = Path.cwd()

DIRS = [
    "ercot_app",
    "pipeline",
    "modeling",
    "storage",
    "ProjectMain/db",
]

FILES = [
    # top-level
    "requirements.txt",
    ".gitignore",
    "README.md",
    # app
    "ercot_app/app.py",
    # pipeline
    "pipeline/__init__.py",
    "pipeline/scrapes.py",
    "pipeline/loads.py",
    "pipeline/jobs.py",
    "pipeline/features.py",
    # modeling
    "modeling/__init__.py",
    "modeling/train.py",
    "modeling/evaluate.py",
    # storage
    "storage/__init__.py",
    "storage/paths.py",
]

def write_empty(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"SKIP  {path.relative_to(ROOT)} (exists)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Truncate to empty
    with open(path, "w", encoding="utf-8"):
        pass
    print(f"WROTE {path.relative_to(ROOT)}")

def main():
    parser = argparse.ArgumentParser(description="Create empty ERCOT NiceGUI scaffold.")
    parser.add_argument("--overwrite", action="store_true", help="Truncate existing files to empty")
    parser.add_argument("--gitkeep", action="store_true", help="Add .gitkeep files to empty dirs")
    args = parser.parse_args()

    # Create directories
    for d in DIRS:
        p = ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        print(f"DIR   {p.relative_to(ROOT)}")
        if args.gitkeep:
            write_empty(p / ".gitkeep", overwrite=True)

    # Create empty files
    for f in FILES:
        write_empty(ROOT / f, overwrite=args.overwrite)

    print("\nDone! Open this folder in VS Code and start filling files.")

if __name__ == "__main__":
    main()
