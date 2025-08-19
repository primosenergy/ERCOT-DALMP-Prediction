# save as print_tree.py and run: python print_tree.py > scaffold.txt
from pathlib import Path

IGNORE = {'.git', '.venv', '__pycache__'}
root = Path('.').resolve()

def walk(dir_path: Path, prefix=""):
  entries = sorted([p for p in dir_path.iterdir() if p.name not in IGNORE], key=lambda p: (p.is_file(), p.name.lower()))
  for i, p in enumerate(entries):
    connector = "└── " if i == len(entries)-1 else "├── "
    print(prefix + connector + p.name + ("/" if p.is_dir() else ""))
    if p.is_dir():
      extension = "    " if i == len(entries)-1 else "│   "
      walk(p, prefix + extension)

print(root.name + "/")
walk(root)
