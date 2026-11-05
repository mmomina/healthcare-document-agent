import re
import sys
from pathlib import Path

source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data.txt")
out_dir = Path("data/synthetic")
out_dir.mkdir(parents=True, exist_ok=True)

header = re.compile(r"^doc\d+_[a-z]+\.txt$")
current_name = None
current_lines = []
saved = []

def flush():
    if current_name is None:
        return
    text = "\n".join(current_lines).strip("\n") + "\n"
    (out_dir / current_name).write_text(text, encoding="utf-8")
    saved.append(current_name)

for line in source.read_text(encoding="utf-8").splitlines():
    if header.match(line.strip()):
        flush()
        current_name = line.strip()
        current_lines = []
    elif current_name is not None:
        current_lines.append(line)
flush()

print(f"Saved {len(saved)} files to {out_dir}:")
for name in saved:
    print(" -", name)
