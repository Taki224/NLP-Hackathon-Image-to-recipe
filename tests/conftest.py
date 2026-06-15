import sys
from pathlib import Path

# ponytail: app.py has no package/__init__, so put its dir on the path directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "ui" / "src"))
