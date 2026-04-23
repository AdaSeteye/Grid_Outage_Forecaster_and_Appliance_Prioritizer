"""Run ``generate_data.py`` then ``export_ui.py``."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    subprocess.check_call([sys.executable, str(root / "generate_data.py")])
    subprocess.check_call([sys.executable, str(root / "export_ui.py")])


if __name__ == "__main__":
    main()
