from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reproducibility" / "output"
SCRIPTS = Path(__file__).resolve().parent


def run(script: str) -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run([sys.executable, str(SCRIPTS / script)], cwd=str(ROOT), env=env, check=True)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    run("make_tables.py")
    run("make_model.py")
    run("verify.py")
    print("Reproducibility run completed.")


if __name__ == "__main__":
    main()
