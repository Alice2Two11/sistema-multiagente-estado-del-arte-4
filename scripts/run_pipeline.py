#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

VENV_PYTHON = Path(os.environ.get("ESTADO_ARTE_PYTHON", "/content/venv_estado_arte/bin/python"))
PROJECT_DIR = Path(os.environ.get("THESIS_PROJECT_DIR", "/content/proyecto_estado_arte"))

def main() -> int:
    if not VENV_PYTHON.is_file():
        print(f"ERROR: no existe {VENV_PYTHON}", file=sys.stderr)
        return 2

    if not PROJECT_DIR.is_dir():
        print(f"ERROR: no existe {PROJECT_DIR}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["THESIS_PROJECT_DIR"] = str(PROJECT_DIR)

    command = [
        str(VENV_PYTHON),
        "-m",
        "src.orchestration.pipeline_orchestrator",
        "--project-dir",
        str(PROJECT_DIR),
        *sys.argv[1:],
    ]

    print("Ejecutando:")
    print(" ".join(command))

    result = subprocess.run(
        command,
        cwd=str(PROJECT_DIR),
        env=env,
    )

    return result.returncode

if __name__ == "__main__":
    raise SystemExit(main())
