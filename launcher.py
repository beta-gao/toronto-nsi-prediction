from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PIPELINE_STEPS = [
    ("Clean raw crime datasets", BASE_DIR / "clean_crime_datasets.py", DATA_DIR),
    ("Build monthly neighbourhood safety dataset", BASE_DIR / "data_processing.py", DATA_DIR),
    ("Train and evaluate models", BASE_DIR / "main.py", BASE_DIR),
]


def run_step(step_name: str, script_path: Path, workdir: Path) -> None:
    print(f"\n=== {step_name} ===")
    print(f"Running: {script_path.name} (cwd={workdir})")
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=workdir,
        check=True,
    )


def main() -> None:
    for step_name, script_path, workdir in PIPELINE_STEPS:
        run_step(step_name, script_path, workdir)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
