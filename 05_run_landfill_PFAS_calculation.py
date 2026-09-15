# 05_run_landfill_PFAS_calculation.py

from pathlib import Path
from src.landfill_PFAS_calculation import run_landfill_central_bau


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    run_landfill_central_bau(project_root=project_root, landfill_type="MLF")