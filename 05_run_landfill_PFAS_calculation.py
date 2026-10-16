# 05_run_landfill_PFAS_calculation.py

from pathlib import Path
from src.landfill_PFAS_calculation import run_landfill_central_bau


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    # Census tract -> landfill assignment results from step 4 (see README, "Data"):
    # annual_PFAS_accum_central_BAU/ (kg/yr) and annual_carpt_volume_central_BAU/ (carpet area, million m2/yr)
    transport_dir = project_root / "data" / "transport_assignment"
    run_landfill_central_bau(project_root=project_root, transport_dir=transport_dir, landfill_type="MLF")
