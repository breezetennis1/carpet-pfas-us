# 05_run_landfill_PFAS_calculation.py

from pathlib import Path
from src.landfill_PFAS_calculation import run_landfill_central_bau

# Collaborator's census tract -> landfill assignment results (read directly, not copied):
# annual_PFAS_accum_central_BAU/ (kg/yr) and annual_carpt_volume_central_BAU/ (carpet area, million m2/yr)
TRANSPORT_DIR = Path(
    r"C:\Users\mchen48\Box\01 Research\PFASs\PFASs_in_Carpet\01 Paper writing\processed_data\09292026"
)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    run_landfill_central_bau(project_root=project_root, transport_dir=TRANSPORT_DIR, landfill_type="MLF")
