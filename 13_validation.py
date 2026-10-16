# 13_validation.py

from pathlib import Path
from src.validation_landfill import run_landfill_validation


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    # LMOP data (lmopdata.csv) and the previous allocation results (LFMass_Accum_StatusQuo.csv)
    validation_dir = project_root / "data" / "validation"
    run_landfill_validation(project_root=project_root, validation_dir=validation_dir)
