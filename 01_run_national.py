from pathlib import Path
from src.runner import run_all_cases


def main() -> None:
    project_root = Path(__file__).resolve().parent

    for scenario in ["BAU", "RECYCLE_24", "PHASE_OUT"]:
        run_all_cases(project_root, scenario=scenario)


if __name__ == "__main__":
    main()