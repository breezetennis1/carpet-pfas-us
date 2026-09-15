from pathlib import Path
from src.sobol_analysis import run_sobol_analysis


def main() -> None:
    project_root = Path(__file__).resolve().parent

    run_sobol_analysis(
        project_root=project_root,
        N=256,                  # start with 128 for testing if needed
        calc_second_order=False
    )


if __name__ == "__main__":
    main()