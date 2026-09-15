from pathlib import Path

from src.monte_carlo import run_and_save_monte_carlo


def main() -> None:
    project_root = Path(__file__).resolve().parent

    run_and_save_monte_carlo(
        project_root=project_root,
        n_runs=10000,   # change to 5000 or 10000 later
        seed=42,
    )


if __name__ == "__main__":
    main()