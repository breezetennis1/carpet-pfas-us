from pathlib import Path

from src.allocation_county import allocate_central_bau_to_counties


def main() -> None:
    project_root = Path(__file__).resolve().parent
    allocate_central_bau_to_counties(project_root)


if __name__ == "__main__":
    main()