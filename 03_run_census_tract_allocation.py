from pathlib import Path

from src.allocation_census_tract import allocate_national_to_census_tracts


def main() -> None:
    project_root = Path(__file__).resolve().parent
    allocate_national_to_census_tracts(project_root)


if __name__ == "__main__":
    main()