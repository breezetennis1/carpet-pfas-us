from pathlib import Path

from src.plot_county_distribution import plot_county_figure3


def main() -> None:
    project_root = Path(__file__).resolve().parent
    plot_county_figure3(project_root)


if __name__ == "__main__":
    main()