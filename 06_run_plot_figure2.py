from pathlib import Path

from src.paths import get_output_dirs
from src.plot_figure2 import plot_figure2


def main() -> None:
    project_root = Path(__file__).resolve().parent
    print(f"Project root: {project_root}")

    dirs = get_output_dirs(project_root)
    tables_dir = dirs["tables_national"]

    print(f"National tables dir: {tables_dir}")
    print(f"National tables dir exists: {tables_dir.exists()}")

    # High estimate figure for main text
    figure_high = dirs["figures_main"] / "Figure_2_high.png"
    plot_figure2(
        tables_dir=tables_dir,
        figure_path=figure_high,
        estimate="HIGH"
    )
    print(f"Saved: {figure_high}")

    # Low estimate figure for SI
    figure_low = dirs["figures_si"] / "Figure_S3_low.png"
    plot_figure2(
        tables_dir=tables_dir,
        figure_path=figure_low,
        estimate="LOW"
    )
    print(f"Saved: {figure_low}")

    # Central estimate figure for SI
    figure_central = dirs["figures_si"] / "Figure_S3_central.png"
    plot_figure2(
        tables_dir=tables_dir,
        figure_path=figure_central,
        estimate="CENTRAL"
    )
    print(f"Saved: {figure_central}")


if __name__ == "__main__":
    main()