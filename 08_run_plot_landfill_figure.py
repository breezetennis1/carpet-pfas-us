from pathlib import Path
from src.plot_landfill_figure import make_landfill_figure


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    make_landfill_figure(project_root)