from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.paths import get_output_dirs


plt.rcParams["font.family"] = "Arial"


OUTPUT_ORDER = [
    "stock",
    "emission",
    "landfill_accu",
    "landfill_emission",
]

MC_VAR_MAP = {
    "stock": "F_stock_total",
    "emission": "F6_total",
    "landfill_accu": "F_accu_landfill_total",
    "landfill_emission": "PFAS_landfill_leachate_total",
}

MC_YLABELS = {
    "stock": "tonnes",
    "emission": "tonnes/year",
    "landfill_accu": "tonnes",
    "landfill_emission": "kg/year",
}

# kg -> tonnes, except leachate (about 1 kg/yr), plotted in kg/year
MC_UNIT_DIVISOR = {
    "stock": 1000,
    "emission": 1000,
    "landfill_accu": 1000,
    "landfill_emission": 1,
}

PANEL_TITLES = {
    "stock": "In-use stock",
    "emission": "In-use emission",
    "landfill_accu": "Landfill accumulation",
    "landfill_emission": "Landfill leachate",
}


def plot_mc_sobol_2x4(
    project_root: Path,
    year: int = 2020,
) -> None:
    dirs = get_output_dirs(project_root)

    mc_path = dirs["tables_monte_carlo"] / "MC_BAU_timeseries_summary.csv"
    sobol_path = dirs["tables_sobol"] / "sobol_results_all_years.csv"

    fig_dir = dirs["figures_sobol"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    mc = pd.read_csv(mc_path)
    sobol = pd.read_csv(sobol_path)

    fig = plt.figure(figsize=(23, 8))

    gs = GridSpec(
        2,
        4,
        figure=fig,
        height_ratios=[0.8, 1.0],
        hspace=0.30,
        wspace=0.25,
    )

    axes_top = [fig.add_subplot(gs[0, i]) for i in range(4)]
    axes_bottom = [fig.add_subplot(gs[1, i]) for i in range(4)]

    top_labels = ["(a)", "(b)", "(c)", "(d)"]
    bottom_labels = ["(e)", "(f)", "(g)", "(h)"]

    # =========================
    # Row 1: Monte Carlo
    # =========================
    for i, out in enumerate(OUTPUT_ORDER):
        ax = axes_top[i]

        var = MC_VAR_MAP[out]
        x = mc["year"]

        y_med = mc[f"{var}_median"] / MC_UNIT_DIVISOR[out]
        y_low = mc[f"{var}_p2_5"] / MC_UNIT_DIVISOR[out]
        y_high = mc[f"{var}_p97_5"] / MC_UNIT_DIVISOR[out]

        ax.plot(x, y_med, linewidth=1.8, label="Median")
        ax.fill_between(
            x,
            y_low,
            y_high,
            alpha=0.3,
            label="95% uncertainty interval",
        )

        ax.set_title(
            f"{top_labels[i]} {PANEL_TITLES[out]}",
            fontsize=16,
        )
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel(MC_YLABELS[out], fontsize=14)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1990, 2060)

        # Make first-row Monte Carlo panels wider
        ax.set_box_aspect(None)

        if i == 0:
            ax.legend(frameon=False, fontsize=11, loc="upper right")

    # =========================
    # Row 2: Sobol, selected year
    # =========================
    for i, out in enumerate(OUTPUT_ORDER):
        ax = axes_bottom[i]

        df_plot = (
            sobol[
                (sobol["year"] == year)
                & (sobol["output"] == out)
                ]
            .sort_values("ST", ascending=True)
            .copy()
        )

        # Keep only top 10 most influential parameters
        df_plot = df_plot.sort_values("ST", ascending=False).head(10)
        df_plot = df_plot.sort_values("ST", ascending=True)

        ax.barh(df_plot["parameter"], df_plot["ST"])

        ax.set_title(
            f"{bottom_labels[i]} {PANEL_TITLES[out]} ({year})",
            fontsize=16,
        )
        ax.set_xlabel("Total Sobol index (ST)", fontsize=14)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, alpha=0.3)

        # Keep second-row Sobol panels close to the original shape
        ax.set_box_aspect(1.0)

    fig_path = fig_dir / f"MC_Sobol_2x4_{year}.png"
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    plot_mc_sobol_2x4(project_root, year=2020)