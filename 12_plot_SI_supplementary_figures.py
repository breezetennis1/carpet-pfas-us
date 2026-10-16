"""
Supplementary Information figures added during manuscript revision.

Each SI figure gets its own self-contained plotting function (reads its
input data, builds the figure, saves it). Add new figures by writing a new
plot_*() function below and calling it from the __main__ block -- no new
files needed.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# =====================================================
# Global font / style settings
# (kept consistent with src/plot_landfill_figure.py)
# =====================================================
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10


def style_axis(ax):
    """Shared axis styling, matching src/plot_landfill_figure.py::style_axis."""
    ax.set_facecolor("white")

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1)

    ax.tick_params(axis="x", colors="black", direction="out", length=6, width=1)
    ax.tick_params(axis="y", colors="black", direction="out", length=6, width=1)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


def get_si_figure_output_dir(project_root: Path) -> Path:
    """Shared output location for this script's figures (outputs/figures/SI)."""
    output_dir = project_root / "outputs" / "figures" / "SI"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_carpet_enduse_trend(project_root: Path) -> None:
    """
    Figure SX. U.S. carpet and area rug sales by end-use market, 2012-2025.
    Residential = residential replacement + residential builder + manufactured
    housing. Commercial = commercial replacement + commercial new
    construction. 2013 not available. In the 2012 report, the "residential
    builder" category is labeled "residential contract"; the dollar value is
    included in the residential total but this label difference has not been
    confirmed to denote an identical category across years. Source: Catalina
    Research, Inc., as reported in Floor Covering Weekly, The Statistical
    Report (2012, 2014-2025 editions).
    """
    csv_path = project_root / "data" / "raw" / "carpet_enduse_split_2012_2025.csv"

    df = pd.read_csv(csv_path)
    df = df[
        ["year", "residential_pct", "commercial_pct", "transportation_pct"]
    ].copy()

    df = df.sort_values("year").reset_index(drop=True)

    series_style = {
        "residential_pct": {
            "label": "Residential",
            "color": "#0072B2",
            "marker": "o",
            "linestyle": "-",
        },
        "commercial_pct": {
            "label": "Commercial",
            "color": "#D55E00",
            "marker": "s",
            "linestyle": "-",
        },
        "transportation_pct": {
            "label": "Transportation & other",
            "color": "#009E73",
            "marker": "^",
            "linestyle": "-",
        },
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for col, style in series_style.items():
        ax.plot(
            df["year"],
            df[col],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=6,
            label=style["label"],
        )

    full_years = set(range(int(df["year"].min()), int(df["year"].max()) + 1))
    missing_years = sorted(full_years - set(df["year"]))
    for year in missing_years:
        ax.text(
            year,
            71,
            "2013 not available",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontstyle="italic",
            color="dimgray",
            fontname="Arial",
        )

    ax.set_xlabel("Year", fontname="Arial")
    ax.set_ylabel(
        "Share of total carpet & area rug sales (%)", fontname="Arial"
    )
    ax.set_ylim(0, 75)
    ax.set_xticks(sorted(full_years))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    style_axis(ax)

    ax.legend(loc="upper right", frameon=False)

    output_dir = get_si_figure_output_dir(project_root)
    fig_path = output_dir / "carpet_enduse_trend_2012_2025.png"

    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {fig_path}")


def plot_national_carpet_data(project_root: Path) -> None:
    """
    Figure S2. National carpet data: (a) national and per capita sales, 2002-2025 (CARE 2002-2019,
    FCW 2008-2025 with transportation carpet excluded; per capita sales from the combined series used
    in the model), and (b) national sales, import and export, 1990-2060 (model inputs, data/Basic_data.csv).
    """
    care = pd.read_excel(
        project_root / "data" / "raw" / "national_inputs" / "carpet_background_data_2.xlsx",
        sheet_name="CARE DATA",
    )
    care.columns = [" ".join(str(c).split()) for c in care.columns]
    care = care.dropna(subset=["Year"]).astype({"Year": int}).set_index("Year")
    pop = care["US_POP （million）"]
    sales_care = care["Sales(CARE) (million m2)"].dropna()
    sales_fcw = care["Sales(FCW) (million m2)"].dropna()
    sales_combined = care["Sales combination data (FCW & CARE) (million m2)"].dropna()
    per_capita = sales_combined / pop.loc[sales_combined.index]

    basic = pd.read_csv(project_root / "data" / "Basic_data.csv").set_index("year")
    if not (basic.loc[sales_combined.index, "V_sales"] - sales_combined).abs().max() < 1e-6:
        raise ValueError("Combined CARE/FCW sales differ from V_sales in data/Basic_data.csv")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) national and per capita sales
    ax_a.plot(sales_care.index, sales_care, color="#404040", linewidth=1.8, label="CARE")
    ax_a.plot(sales_fcw.index, sales_fcw, color="#E31A1C", linewidth=1.8, label="FCW")
    ax_a.set_xlim(2002, int(sales_combined.index.max()))
    ax_a.set_xticks(range(2002, int(sales_combined.index.max()) + 1, 2))
    plt.setp(ax_a.get_xticklabels(), rotation=45, ha="right")
    ax_a.set_ylim(0, 1300)
    ax_a.set_xlabel("Year")
    ax_a.set_ylabel("National carpet sales (million m$^2$)")
    style_axis(ax_a)
    ax_pc = ax_a.twinx()
    ax_pc.plot(per_capita.index, per_capita, color="#1F78B4", linestyle="--", linewidth=1.8, label="Per capita")
    ax_pc.set_ylim(0, 4.0)
    ax_pc.set_ylabel("Per capita carpet sales (m$^2$/cap)", color="#1F78B4")
    ax_pc.tick_params(axis="y", colors="#1F78B4", direction="out", length=6, width=1)
    ax_pc.spines["right"].set_color("#1F78B4")
    ax_pc.spines["top"].set_visible(False)
    handles = ax_a.get_legend_handles_labels()[0] + ax_pc.get_legend_handles_labels()[0]
    ax_a.legend(handles=handles, loc="lower left", frameon=False, ncol=3)
    ax_a.text(0.02, 0.98, "(a)", transform=ax_a.transAxes, fontsize=13, fontweight="bold", va="top")

    # (b) national sales, import and export (model inputs)
    for col, label, color in [("V_sales", "Sales", "#404040"), ("V_import", "Import", "#E31A1C"),
                              ("V_export", "Export", "#1F78B4")]:
        ax_b.plot(basic.index, basic[col], color=color, linewidth=1.8, label=label)
    ax_b.set_xlim(1990, 2060)
    ax_b.set_ylim(0, 1300)
    ax_b.set_xlabel("Year")
    ax_b.set_ylabel("National carpet sales, import and export\n(million m$^2$)")
    style_axis(ax_b)
    ax_b.legend(loc="center right", frameon=False)
    ax_b.text(0.02, 0.98, "(b)", transform=ax_b.transAxes, fontsize=13, fontweight="bold", va="top")

    fig.tight_layout()
    fig_path = get_si_figure_output_dir(project_root) / "Figure_S2_national_carpet_data.png"
    fig.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent

    plot_carpet_enduse_trend(project_root)
    plot_national_carpet_data(project_root)
    # Add new SI figure calls here, e.g.:
    # plot_next_si_figure(project_root)
