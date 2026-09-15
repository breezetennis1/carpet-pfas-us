from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

# =====================================================
# Global font settings
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

START_YEAR = 1990
END_YEAR = 2060
YEARS = [str(y) for y in range(START_YEAR, END_YEAR + 1)]


def find_state_shapefile(project_root: Path) -> Path:
    candidates = list(project_root.glob("data/**/tl_2020_us_state.shp"))

    if len(candidates) == 0:
        raise FileNotFoundError(
            "Cannot find tl_2020_us_state.shp anywhere under data/"
        )

    return candidates[0]


def prepare_landfill_gdf(csv_path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.astype(str).str.strip()

    x_candidates = [
        "facility_x",
        "longitude",
        "Longitude",
        "LONGITUDE",
        "lon",
        "x",
    ]
    y_candidates = [
        "facility_y",
        "latitude",
        "Latitude",
        "LATITUDE",
        "lat",
        "y",
    ]

    x_col = next((c for c in x_candidates if c in df.columns), None)
    y_col = next((c for c in y_candidates if c in df.columns), None)

    if x_col is None or y_col is None:
        raise ValueError(
            f"Cannot find coordinate columns. Available columns: {list(df.columns)}"
        )

    df[YEARS] = df[YEARS].apply(pd.to_numeric, errors="coerce").fillna(0)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[x_col], df[y_col]),
        crs="EPSG:4269",
    )

    return gdf


def style_axis(ax):
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


def plot_landfill_map_percentile(
    ax,
    gdf,
    states_sub,
    value_col,
    unit_label,
    cbar_position,
    marker_size_factor,
    marker_size_min=0.5,
    marker_size_max=None,
):
    plot_gdf = gdf.copy()
    plot_gdf = plot_gdf[plot_gdf[value_col].fillna(0) > 0].copy()

    # color uses percentile
    plot_gdf["plot_percentile"] = plot_gdf[value_col].rank(pct=True)

    # size uses actual value, following original notebook
    plot_gdf["plot_size"] = plot_gdf[value_col] * marker_size_factor
    plot_gdf["plot_size"] = plot_gdf["plot_size"].clip(lower=marker_size_min)

    if marker_size_max is not None:
        plot_gdf["plot_size"] = plot_gdf["plot_size"].clip(upper=marker_size_max)

    states_sub.boundary.plot(
        ax=ax,
        color="grey",
        linewidth=0.15,
        zorder=1,
    )

    plot_gdf.plot(
        column="plot_percentile",
        ax=ax,
        cmap="hot_r",
        edgecolor="white",
        linewidth=0.1,
        markersize=plot_gdf["plot_size"],
        alpha=0.5,
        vmin=0,
        vmax=1,
        zorder=2,
    )

    ax.set_xlim((-125, -66))
    ax.set_ylim((24, 50))
    ax.set_axis_off()

    sm = plt.cm.ScalarMappable(
        cmap="hot_r",
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    sm._A = []

    cax = ax.figure.add_axes(cbar_position)
    cbar = ax.figure.colorbar(sm, cax=cax, orientation="vertical")

    tick_percentiles = [0, 0.5, 1.0]
    actual_values = np.percentile(
        plot_gdf[value_col].dropna(),
        [0, 50, 100],
    )

    cbar.set_ticks(tick_percentiles)
    cbar.set_ticklabels([f"{v:.2f}" for v in actual_values])
    cbar.ax.tick_params(labelsize=10, length=0)
    cbar.outline.set_edgecolor("white")
    cbar.set_label(unit_label, fontsize=14, labelpad=10, fontname="Arial")


def plot_accumulation_lines(ax, landfill_accum):
    years = list(range(START_YEAR, END_YEAR + 1))
    landfill_accum_sub = landfill_accum[YEARS]

    highlighted_accum = {
        701: "Warren County District Landfill, NJ",
        693: "Middlesex County LF, NJ",
        1325: "Lansing Landfill, IL",
    }

    highlight_colors_accum = {
        701: "red",
        693: "blue",
        1325: "green",
    }

    for index, row in landfill_accum_sub.iterrows():
        row = row.astype(float)

        if index in highlighted_accum:
            color = highlight_colors_accum[index]
            ax.plot(years, row, linewidth=2.3, color=color)

            if index == 701:
                ax.annotate(
                    highlighted_accum[index],
                    xy=(years[-1], row.iloc[-1]),
                    xytext=(2034, row.iloc[-1] + 300),
                    fontsize=14,
                    color=color,
                    fontname="Arial",
                )
            elif index == 693:
                ax.annotate(
                    highlighted_accum[index],
                    xy=(years[-1], row.iloc[-1]),
                    xytext=(2031, row.iloc[-1] - 4700),
                    fontsize=14,
                    color=color,
                    fontname="Arial",
                )
            elif index == 1325:
                ax.annotate(
                    highlighted_accum[index],
                    xy=(years[-1], row.iloc[-1]),
                    xytext=(2031, row.iloc[-1] - 5200),
                    fontsize=14,
                    color=color,
                    fontname="Arial",
                )
        else:
            ax.plot(years, row, color="grey", linewidth=0.05, alpha=0.35)

    style_axis(ax)
    ax.set_ylabel("PFAS accumulation in landfills (kg)", fontsize=12, fontname="Arial")
    ax.set_xlim([START_YEAR, END_YEAR])
    ax.tick_params(labelsize=10)


def plot_leachate_lines(ax, landfill_leachate):
    years = list(range(START_YEAR, END_YEAR + 1))
    leachate_sub = landfill_leachate[YEARS]

    highlighted_leachate = {
        701: "Warren County District Landfill, NJ",
        1138: "Cedar Hills Regional Landfill, WA",
        1310: "Braintree Landfill, MA"
    }

    highlight_colors_leachate = {
        701: "red",
        1138: "blue",
        1310: "green",
    }

    for index, row in leachate_sub.iterrows():
        row = row.astype(float)

        if index in highlighted_leachate:
            color = highlight_colors_leachate[index]
            ax.plot(years, row, linewidth=2.3, color=color)

            if index == 701:
                ax.annotate(
                    highlighted_leachate[index],
                    xy=(years[-1], row.iloc[-1]),
                    xytext=(2016, row.iloc[-1] - 12),
                    fontsize=14,
                    color=color,
                    fontname="Arial",
                )
            elif index == 1138:
                ax.annotate(
                    highlighted_leachate[index],
                    xy=(years[-1], row.iloc[-1]),
                    xytext=(2001, row.iloc[-1] - 5),
                    fontsize=14,
                    color=color,
                    fontname="Arial",
                )
            elif index == 1310:
                ax.annotate(
                    highlighted_leachate[index],
                    xy=(years[-1], row.iloc[-1]),
                    xytext=(2018, row.iloc[-1] + 3),
                    fontsize=14,
                    color=color,
                    fontname="Arial",
                )
        else:
            ax.plot(years, row, color="grey", linewidth=0.05, alpha=0.35)

    style_axis(ax)
    ax.set_ylabel("PFAS in landfill leachate (kg/year)", fontsize=12, fontname="Arial")
    ax.set_xlim([START_YEAR, END_YEAR])
    ax.tick_params(labelsize=10)


def make_landfill_figure(project_root: Path):
    input_dir = (
        project_root
        / "outputs"
        / "tables"
        / "landfill_central_BAU"
    )

    accum_path = input_dir / "landfill_accum_without_leachate_central_BAU_MLF.csv"
    leachate_path = input_dir / "landfill_leachate_total_central_BAU_MLF.csv"

    output_dir = project_root / "outputs" / "figures" / "landfill"
    output_dir.mkdir(parents=True, exist_ok=True)

    state_shp_path = find_state_shapefile(project_root)

    states = gpd.read_file(state_shp_path)
    states_sub = states[
        ~states["NAME"].isin(["Alaska", "Hawaii", "Puerto Rico"])
    ].copy()

    landfill_accum = prepare_landfill_gdf(accum_path).to_crs(states_sub.crs)
    landfill_leachate = prepare_landfill_gdf(leachate_path).to_crs(states_sub.crs)

    fig = plt.figure(figsize=(18, 10))

    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[2.8, 1.2],
        height_ratios=[1, 1],
        wspace=0.08,
        hspace=0.18,
    )

    ax_a = plt.subplot(gs[0, 0])
    ax_b = plt.subplot(gs[0, 1])
    ax_c = plt.subplot(gs[1, 0])
    ax_d = plt.subplot(gs[1, 1])

    plot_landfill_map_percentile(
        ax=ax_a,
        gdf=landfill_accum,
        states_sub=states_sub,
        value_col="2020",
        unit_label=r"$kg$",
        cbar_position=[0.55, 0.61, 0.012, 0.20],
        marker_size_factor=1 / 15,
        marker_size_min=0.5,
        marker_size_max=900,
    )

    plot_accumulation_lines(
        ax=ax_b,
        landfill_accum=landfill_accum,
    )

    plot_landfill_map_percentile(
        ax=ax_c,
        gdf=landfill_leachate,
        states_sub=states_sub,
        value_col="2020",
        unit_label=r"$kg/year$",
        cbar_position=[0.55, 0.18, 0.012, 0.20],
        marker_size_factor=8,
        marker_size_min=0.5,
        marker_size_max=900,
    )

    plot_leachate_lines(
        ax=ax_d,
        landfill_leachate=landfill_leachate,
    )

    fig.text(
        0.56, 0.89, "(a)",
        ha="center",
        fontsize=24,
        fontweight="bold",
        fontname="Arial"
    )

    fig.text(
        0.70, 0.89, "(b)",
        ha="center",
        fontsize=24,
        fontweight="bold",
        fontname="Arial"
    )

    fig.text(
        0.56, 0.46, "(c)",
        ha="center",
        fontsize=24,
        fontweight="bold",
        fontname="Arial"
    )

    fig.text(
        0.70, 0.46, "(d)",
        ha="center",
        fontsize=24,
        fontweight="bold",
        fontname="Arial"
    )

    plt.savefig(
        output_dir / "landfill_accum_leachate_central_BAU.png",
        dpi=1500,
        bbox_inches="tight",
    )

    plt.savefig(
        output_dir / "landfill_accum_leachate_central_BAU.pdf",
        bbox_inches="tight",
    )

    # plt.show()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    make_landfill_figure(project_root)