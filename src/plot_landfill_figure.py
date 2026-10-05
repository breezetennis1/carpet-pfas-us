import re
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.lines import Line2D

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


# Five classes with round breaks that split the high end (2020 values)
ACCUM_BREAKS = [100, 500, 2000, 5000]       # kg
LEACHATE_BREAKS = [0.1, 0.5, 2, 5]          # kg/yr
CLASS_SIZES = [3, 10, 30, 70, 140]          # marker area (points^2), smallest to largest class
CLASS_COLORS = plt.get_cmap("hot_r")([0.25, 0.42, 0.58, 0.74, 0.9])

MAP_CRS = "EPSG:5070"                       # Contiguous US Albers equal area (as in Figure 3)
NON_CONTIGUOUS_STATEFP = ["02", "15", "60", "66", "69", "72", "78"]
MAP_MARGIN = 0.02                           # fraction of the map extent on each side
N_TOP_MAP = 3                               # landfills outlined in black on each map (largest values)


def format_value(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 10 else f"{v:g}"


def plot_landfill_map_classes(ax, gdf, states_sub, value_col: str, legend_title: str, breaks: list) -> None:
    """
    Landfills as circles in 5 size/color classes; the N_TOP_MAP largest are outlined in black.
    The size legend sits just outside the map on the right.
    """
    plot_gdf = gdf[gdf[value_col].fillna(0) > 0].copy()
    plot_gdf["value_class"] = np.digitize(plot_gdf[value_col], breaks)

    states_sub.boundary.plot(ax=ax, color="grey", linewidth=0.15, zorder=1)

    for k in range(len(CLASS_SIZES)):
        sub = plot_gdf[plot_gdf["value_class"] == k]
        ax.scatter(sub.geometry.x, sub.geometry.y, s=CLASS_SIZES[k], color=CLASS_COLORS[k],
                   edgecolor="white", linewidth=0.1, alpha=0.7, zorder=2 + k)

    top = plot_gdf.sort_values(value_col, ascending=False).head(N_TOP_MAP)
    ax.scatter(top.geometry.x, top.geometry.y, s=[CLASS_SIZES[k] for k in top["value_class"]],
               facecolor="none", edgecolor="black", linewidth=1.2, zorder=10)

    xmin, ymin, xmax, ymax = states_sub.total_bounds
    dx, dy = xmax - xmin, ymax - ymin
    ax.set_xlim(xmin - MAP_MARGIN * dx, xmax + MAP_MARGIN * dx)
    ax.set_ylim(ymin - MAP_MARGIN * dy, ymax + MAP_MARGIN * dy)
    ax.set_aspect("equal")
    ax.set_axis_off()

    labels = [f"< {format_value(breaks[0])}"]
    labels += [f"{format_value(lo)}–{format_value(hi)}" for lo, hi in zip(breaks, breaks[1:])]
    labels += [f"> {format_value(breaks[-1])}"]
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=np.sqrt(size), markerfacecolor=color,
               markeredgecolor="white", markeredgewidth=0.3, alpha=0.9)
        for size, color in zip(CLASS_SIZES, CLASS_COLORS)
    ]
    handles.append(Line2D([0], [0], marker="o", linestyle="none", markersize=np.sqrt(CLASS_SIZES[-1]),
                          markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.2))
    labels.append(f"Top {N_TOP_MAP} landfills ({value_col})")
    legend = ax.legend(handles, labels, title=legend_title, loc="lower left", bbox_to_anchor=(1.0, 0.0),
                       frameon=False, fontsize=10, title_fontsize=10.5, labelspacing=0.45, borderpad=0.1,
                       handletextpad=0.5)
    legend._legend_box.align = "left"
    legend.set_zorder(20)

    counts = plot_gdf["value_class"].value_counts().sort_index().tolist()
    print(f"Map classes ({legend_title}): breaks {breaks}, landfills per class {counts}")
    names = top["Landfill Name"] if "Landfill Name" in top else top.index
    print(f"  outlined top {N_TOP_MAP} ({value_col}): " + "; ".join(
        f"{n} = {v:.3g}" for n, v in zip(names, top[value_col])))


STATE_ABBREV = {
    1: "AL", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE", 11: "DC", 12: "FL", 13: "GA",
    16: "ID", 17: "IL", 18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV", 33: "NH", 34: "NJ", 35: "NM",
    36: "NY", 37: "NC", 38: "ND", 39: "OH", 40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD",
    47: "TN", 48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI", 56: "WY",
}

# Colors assigned per landfill (same landfill, same color in both panels), in order of first appearance
HIGHLIGHT_COLORS = ["red", "blue", "green", "purple", "darkorange", "saddlebrown"]
HIGHLIGHT_YEAR = "2060"
N_HIGHLIGHT = 3
LEGEND_HEADROOM = 1.35  # y-axis upper limit = max x headroom, leaving space for the legend


def landfill_label(row) -> str:
    name = re.sub(r"\bLF\b", "Landfill", str(row["Landfill Name"]).strip())
    state = STATE_ABBREV.get(int(row["STATEFP"]), "") if pd.notna(row.get("STATEFP")) else ""
    return f"{name}, {state}" if state else name


def top_named_landfills(gdf, year: str = HIGHLIGHT_YEAR, n: int = N_HIGHLIGHT):
    """The n landfills with the highest value in `year`, skipping landfills without a name."""
    names = gdf["Landfill Name"].astype(str).str.strip()
    named = gdf[gdf["Landfill Name"].notna() & ~names.isin(["", "<Null>", "nan"])]
    return named.sort_values(year, ascending=False).head(n)


def highlight_colors(*tops) -> dict:
    """One color per highlighted facility_id, shared across panels."""
    colors = {}
    for top in tops:
        for fid in top["facility_id"]:
            if fid not in colors:
                colors[fid] = HIGHLIGHT_COLORS[len(colors) % len(HIGHLIGHT_COLORS)]
    return colors


def plot_lines_with_top(ax, gdf, ylabel: str, top, colors: dict):
    """Grey line per landfill; the top named landfills in 2060 highlighted, named in a legend."""
    years = list(range(START_YEAR, END_YEAR + 1))

    for idx, row in gdf[YEARS].iterrows():
        if idx not in top.index:
            ax.plot(years, row.astype(float), color="grey", linewidth=0.05, alpha=0.35)

    for idx, row in top.iterrows():
        ax.plot(years, row[YEARS].astype(float), linewidth=2.3, color=colors[row["facility_id"]],
                label=landfill_label(row))

    style_axis(ax)
    ax.set_ylabel(ylabel, fontsize=12, fontname="Arial")
    ax.set_xlim([START_YEAR, END_YEAR])
    ax.set_ylim(0, float(gdf[YEARS].to_numpy().max()) * LEGEND_HEADROOM)
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=11, handlelength=1.6)

    print(f"Highlighted ({ylabel}, {HIGHLIGHT_YEAR}): "
          + "; ".join(f"{landfill_label(r)} = {r[HIGHLIGHT_YEAR]:.2f}" for _, r in top.iterrows()))


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
    states_sub = states[~states["STATEFP"].isin(NON_CONTIGUOUS_STATEFP)].to_crs(MAP_CRS)

    landfill_accum = prepare_landfill_gdf(accum_path).to_crs(MAP_CRS)
    landfill_leachate = prepare_landfill_gdf(leachate_path).to_crs(MAP_CRS)

    top_accum = top_named_landfills(landfill_accum)
    top_leachate = top_named_landfills(landfill_leachate)
    colors = highlight_colors(top_accum, top_leachate)

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

    plot_landfill_map_classes(ax_a, landfill_accum, states_sub, "2020", "PFAS accumulation\n2020 (kg)", ACCUM_BREAKS)

    plot_lines_with_top(ax_b, landfill_accum, "PFAS accumulation in landfills (kg)", top_accum, colors)

    plot_landfill_map_classes(ax_c, landfill_leachate, states_sub, "2020", "PFAS in leachate\n2020 (kg/year)", LEACHATE_BREAKS)

    plot_lines_with_top(ax_d, landfill_leachate, "PFAS in landfill leachate (kg/year)", top_leachate, colors)

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
        dpi=600,
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