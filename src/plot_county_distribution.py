from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


YEAR_START = 1990
YEAR_END = 2060


def load_county_shape(project_root: Path) -> gpd.GeoDataFrame:
    shp = (
        project_root
        / "data"
        / "county"
        / "tl_2020_us_county"
        / "tl_2020_us_county.shp"
    )

    gdf = gpd.read_file(shp)

    # Remove Alaska, Hawaii, Puerto Rico, and territories
    gdf = gdf[
        ~gdf["STATEFP"].isin(
            ["02", "15", "72", "60", "66", "69", "78"]
        )
    ].copy()

    gdf["GeoID"] = gdf["GEOID"].astype(str).str.zfill(5)

    state_abbrev = {
        "01": "AL", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
        "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA",
        "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS",
        "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA",
        "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
        "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
        "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK",
        "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
        "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA",
        "53": "WA", "54": "WV", "55": "WI", "56": "WY",
    }

    gdf["County"] = gdf["NAME"]
    gdf["State"] = gdf["STATEFP"].map(state_abbrev)

    # Contiguous U.S. Albers projection
    gdf = gdf.to_crs("EPSG:5070")

    return gdf


def wide_to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    id_cols = [c for c in ["GeoID", "County", "State"] if c in df.columns]
    year_cols = [c for c in df.columns if c.isdigit()]

    out = df.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="year",
        value_name=value_name,
    )

    out["year"] = out["year"].astype(int)
    return out


def plot_county_map_percentile(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    pct_col: str,
    ax,
    cmap: str,
    cbar_label: str,
    panel_label: str,
) -> None:
    values = gdf[value_col].dropna()

    quantiles = values.quantile([0, 0.25, 0.5, 0.75, 1])

    def format_tick_value(v: float) -> str:
        if abs(v) >= 100:
            return f"{v:.0f}"
        elif abs(v) >= 10:
            return f"{v:.1f}"
        else:
            return f"{v:.2f}"

    gdf.plot(
        column=pct_col,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidth=0.05,
        edgecolor="white",
        legend=False,
        missing_kwds={"color": "lightgrey"},
    )

    ax.set_axis_off()

    ax.text(
        0.95,
        0.95,
        panel_label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
    )

    # Colorbar based on percentile rank, labeled with raw values
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    sm.set_array([])

    bbox = ax.get_position()

    cax = ax.figure.add_axes(
        [
            bbox.x1 - 0.018,
            bbox.y0 + bbox.height * 0.25,
            0.012,
            bbox.height * 0.45,
        ]
    )

    cbar = ax.figure.colorbar(
        sm,
        cax=cax,
        orientation="vertical",
    )

    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    cbar.set_ticklabels(
        [format_tick_value(v) for v in quantiles]
    )

    cbar.set_label(cbar_label, fontsize=10, labelpad=6)
    cbar.ax.tick_params(labelsize=8)


def plot_county_figure3(project_root: Path) -> None:
    county_dir = project_root / "outputs" / "tables" / "county"
    fig_dir = project_root / "outputs" / "figures" / "county"
    fig_dir.mkdir(parents=True, exist_ok=True)

    stock = pd.read_csv(
        county_dir / "County_InUseStock_CENTRAL_BAU_kg.csv"
    )

    pc_emission = pd.read_csv(
        county_dir / "County_PC_InUseEmission_CENTRAL_BAU_mg_cap_year.csv"
    )

    # Keep GeoID as 5-digit FIPS
    stock["GeoID"] = stock["GeoID"].astype(str).str.zfill(5)
    pc_emission["GeoID"] = pc_emission["GeoID"].astype(str).str.zfill(5)

    gdf = load_county_shape(project_root)

    stock_long = wide_to_long(stock, "stock_kg")
    pcemi_long = wide_to_long(
        pc_emission,
        "pc_emission_mg_cap_year",
    )

    # 2020 map data
    stock_2020 = stock_long[stock_long["year"] == 2020]
    pcemi_2020 = pcemi_long[pcemi_long["year"] == 2020]

    g_stock = gdf.merge(stock_2020, on="GeoID", how="left")
    g_pcemi = gdf.merge(pcemi_2020, on="GeoID", how="left")

    # Percentile rank mapping, following the original Figure 3 logic
    g_stock["stock_kg_pct"] = g_stock["stock_kg"].rank(pct=True)

    g_pcemi["pc_emission_mg_cap_year_pct"] = (
        g_pcemi["pc_emission_mg_cap_year"].rank(pct=True)
    )

    county_names = gdf[["GeoID", "County", "State"]].copy()

    # Top counties for time-series highlight
    stock_top_full = (
        stock_long[stock_long["year"] == 2060]
        .merge(county_names, on="GeoID", how="left")
        .sort_values("stock_kg", ascending=False)
        .head(3)[["GeoID", "County", "State", "stock_kg"]]
    )
    stock_top = stock_top_full[["GeoID", "County", "State"]]

    pcemi_top_full = (
        pcemi_long[pcemi_long["year"] == 2060]
        .merge(county_names, on="GeoID", how="left")
        .sort_values("pc_emission_mg_cap_year", ascending=False)
        .head(3)[["GeoID", "County", "State", "pc_emission_mg_cap_year"]]
    )
    pcemi_top = pcemi_top_full[["GeoID", "County", "State"]]

    print("Top 3 counties by in-use stock (2060):")
    for _, row in stock_top_full.iterrows():
        print(
            f"  {row['GeoID']}  {row['County']}, {row['State']}  "
            f"stock_kg={row['stock_kg']:.4f}"
        )

    print("Top 3 counties by per-capita in-use emission (2060):")
    for _, row in pcemi_top_full.iterrows():
        print(
            f"  {row['GeoID']}  {row['County']}, {row['State']}  "
            f"pc_emission_mg_cap_year={row['pc_emission_mg_cap_year']:.4f}"
        )

    plt.rcParams["font.family"] = "Arial"

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_axes([0.04, 0.55, 0.48, 0.38])
    ax2 = fig.add_axes([0.60, 0.56, 0.35, 0.35])
    ax3 = fig.add_axes([0.04, 0.08, 0.48, 0.38])
    ax4 = fig.add_axes([0.60, 0.09, 0.35, 0.35])

    cmap = "OrRd"

    # (a) County in-use stock, 2020
    plot_county_map_percentile(
        gdf=g_stock,
        value_col="stock_kg",
        pct_col="stock_kg_pct",
        ax=ax1,
        cmap=cmap,
        cbar_label="kg",
        panel_label="(a)",
    )

    # (b) County in-use stock time series
    for geoid in stock_long["GeoID"].unique():
        sub = stock_long[stock_long["GeoID"] == geoid]
        ax2.plot(
            sub["year"],
            sub["stock_kg"],
            color="lightgrey",
            alpha=0.22,
            linewidth=0.5,
        )

    for _, row in stock_top.iterrows():
        sub = stock_long[stock_long["GeoID"] == row["GeoID"]]
        label = f"{row['County']}, {row['State']}"
        ax2.plot(
            sub["year"],
            sub["stock_kg"],
            linewidth=2.2,
            label=label,
        )

    ax2.set_ylabel("In-use stock (kg)")
    ax2.set_xlim(YEAR_START, YEAR_END)
    ax2.legend(frameon=False, fontsize=8)
    ax2.text(
        0.02,
        0.95,
        "(b)",
        transform=ax2.transAxes,
        fontsize=16,
        fontweight="bold",
    )

    # (c) County per capita in-use emission, 2020
    plot_county_map_percentile(
        gdf=g_pcemi,
        value_col="pc_emission_mg_cap_year",
        pct_col="pc_emission_mg_cap_year_pct",
        ax=ax3,
        cmap=cmap,
        cbar_label="mg/cap/year",
        panel_label="(c)",
    )

    # (d) County per capita in-use emission time series
    for geoid in pcemi_long["GeoID"].unique():
        sub = pcemi_long[pcemi_long["GeoID"] == geoid]
        ax4.plot(
            sub["year"],
            sub["pc_emission_mg_cap_year"],
            color="lightgrey",
            alpha=0.22,
            linewidth=0.5,
        )

    for _, row in pcemi_top.iterrows():
        sub = pcemi_long[pcemi_long["GeoID"] == row["GeoID"]]
        label = f"{row['County']}, {row['State']}"
        ax4.plot(
            sub["year"],
            sub["pc_emission_mg_cap_year"],
            linewidth=2.2,
            label=label,
        )

    ax4.set_ylabel("Per capita in-use emission (mg/cap/year)")
    ax4.set_xlim(YEAR_START, YEAR_END)
    ax4.set_ylim(0, 30)
    ax4.legend(frameon=False, fontsize=8)
    ax4.text(
        0.02,
        0.95,
        "(d)",
        transform=ax4.transAxes,
        fontsize=16,
        fontweight="bold",
    )

    fig_path_png = (
        fig_dir
        / "Figure_3_county_distribution_CENTRAL_BAU_percentile.png"
    )

    plt.savefig(fig_path_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {fig_path_png}")