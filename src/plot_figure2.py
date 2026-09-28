from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


STACK_COLORS = {
    "SP": "#CC6666",
    "NP": "#1DACD6",
    "PFAA": "#6E5160",
}

RC_COLORS = {
    "R": "#8C9A5B",
    "C": "#E0A43A",
}

RC_LABELS = {
    "R": "Residential",
    "C": "Commercial",
}

SCENARIO_COLORS = {
    "BAU": "black",
    "RECYCLE_24": "blue",
    "PHASE_OUT": "red",
}

SCENARIO_LABELS = {
    "BAU": "Business-as-usual",
    "RECYCLE_24": "24% recycling",
    "PHASE_OUT": "Phase-out",
}

SCENARIO_LINESTYLES = {
    "BAU": "-",
    "RECYCLE_24": "--",
    "PHASE_OUT": "--",
}

SCENARIO_START_YEAR = 2020
UNIT_DIVISOR = 1000


def _standardize_year_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "year" in df.columns:
        return df
    if "Year" in df.columns:
        return df.rename(columns={"Year": "year"})
    raise ValueError("No year/Year column found in file.")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    print(f"Loading: {path}")
    df = pd.read_csv(path)
    return _standardize_year_column(df)


def _load_subclass_metric(
    tables_dir: Path,
    subclass: str,
    estimate: str,
    scenario: str,
    metric: str,
) -> pd.DataFrame:
    file_path = tables_dir / f"{subclass}_{estimate.upper()}_{scenario}.csv"
    df = _load_csv(file_path)

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in {file_path.name}")

    return df[["year", metric]].copy()


def _load_total_metric(
    tables_dir: Path,
    estimate: str,
    scenario: str,
    metric: str,
    suffix: str = "",
) -> pd.DataFrame:
    file_path = tables_dir / f"TOTAL_{estimate.upper()}_ALL_SUBCLASSES_{scenario}{suffix}.csv"
    df = _load_csv(file_path)

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in {file_path.name}")

    return df[["year", metric]].copy()


def _build_stacked_df(
    tables_dir: Path,
    estimate: str,
    scenario: str,
    metric: str,
) -> pd.DataFrame:
    sp = _load_subclass_metric(tables_dir, "SP", estimate, scenario, metric).rename(columns={metric: "SP"})
    np_df = _load_subclass_metric(tables_dir, "NP", estimate, scenario, metric).rename(columns={metric: "NP"})
    pfaa = _load_subclass_metric(tables_dir, "PFAA", estimate, scenario, metric).rename(columns={metric: "PFAA"})

    df = sp.merge(np_df, on="year").merge(pfaa, on="year")

    for col in ["SP", "NP", "PFAA"]:
        df[col] = df[col] / UNIT_DIVISOR

    return df


def _build_scenario_df(
    tables_dir: Path,
    estimate: str,
    metric: str,
) -> pd.DataFrame:
    bau = _load_total_metric(tables_dir, estimate, "BAU", metric).rename(columns={metric: "BAU"})
    recycle = _load_total_metric(tables_dir, estimate, "RECYCLE_24", metric).rename(columns={metric: "RECYCLE_24"})
    phase = _load_total_metric(tables_dir, estimate, "PHASE_OUT", metric).rename(columns={metric: "PHASE_OUT"})

    df = bau.merge(recycle, on="year").merge(phase, on="year")

    for col in ["BAU", "RECYCLE_24", "PHASE_OUT"]:
        df[col] = df[col] / UNIT_DIVISOR

    return df


def _build_rc_df(
    tables_dir: Path,
    estimate: str,
    scenario: str,
    metric: str,
) -> pd.DataFrame:
    res = _load_total_metric(tables_dir, estimate, scenario, metric, suffix="_R").rename(columns={metric: "R"})
    com = _load_total_metric(tables_dir, estimate, scenario, metric, suffix="_C").rename(columns={metric: "C"})

    df = res.merge(com, on="year")

    for col in ["R", "C"]:
        df[col] = df[col] / UNIT_DIVISOR

    return df


def _set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 18,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def _format_axis(ax, ylabel: str, panel_label: str, show_xlabel: bool = False, label_right: bool = False) -> None:
    ax.set_xlim(1990, 2060)
    ax.set_ylim(bottom=0)
    ax.set_xticks([2000, 2020, 2040, 2060])
    ax.set_ylabel(ylabel)

    if show_xlabel:
        ax.set_xlabel("Year")

    ax.grid(False)
    ax.tick_params(axis="both", which="major", width=1.0, length=4)

    ax.text(
        0.98 if label_right else 0.02,
        0.96,
        panel_label,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        va="top",
        ha="right" if label_right else "left",
    )


def plot_figure2(
    tables_dir: Path,
    figure_path: Path,
    estimate: str = "CENTRAL",
) -> None:
    print(f"Start plotting Figure 2 for estimate={estimate}")
    print(f"Tables dir: {tables_dir}")
    print(f"Figure path: {figure_path}")

    _set_plot_style()

    metric_map = [
        ("F_stock", "In-use Stock \n(tonnes)"),
        ("F6", "In-use Emission \n(tonnes/year)"),
        ("F_accu_landfill", "Landfill Accumulation \n(tonnes)"),
        ("PFAS_landfill_leachate", "Landfill Leachate \n(tonnes/year)"),
    ]

    panel_labels_rc = ["(a)", "(b)", "(c)", "(d)"]
    panel_labels_top = ["(e)", "(f)", "(g)", "(h)"]
    panel_labels_bottom = ["(i)", "(j)", "(k)", "(l)"]

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(25, 15),
        sharex=True,
        constrained_layout=False,
    )

    # Top row: stacked residential / commercial plots under BAU
    rc_totals = {}
    for j, (metric, ylabel) in enumerate(metric_map):
        ax = axes[0, j]

        df = _build_rc_df(
            tables_dir=tables_dir,
            estimate=estimate,
            scenario="BAU",
            metric=metric,
        )
        rc_totals[metric] = df.set_index("year")["R"] + df.set_index("year")["C"]

        ax.stackplot(
            df["year"],
            df["R"],
            df["C"],
            colors=[RC_COLORS["R"], RC_COLORS["C"]],
            alpha=0.5,
        )

        _format_axis(ax, ylabel, panel_labels_rc[j], show_xlabel=False, label_right=(j in (0, 1)))

    # Middle row: stacked subclass plots under BAU
    for j, (metric, ylabel) in enumerate(metric_map):
        ax = axes[1, j]

        df = _build_stacked_df(
            tables_dir=tables_dir,
            estimate=estimate,
            scenario="BAU",
            metric=metric,
        )

        ax.stackplot(
            df["year"],
            df["SP"],
            df["NP"],
            df["PFAA"],
            colors=[STACK_COLORS["SP"], STACK_COLORS["NP"], STACK_COLORS["PFAA"]],
            alpha=0.5,
        )

        _format_axis(ax, ylabel, panel_labels_top[j], show_xlabel=False, label_right=(j in (0, 1)))

        # Residential + commercial and SP + NP + PFAA are the same total: same y-axis
        subclass_total = df.set_index("year")[["SP", "NP", "PFAA"]].sum(axis=1)
        max_diff = (rc_totals[metric] - subclass_total).abs().max() / subclass_total.abs().max()
        print(f"{metric}: max |(R + C) - (SP + NP + PFAA)| / max total = {max_diff:.2e}")
        if max_diff > 1e-9:
            raise ValueError(f"Residential + commercial total does not match subclass total for {metric}")
        axes[0, j].set_ylim(ax.get_ylim())

    # Bottom row: scenario comparison
    for j, (metric, ylabel) in enumerate(metric_map):
        ax = axes[2, j]

        df = _build_scenario_df(
            tables_dir=tables_dir,
            estimate=estimate,
            metric=metric,
        )

        ax.plot(
            df["year"],
            df["BAU"],
            color=SCENARIO_COLORS["BAU"],
            linestyle=SCENARIO_LINESTYLES["BAU"],
            linewidth=4.5,
            label=SCENARIO_LABELS["BAU"],
        )

        mask = df["year"] >= SCENARIO_START_YEAR

        ax.plot(
            df.loc[mask, "year"],
            df.loc[mask, "RECYCLE_24"],
            color=SCENARIO_COLORS["RECYCLE_24"],
            linestyle=SCENARIO_LINESTYLES["RECYCLE_24"],
            linewidth=4.5,
            label=SCENARIO_LABELS["RECYCLE_24"],
        )

        ax.plot(
            df.loc[mask, "year"],
            df.loc[mask, "PHASE_OUT"],
            color=SCENARIO_COLORS["PHASE_OUT"],
            linestyle=SCENARIO_LINESTYLES["PHASE_OUT"],
            linewidth=4.5,
            label=SCENARIO_LABELS["PHASE_OUT"],
        )

        _format_axis(ax, ylabel, panel_labels_bottom[j], show_xlabel=False, label_right=(j in (0, 1)))

    scenario_handles = [
        Line2D([0], [0], color=SCENARIO_COLORS["BAU"], lw=6, linestyle="-", label=SCENARIO_LABELS["BAU"]),
        Line2D([0], [0], color=SCENARIO_COLORS["RECYCLE_24"], lw=6, linestyle="--", label=SCENARIO_LABELS["RECYCLE_24"]),
        Line2D([0], [0], color=SCENARIO_COLORS["PHASE_OUT"], lw=6, linestyle="--", label=SCENARIO_LABELS["PHASE_OUT"]),
    ]

    stack_handles = [
        Patch(facecolor=STACK_COLORS["SP"], edgecolor="none", alpha=0.5, label="SP"),
        Patch(facecolor=STACK_COLORS["NP"], edgecolor="none", alpha=0.5, label="NP"),
        Patch(facecolor=STACK_COLORS["PFAA"], edgecolor="none", alpha=0.5, label="PFAA"),
    ]

    rc_handles = [
        Patch(facecolor=RC_COLORS["R"], edgecolor="none", alpha=0.5, label=RC_LABELS["R"]),
        Patch(facecolor=RC_COLORS["C"], edgecolor="none", alpha=0.5, label=RC_LABELS["C"]),
    ]

    fig.legend(
        handles=rc_handles + stack_handles + scenario_handles,
        loc="lower center",
        ncol=8,
        frameon=True,
        bbox_to_anchor=(0.5, 0.01),
        handlelength=2.6,
        columnspacing=1.6,
    )

    plt.subplots_adjust(
        left=0.08,
        right=0.99,
        top=0.96,
        bottom=0.12,
        wspace=0.35,
        hspace=0.28,
    )

    figure_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving figure to: {figure_path}")
    plt.savefig(figure_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Finished plotting: {figure_path.name}")