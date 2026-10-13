from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.paths import get_output_dirs

from src.config import GEOMETRIC_MEAN_PARAMS, LOG_UNIFORM_PARAMS, PFAS_CONFIGS, PFASConfig, CarpetConfig
from src.io_utils import save_csv
from src.uncertainty import (
    EMISSION_RATES,
    carpet_bounds,
    check_nonnegative,
    default_workers,
    emission_rate_bounds,
    evaluate_samples,
)


# =========================
# User-adjustable settings
# =========================

# Parameters sampled as uniform between low and high
UNIFORM_PARAMS = {
    "Prod_emi_R",
}

# Parameters sampled as log-uniform between low and high: LOG_UNIFORM_PARAMS (src/config.py)
# (the in-use emission rates k_air, k_dust are also log-uniform; see sample_between_low_high)

# The central estimates in config.py are the medians of these distributions: geometric means of low and
# high for exactly the log-uniform parameters and the in-use losses (and P_export = P_production)
if GEOMETRIC_MEAN_PARAMS != LOG_UNIFORM_PARAMS | {"P_export", *EMISSION_RATES.values()}:
    raise ValueError("Geometric-mean central parameters do not match the log-uniform Monte Carlo parameters")

# Outputs to track from total PFAS
OUTPUT_COLS = [
    "F_stock_total",
    "F6_total",
    "F_accu_landfill_total",
    "PFAS_landfill_leachate_total",
]


def sample_uniform(low: float, high: float, rng: np.random.Generator) -> float:
    lo = min(low, high)
    hi = max(low, high)
    if np.isclose(lo, hi):
        return float(lo)
    return float(rng.uniform(lo, hi))


def sample_log_uniform(low: float, high: float, rng: np.random.Generator) -> float:
    lo = min(low, high)
    hi = max(low, high)

    if lo <= 0 or hi <= 0:
        raise ValueError(f"log-uniform requires positive bounds, got low={lo}, high={hi}")

    if np.isclose(lo, hi):
        return float(lo)

    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def get_param_value(cfg: PFASConfig, param_name: str) -> Optional[float]:
    return getattr(cfg, param_name)


def sample_between_low_high(
    low_cfg: PFASConfig,
    high_cfg: PFASConfig,
    rng: np.random.Generator,
) -> PFASConfig:
    """
    Create one sampled config for a subclass using the low/high pair as bounds.
    P_export is not sampled: exported carpet carries the production PFAS content (P_export =
    P_production, as in every config estimate).
    In-use emission: the first-order rates k_air, k_dust (P_yearlyloss / P_production) are sampled
    between the low and high estimates, and P_yearlyloss = k x sampled P_production, so the rates
    stay within the range of the config estimates.
    """
    sampled_values: Dict[str, Optional[float]] = {}

    param_names = [
        "P_import",
        "P_production",
        "Prod_emi_R",
        "xt",
        "P_protector",
    ]

    for p in param_names:
        low_val = get_param_value(low_cfg, p)
        high_val = get_param_value(high_cfg, p)

        if low_val is None and high_val is None:
            sampled_values[p] = None
            continue

        if low_val is None:
            sampled_values[p] = high_val
            continue

        if high_val is None:
            sampled_values[p] = low_val
            continue

        lo = float(min(low_val, high_val))
        hi = float(max(low_val, high_val))

        # Fixed value
        if np.isclose(lo, hi):
            sampled_values[p] = lo

        elif p in UNIFORM_PARAMS:
            sampled_values[p] = sample_uniform(lo, hi, rng)

        elif p in LOG_UNIFORM_PARAMS:
            # log-uniform only for strictly positive bounds
            if lo > 0 and hi > 0:
                sampled_values[p] = sample_log_uniform(lo, hi, rng)
            else:
                sampled_values[p] = sample_uniform(lo, hi, rng)

        else:
            sampled_values[p] = sample_uniform(lo, hi, rng)

    sampled_values["P_export"] = sampled_values["P_production"]

    for rate, loss in EMISSION_RATES.items():
        lo, hi = emission_rate_bounds(low_cfg, high_cfg, rate)
        if np.isclose(lo, hi):
            k = lo
        elif lo > 0:
            k = sample_log_uniform(lo, hi, rng)
        else:
            k = sample_uniform(lo, hi, rng)
        sampled_values[loss] = k * sampled_values["P_production"]

    sampled_cfg = PFASConfig(
        subclass=high_cfg.subclass,
        estimate="mc",
        P_import=float(sampled_values["P_import"]),
        P_production=float(sampled_values["P_production"]),
        P_export=float(sampled_values["P_export"]),
        P_yearlyloss_air=float(sampled_values["P_yearlyloss_air"]),
        P_yearlyloss_dust=float(sampled_values["P_yearlyloss_dust"]),
        Prod_emi_R=float(sampled_values["Prod_emi_R"]),
        xt=sampled_values["xt"],
        P_protector=sampled_values["P_protector"],
        has_leachate=high_cfg.has_leachate,
        use_protector=high_cfg.use_protector,
    )
    return sampled_cfg


def sample_carpet_config(rng: np.random.Generator) -> CarpetConfig:
    """Carpet parameters sampled uniformly between the low and high estimates in config.py."""
    bounds = carpet_bounds()
    return CarpetConfig(
        estimate="mc",
        carpet_density=sample_uniform(*bounds["carpet_density"], rng),
        lifespan_mean=sample_uniform(*bounds["lifespan_mean"], rng),
        nylon_fmin=sample_uniform(*bounds["nylon_fmin"], rng),
        average_floor_area=sample_uniform(*bounds["average_floor_area"], rng),
        initial_stock_age=sample_uniform(*bounds["initial_stock_age"], rng),
    )


def build_sampled_subclass_configs(
    rng: np.random.Generator,
) -> Dict[str, PFASConfig]:
    """
    Sample NP, PFAA, SP configs once for a single Monte Carlo iteration.
    """
    subclasses = ["NP", "PFAA", "SP"]
    sampled = {}

    for subclass in subclasses:
        low_cfg = PFAS_CONFIGS[(subclass, "low")]
        high_cfg = PFAS_CONFIGS[(subclass, "high")]
        sampled[subclass] = sample_between_low_high(low_cfg, high_cfg, rng)

    return sampled


def run_monte_carlo_timeseries(
    project_root: Path,
    n_runs: int = 1000,
    seed: int = 42,
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Monte Carlo for BAU scenario only, national R + C totals (see src/uncertainty.py).
    All samples are drawn first in a fixed order, so results do not depend on n_workers.
    Returns long dataframe: year × outputs × iteration.
    """
    rng = np.random.default_rng(seed)

    samples = []
    for _ in range(n_runs):
        sampled_cfgs = build_sampled_subclass_configs(rng)
        sampled_carpet_config = sample_carpet_config(rng)
        samples.append((sampled_cfgs, sampled_carpet_config))

    totals = evaluate_samples(project_root, samples, n_workers=n_workers or default_workers())
    check_nonnegative(totals)

    results: List[pd.DataFrame] = []
    for i, total_df in enumerate(totals):
        out = total_df[["year"] + OUTPUT_COLS].copy()
        out["iteration"] = i
        results.append(out)

    mc_results = pd.concat(results, ignore_index=True)
    return mc_results


def summarize_monte_carlo(mc_results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annual p2.5, median, p97.5 for each target output.
    """
    # numpy quantiles (linear interpolation, as pandas' default); avoids a pandas groupby.quantile
    # "No matching signature found" error seen with the full 10,000-run table
    q_map = {
        0.025: "p2_5",
        0.5: "median",
        0.975: "p97_5",
    }
    years = np.sort(mc_results["year"].unique())
    summary = pd.DataFrame({"year": years})
    for col in OUTPUT_COLS:
        wide = mc_results.pivot(index="iteration", columns="year", values=col)[years].to_numpy(dtype=float)
        for q, name in q_map.items():
            summary[f"{col}_{name}"] = np.quantile(wide, q, axis=0)
    return summary


def plot_monte_carlo_summary(
        summary: pd.DataFrame,
        fig_path: Path,
        layout: str = "2x2",
) -> None:
    plt.rcParams["font.family"] = "Arial"

    if layout == "2x2":
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()

    elif layout == "1x4":
        fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharex=True)

    else:
        raise ValueError(f"Unknown layout: {layout}")

    plot_info = [
        ("F_stock_total", "tonnes", "(a) In-use stock"),
        ("F6_total", "tonnes/year", "(b) In-use emission"),
        ("F_accu_landfill_total", "tonnes", "(c) Landfill accumulation"),
        ("PFAS_landfill_leachate_total", "tonnes/year", "(d) Landfill leachate"),
    ]

    for ax, (var, ylabel, title) in zip(axes, plot_info):
        x = summary["year"]
        # Convert kg to tonnes only for plotting
        y_med = summary[f"{var}_median"] / 1000
        y_low = summary[f"{var}_p2_5"] / 1000
        y_high = summary[f"{var}_p97_5"] / 1000

        ax.plot(x, y_med, linewidth=1.8, label="Median")
        ax.fill_between(x, y_low, y_high, alpha=0.3, label="95% uncertainty interval")

        ax.set_title(title, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(True, alpha=0.3)

    if layout == "2x2":
        axes[2].set_xlabel("Year", fontsize=12)
        axes[3].set_xlabel("Year", fontsize=12)
    else:
        for ax in axes:
            ax.set_xlabel("Year", fontsize=12)
    axes[0].legend(frameon=False, fontsize=12)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close()


def run_and_save_monte_carlo(
    project_root: Path,
    n_runs: int = 1000,
    seed: int = 42,
    n_workers: Optional[int] = None,
) -> None:
    dirs = get_output_dirs(project_root)
    output_dir = dirs["tables_monte_carlo"]
    output_dir.mkdir(parents=True, exist_ok=True)

    mc_results = run_monte_carlo_timeseries(
        project_root=project_root,
        n_runs=n_runs,
        seed=seed,
        n_workers=n_workers,
    )

    save_csv(mc_results, output_dir / "MC_BAU_timeseries_results.csv")  # raw results first

    summary = summarize_monte_carlo(mc_results)
    save_csv(summary, output_dir / "MC_BAU_timeseries_summary.csv")
    # Original 2×2 figure
    plot_monte_carlo_summary(
        summary,
        output_dir / "MC_BAU_uncertainty_plot_2x2.png",
        layout="2x2",
    )

    # New 1×4 figure
    plot_monte_carlo_summary(
        summary,
        output_dir / "MC_BAU_uncertainty_plot_1x4.png",
        layout="1x4",
    )
    print("Saved: MC_BAU_timeseries_results.csv")
    print("Saved: MC_BAU_timeseries_summary.csv")
    print("Saved: MC_BAU_uncertainty_plot.png")