from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from src.config import CARPET_CONFIGS, PFAS_CONFIGS, PFASConfig, CarpetConfig
from src.io_utils import save_csv
from src.paths import get_output_dirs
from src.uncertainty import (
    EMISSION_RATES,
    carpet_bounds,
    check_nonnegative,
    default_workers,
    emission_rate_bounds,
    evaluate_samples,
)


# =========================
# 1. User settings
# =========================
YEARS_TO_ANALYZE = [2020, 2040, 2060]
SOBOL_SEED = 42  # fixed seed for the scrambled Sobol sequence (reproducible sample matrix)
OUTPUT_ORDER = ["stock", "emission", "landfill_accu", "landfill_emission"]
TOP_N_EVOLUTION = 5

OUTPUT_MAP = {
    "stock": "F_stock_total",
    "emission": "F6_total",
    "landfill_accu": "F_accu_landfill_total",
    "landfill_emission": "PFAS_landfill_leachate_total",
}

OUTPUT_TITLES = {
    "stock": "(a) In-use stock",
    "emission": "(b) In-use emission",
    "landfill_accu": "(c) Landfill accumulation",
    "landfill_emission": "(d) Landfill leachate",
}


# Parameters to include in Sobol
# You can remove any parameter here if you do not want it in the analysis.
# P_export is not a separate parameter: it is set equal to P_production (as in every config estimate).
# In-use emission enters through the first-order rates k_air, k_dust (P_yearlyloss / P_production), with
# P_yearlyloss = k x P_production, so the rates stay within the range of the config estimates.
PARAMS_BY_SUBCLASS = {
    "NP": [
        "P_import",
        "P_production",
        "k_air",
        "k_dust",
        "Prod_emi_R",
        "xt",
    ],
    "PFAA": [
        "P_import",
        "P_production",
        "k_air",
        "k_dust",
        "Prod_emi_R",
        "xt",
    ],
    "SP": [
        "P_import",
        "P_production",
        "k_air",
        "k_dust",
        "Prod_emi_R",
        "P_protector",
    ],
}

# Same carpet parameters as the Monte Carlo analysis, bounds from the low / high estimates in config.py
CARPET_PARAMS = [
    "carpet_density",
    "lifespan_mean",
    "nylon_fmin",
    "average_floor_area",
    "initial_stock_age",
]


def build_sobol_problem() -> Tuple[dict, List[Tuple[str, str]]]:
    """
    Build a SALib Sobol problem from low/high configs.
    Returns:
        problem: SALib problem dict
        param_key_order: list of (subclass, param_name) in the same order as problem['names']
    """
    names = []
    bounds = []
    param_key_order = []

    for subclass, param_list in PARAMS_BY_SUBCLASS.items():
        low_cfg = PFAS_CONFIGS[(subclass, "low")]
        high_cfg = PFAS_CONFIGS[(subclass, "high")]

        for p in param_list:
            if p in EMISSION_RATES:
                lo, hi = emission_rate_bounds(low_cfg, high_cfg, p)
            else:
                low_val = getattr(low_cfg, p)
                high_val = getattr(high_cfg, p)

                if low_val is None or high_val is None:
                    continue

                lo = float(min(low_val, high_val))
                hi = float(max(low_val, high_val))

            # Skip fixed parameters with zero range
            if np.isclose(lo, hi):
                continue

            names.append(f"{subclass}_{p}")
            bounds.append([lo, hi])
            param_key_order.append((subclass, p))

    bounds_carpet = carpet_bounds()

    for p in CARPET_PARAMS:
        lo, hi = bounds_carpet[p]
        names.append(f"carpet_{p}")
        bounds.append([lo, hi])
        param_key_order.append(("carpet", p))

    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": bounds,
    }

    return problem, param_key_order


def build_sampled_configs(
    x: np.ndarray,
    param_key_order: List[Tuple[str, str]]
) -> Tuple[Dict[str, PFASConfig], CarpetConfig]:
    """
    Convert one Sobol sample vector into sampled configs for NP / PFAA / SP.
    Starts from the HIGH config template and replaces sampled parameters.
    """
    sampled_values = {
        "NP": {},
        "PFAA": {},
        "SP": {},
    }

    # Start with high configs as template
    for subclass in ["NP", "PFAA", "SP"]:
        high_cfg = PFAS_CONFIGS[(subclass, "high")]
        sampled_values[subclass] = {
            "subclass": high_cfg.subclass,
            "estimate": "sobol",
            "P_import": high_cfg.P_import,
            "P_production": high_cfg.P_production,
            "P_export": high_cfg.P_export,
            "P_yearlyloss_air": high_cfg.P_yearlyloss_air,
            "P_yearlyloss_dust": high_cfg.P_yearlyloss_dust,
            "Prod_emi_R": high_cfg.Prod_emi_R,
            "xt": high_cfg.xt,
            "P_protector": high_cfg.P_protector,
            "has_leachate": high_cfg.has_leachate,
            "use_protector": high_cfg.use_protector,
        }

    # Carpet parameters not in the problem stay at the central estimate
    central = CARPET_CONFIGS["central"]
    carpet_values = {
        "estimate": "sobol",
        "carpet_density": central.carpet_density,
        "lifespan_mean": central.lifespan_mean,
        "nylon_fmin": central.nylon_fmin,
        "average_floor_area": central.average_floor_area,
        "initial_stock_age": central.initial_stock_age,
    }

    rates = {"NP": {}, "PFAA": {}, "SP": {}}
    for value, (subclass, p) in zip(x, param_key_order):
        if subclass == "carpet":
            carpet_values[p] = float(value)
        elif p in EMISSION_RATES:
            rates[subclass][p] = float(value)
        else:
            sampled_values[subclass][p] = float(value)

    for subclass in ["NP", "PFAA", "SP"]:
        sampled_values[subclass]["P_export"] = sampled_values[subclass]["P_production"]
        # Sampled rates -> P_yearlyloss; rates not in the problem (zero range) keep the template value
        for rate, value in rates[subclass].items():
            sampled_values[subclass][EMISSION_RATES[rate]] = value * sampled_values[subclass]["P_production"]

    sampled_cfgs = {
        subclass: PFASConfig(**sampled_values[subclass])
        for subclass in ["NP", "PFAA", "SP"]
    }

    sampled_carpet_config = CarpetConfig(**carpet_values)

    return sampled_cfgs, sampled_carpet_config


def extract_years(total_df: pd.DataFrame) -> Dict[Tuple[int, str], float]:
    """Outputs at the selected years from one BAU evaluation (national R + C totals)."""
    outputs = {}
    for yr in YEARS_TO_ANALYZE:
        row = total_df.loc[total_df["year"] == yr].iloc[0]
        outputs[(yr, "stock")] = row["F_stock_total"]
        outputs[(yr, "emission")] = row["F6_total"]
        outputs[(yr, "landfill_accu")] = row["F_accu_landfill_total"]
        outputs[(yr, "landfill_emission")] = row["PFAS_landfill_leachate_total"]

    return outputs


def run_sobol_analysis(
    project_root: Path,
    N: int = 512,
    calc_second_order: bool = False,
    n_workers: int = None,
) -> None:
    """
    BAU Sobol sensitivity analysis of the national R + C totals, based on low-high bounds in config.py.
    """
    dirs = get_output_dirs(project_root)
    table_dir = dirs["tables_sobol"]
    fig_dir = dirs["figures_sobol"]

    plt.rcParams["font.family"] = "Arial"

    # 1) build problem
    problem, param_key_order = build_sobol_problem()
    param_names = problem["names"]

    print(f"Number of Sobol parameters: {problem['num_vars']}")
    print("Parameters:")
    for name in param_names:
        print("  ", name)

    # 2) sample
    param_values = sobol_sample.sample(
        problem,
        N,
        calc_second_order=calc_second_order,
        seed=SOBOL_SEED,
    )
    print(f"Sobol sample matrix shape: {param_values.shape}")

    # 3) run model (parallel; results in sample order)
    samples = [build_sampled_configs(x, param_key_order) for x in param_values]
    totals = evaluate_samples(project_root, samples, n_workers=n_workers or default_workers())
    check_nonnegative(totals)
    results_list = [extract_years(t) for t in totals]

    # 4) convert outputs
    Y = {}
    for yr in YEARS_TO_ANALYZE:
        for out in OUTPUT_ORDER:
            Y[(yr, out)] = np.array([r[(yr, out)] for r in results_list])

    # 5) analyze
    sobol_tables = {}

    for yr in YEARS_TO_ANALYZE:
        for out in OUTPUT_ORDER:
            Si = sobol_analyze.analyze(
                problem,
                Y[(yr, out)],
                calc_second_order=calc_second_order,
                print_to_console=False
            )

            df = pd.DataFrame({
                "parameter": param_names,
                "S1": Si["S1"],
                "S1_conf": Si["S1_conf"],
                "ST": Si["ST"],
                "ST_conf": Si["ST_conf"],
            })

            df["year"] = yr
            df["output"] = out

            sobol_tables[(yr, out)] = df.sort_values("ST", ascending=False).reset_index(drop=True)

    # 6) combine and save
    sobol_all = pd.concat(sobol_tables.values(), ignore_index=True)
    save_csv(sobol_all, table_dir / "sobol_results_all_years.csv")
    print(f"Saved: {table_dir / 'sobol_results_all_years.csv'}")

    # 7) print example
    for yr in YEARS_TO_ANALYZE:
        print(f"\n===== Sobol ranking for landfill emission ({yr}) =====")
        print(sobol_tables[(yr, "landfill_emission")][["parameter", "S1", "ST"]].head(10))

    # 8) yearly 2x2 plots
    for yr in YEARS_TO_ANALYZE:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()

        for ax, out in zip(axes, OUTPUT_ORDER):
            df_plot = sobol_tables[(yr, out)].sort_values("ST", ascending=True)
            ax.barh(df_plot["parameter"], df_plot["ST"])
            ax.set_title(f"{OUTPUT_TITLES[out]} ({yr})", fontsize=12)
            ax.set_xlabel("Total Sobol index (ST)", fontsize=11)
            ax.tick_params(axis="both", labelsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = fig_dir / f"sobol_2x2_{yr}.png"
        plt.savefig(fig_path, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path}")

    # 8b) 2020 1x4 plot for Figure 5 panels (e)-(h)
    yr = 2020

    output_titles_1x4_2020 = {
        "stock": "(e) In-use stock (2020)",
        "emission": "(f) In-use emission (2020)",
        "landfill_accu": "(g) Landfill accumulation (2020)",
        "landfill_emission": "(h) Landfill leachate (2020)",
    }

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, out in zip(axes, OUTPUT_ORDER):
        df_plot = sobol_tables[(yr, out)].sort_values("ST", ascending=True)

        ax.barh(df_plot["parameter"], df_plot["ST"])
        ax.set_title(output_titles_1x4_2020[out], fontsize=12)
        ax.set_xlabel("Total Sobol index (ST)", fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Parameter", fontsize=10)

    plt.tight_layout()
    fig_path = fig_dir / "sobol_1x4_2020_efgh.png"
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")

    # 9) 2x4 comparison for first and last selected years
    if len(YEARS_TO_ANALYZE) >= 2:
        year_1 = YEARS_TO_ANALYZE[0]
        year_2 = YEARS_TO_ANALYZE[-1]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=False, sharey=False)

        panel_labels = {
            (0, 0): "(a)",
            (0, 1): "(b)",
            (0, 2): "(c)",
            (0, 3): "(d)",
            (1, 0): "(e)",
            (1, 1): "(f)",
            (1, 2): "(g)",
            (1, 3): "(h)",
        }

        year_order = [year_1, year_2]

        for r, yr in enumerate(year_order):
            for c, out in enumerate(OUTPUT_ORDER):
                ax = axes[r, c]
                df_plot = sobol_tables[(yr, out)].sort_values("ST", ascending=True)

                ax.barh(df_plot["parameter"], df_plot["ST"])
                ax.set_title(f"{panel_labels[(r, c)]} {OUTPUT_TITLES[out][4:]} ({yr})", fontsize=12)
                ax.set_xlabel("Total Sobol index (ST)", fontsize=10)
                ax.tick_params(axis="both", labelsize=9)
                ax.grid(True, alpha=0.3)

                if c == 0:
                    ax.set_ylabel("Parameter", fontsize=10)

        plt.tight_layout()
        fig_path = fig_dir / f"sobol_2x4_compare_{year_1}_{year_2}.png"
        plt.savefig(fig_path, dpi=600, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fig_path}")

    # 10) parameter importance evolution plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for ax, out in zip(axes, OUTPUT_ORDER):
        st_table = pd.DataFrame(index=param_names, columns=YEARS_TO_ANALYZE, dtype=float)

        for yr in YEARS_TO_ANALYZE:
            df_plot = sobol_tables[(yr, out)].set_index("parameter")
            st_table[yr] = df_plot["ST"].reindex(param_names)

        st_table["max_ST"] = st_table.max(axis=1)
        top_params = st_table["max_ST"].sort_values(ascending=False).head(TOP_N_EVOLUTION).index.tolist()
        st_table = st_table.drop(columns="max_ST")

        for p in top_params:
            ax.plot(
                YEARS_TO_ANALYZE,
                st_table.loc[p, YEARS_TO_ANALYZE].values,
                marker="o",
                linewidth=1.8,
                label=p
            )

        ax.set_title(OUTPUT_TITLES[out], fontsize=14)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Total Sobol index (ST)", fontsize=12)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=9, loc="upper right")

    plt.tight_layout()
    fig_path = fig_dir / "sobol_parameter_importance_evolution.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")

    # 11) save ST evolution tables
    for out in OUTPUT_ORDER:
        st_table = pd.DataFrame(index=param_names, columns=YEARS_TO_ANALYZE, dtype=float)

        for yr in YEARS_TO_ANALYZE:
            df_plot = sobol_tables[(yr, out)].set_index("parameter")
            st_table[yr] = df_plot["ST"].reindex(param_names)

        st_table = st_table.reset_index().rename(columns={"index": "parameter"})
        save_csv(st_table, table_dir / f"sobol_ST_evolution_{out}.csv")
        print(f"Saved: {table_dir / f'sobol_ST_evolution_{out}.csv'}")