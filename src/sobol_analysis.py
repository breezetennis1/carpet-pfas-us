from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

from src.config import PFAS_CONFIGS, PFASConfig, CarpetConfig
from src.io_utils import load_basic_data, save_csv
from src.model import run_national_model
from src.scenario import apply_scenario
from src.paths import get_output_dirs


# =========================
# 1. User settings
# =========================
YEARS_TO_ANALYZE = [2020, 2040, 2060]
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
PARAMS_BY_SUBCLASS = {
    "NP": [
        "P_import",
        "P_production",
        "P_export",
        "P_yearlyloss_air",
        "P_yearlyloss_dust",
        "Prod_emi_R",
        "xt",
    ],
    "PFAA": [
        "P_import",
        "P_production",
        "P_export",
        "P_yearlyloss_air",
        "P_yearlyloss_dust",
        "Prod_emi_R",
        "xt",
    ],
    "SP": [
        "P_import",
        "P_production",
        "P_export",
        "P_yearlyloss_air",
        "P_yearlyloss_dust",
        "Prod_emi_R",
        "P_protector",
    ],
}

CARPET_PARAMS = [
    "carpet_density",
    "lifespan_mean",
    "nylon_fmin",
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

    carpet_bounds = {
        "carpet_density": [2.2 * 1000, 2.5 * 1000],
        "lifespan_mean": [7.0, 11.25],
        "nylon_fmin": [0.20, 0.40],
        "average_floor_area": [143.1, 174.9],
        "initial_stock_age": [3.87, 4.73],
    }

    for p in CARPET_PARAMS:
        lo, hi = carpet_bounds[p]
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

    carpet_values = {
        "estimate": "sobol",
        "carpet_density": 2.35 * 1000,
        "lifespan_mean": 9.0,
        "nylon_fmin": 0.30,
        "average_floor_area": 159.0,
        "initial_stock_age": 4.3,
    }

    for value, (subclass, p) in zip(x, param_key_order):
        if subclass == "carpet":
            carpet_values[p] = float(value)
        else:
            sampled_values[subclass][p] = float(value)

    sampled_cfgs = {
        subclass: PFASConfig(**sampled_values[subclass])
        for subclass in ["NP", "PFAA", "SP"]
    }

    sampled_carpet_config = CarpetConfig(**carpet_values)

    return sampled_cfgs, sampled_carpet_config


def combine_total_outputs(result_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    years = result_dict["NP"]["year"].copy()

    total = pd.DataFrame({
        "year": years,
        "F_stock_total": 0.0,
        "F6_total": 0.0,
        "F_accu_landfill_total": 0.0,
        "PFAS_landfill_leachate_total": 0.0,
    })

    for subclass, df in result_dict.items():
        total["F_stock_total"] += df["F_stock"].fillna(0.0)
        total["F6_total"] += df["F6"].fillna(0.0)
        total["F_accu_landfill_total"] += df["F_accu_landfill"].fillna(0.0)

        if "PFAS_landfill_leachate" in df.columns:
            total["PFAS_landfill_leachate_total"] += df["PFAS_landfill_leachate"].fillna(0.0)

    return total


def run_model_extract_years(
    x: np.ndarray,
    project_root: Path,
    param_key_order: List[Tuple[str, str]],
) -> Dict[Tuple[int, str], float]:
    """
    Run one BAU model evaluation and extract outputs at selected years.
    """
    input_csv = project_root / "data" / "Basic_data.csv"
    base_df = load_basic_data(input_csv)
    bau_df = apply_scenario(base_df, "BAU")

    sampled_cfgs, sampled_carpet_config = build_sampled_configs(x, param_key_order)

    subclass_results = {}
    for subclass in ["NP", "PFAA", "SP"]:
        subclass_results[subclass] = run_national_model(
            bau_df,
            sampled_cfgs[subclass],
            sampled_carpet_config,
        )

    total_df = combine_total_outputs(subclass_results)

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
) -> None:
    """
    BAU Sobol sensitivity analysis based on low-high bounds in config.py.
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
        calc_second_order=calc_second_order
    )
    print(f"Sobol sample matrix shape: {param_values.shape}")

    # 3) run model
    results_list = []
    for i, x in enumerate(param_values):
        outputs = run_model_extract_years(x, project_root, param_key_order)
        results_list.append(outputs)

        if (i + 1) % 100 == 0:
            print(f"{i + 1} Sobol runs finished")

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