from __future__ import annotations

from pathlib import Path
import pandas as pd
from pandas.api.types import is_numeric_dtype
from src.paths import get_output_dirs

from src.config import PFAS_CONFIGS, CARPET_CONFIGS, MODEL_CONSTANTS, CarpetConfig
from src.io_utils import (
    load_basic_data,
    load_cbecs_commercial_floorspace_1990,
    load_commercial_share,
    save_csv,
)
from src.model import run_national_model
from src.postprocess import CORE_METRICS, build_total_output, build_contribution_long
from src.scenario import apply_scenario


COMMERCIAL_SHARE_CSV = Path("data") / "raw" / "national_inputs" / "carpet_residential_commercial_share_1990_2060.csv"
CBECS_FLOORSPACE_CSV = Path("data") / "raw" / "CBECS_1992_floorspace_by_PBA.csv"

# Initial commercial stock S0_C:
#   "cbecs"       : S0_C = CBECS commercial floor area (1990) x P_cf x Percent_nylon (same form as Eq. S24)
#   "sales_ratio" : S0_C = S0_R x phi_1990 / (1 - phi_1990)   (sensitivity case only)
S0_C_METHOD = "cbecs"
S0_C_METHODS = ("cbecs", "sales_ratio")

ESTIMATES = ["high", "central", "low"]

# Residential (R) and commercial (C) streams; TOTAL = R + C keeps the original file names.
STREAM_SUFFIXES = {"TOTAL": "", "R": "_R", "C": "_C"}

# Volume inputs split by the commercial share phi_k: R = (1 - phi_k) x input, C = phi_k x input.
# V_sales is not used by the model; it is split so that R + C = TOTAL holds for every volume column.
STREAM_SPLIT_COLUMNS = ["V_sales", "V_sales_nylon", "V_production", "V_import", "V_export"]

# Columns that are identical in both streams (rates, shares, fit parameters, labels): TOTAL takes them from R.
# Every other output column is a volume, mass or PFAS stock/flow and is summed.
NON_ADDITIVE_COLUMNS = {
    "year",
    "us_pop",
    "Percent_nylon",
    "Percent_nylon_fitted",
    "Percent_nylon_original",
    "Percent_nylon_source",
    "nylon_fmin",
    "nylon_fmax",
    "nylon_r_fit",
    "nylon_t0_fit",
    "RR_total",
    "RR_total_24",
    "RR_effective",
    "IncR",
    "PhaseOut",
    "PhaseOut_effective",
    "landfill_ratio",
    "scenario",
}


def split_streams(scenario_df: pd.DataFrame, phi: pd.Series) -> dict[str, pd.DataFrame]:
    phi_k = scenario_df["year"].map(phi)
    if phi_k.isna().any():
        missing = scenario_df.loc[phi_k.isna(), "year"].tolist()
        raise ValueError(f"No commercial share for years {missing}")

    df_R = scenario_df.copy()
    df_C = scenario_df.copy()
    for col in STREAM_SPLIT_COLUMNS:
        df_R[col] = scenario_df[col] * (1 - phi_k)
        df_C[col] = scenario_df[col] * phi_k
    return {"R": df_R, "C": df_C}


def compute_s0_scale_C(
    method: str,
    carpet_config: CarpetConfig,
    us_pop_1990: float,
    phi_1990: float,
    floorspace_c_1990: float,
) -> float:
    if method == "cbecs":
        c = MODEL_CONSTANTS
        residential_floorspace_1990 = carpet_config.average_floor_area / c.initial_household_size * us_pop_1990
        return floorspace_c_1990 / residential_floorspace_1990
    if method == "sales_ratio":
        return phi_1990 / (1 - phi_1990)
    raise ValueError(f"Unsupported S0_C method: {method}")


def build_s0_scales(
    base_df: pd.DataFrame,
    phi: pd.Series,
    floorspace_c_1990: float,
    method: str = S0_C_METHOD,
) -> dict[str, dict[str, float]]:
    # Same first-row values that compute_initial_stock_volume uses for S0
    us_pop_1990 = float(base_df.loc[0, "us_pop"])
    phi_1990 = float(phi.loc[int(base_df.loc[0, "year"])])
    return {
        "R": {estimate: 1.0 for estimate in ESTIMATES},
        "C": {
            estimate: compute_s0_scale_C(method, CARPET_CONFIGS[estimate], us_pop_1990, phi_1990, floorspace_c_1990)
            for estimate in ESTIMATES
        },
    }


def combine_streams(res_R: pd.DataFrame, res_C: pd.DataFrame) -> pd.DataFrame:
    if list(res_R.columns) != list(res_C.columns) or not res_R["year"].equals(res_C["year"]):
        raise ValueError("R and C results do not have the same columns and years")

    total = res_R.copy()
    for col in res_R.columns:
        if col in NON_ADDITIVE_COLUMNS:
            if not res_R[col].equals(res_C[col]):
                raise ValueError(f"Non-additive column '{col}' differs between R and C")
            continue
        if not is_numeric_dtype(res_R[col]):
            raise ValueError(f"Column '{col}' is neither numeric nor listed as non-additive")
        total[col] = res_R[col] + res_C[col]
    return total


def run_scenario_streams(
    base_df: pd.DataFrame,
    scenario: str,
    phi: pd.Series,
    s0_scales: dict[str, dict[str, float]],
) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    """Results as {stream: {estimate: {subclass: df}}} for stream in R, C, TOTAL."""
    scenario_df = apply_scenario(base_df, scenario)
    stream_inputs = split_streams(scenario_df, phi)

    results = {stream: {estimate: {} for estimate in ESTIMATES} for stream in STREAM_SUFFIXES}

    for (subclass, estimate), config in PFAS_CONFIGS.items():
        carpet_config = CARPET_CONFIGS[estimate]
        for stream in ["R", "C"]:
            results[stream][estimate][subclass] = run_national_model(
                stream_inputs[stream],
                config,
                carpet_config,
                s0_scale=s0_scales[stream][estimate],
            )
        results["TOTAL"][estimate][subclass] = combine_streams(
            results["R"][estimate][subclass],
            results["C"][estimate][subclass],
        )

    return results


def save_stream_outputs(
    grouped_results: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path,
    scenario: str,
    suffix: str,
) -> dict[str, pd.DataFrame]:
    for estimate, by_subclass in grouped_results.items():
        for subclass, result_df in by_subclass.items():
            filename = f"{subclass}_{estimate.upper()}_{scenario}{suffix}.csv"
            save_csv(result_df, output_dir / filename)
            print(f"Saved: {filename}")

    totals = {estimate: build_total_output(grouped_results[estimate]) for estimate in ESTIMATES}
    for estimate in ESTIMATES:
        save_csv(totals[estimate], output_dir / f"TOTAL_{estimate.upper()}_ALL_SUBCLASSES_{scenario}{suffix}.csv")

    contrib_high = build_contribution_long(grouped_results["high"], "high")
    contrib_central = build_contribution_long(grouped_results["central"], "central")
    contrib_low = build_contribution_long(grouped_results["low"], "low")
    contribution_long = pd.concat([contrib_high, contrib_low], ignore_index=True)

    contribution_long["scenario"] = scenario
    save_csv(contribution_long, output_dir / f"PFAS_subclass_contribution_long_{scenario}{suffix}.csv")

    print(f"Saved: TOTAL_*_ALL_SUBCLASSES_{scenario}{suffix}.csv")
    print(f"Saved: PFAS_subclass_contribution_long_{scenario}{suffix}.csv")
    return totals


def build_rc_share_summary(totals: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Long table of all-subclass totals by stream: year, estimate, variable, R, C, TOTAL, C_share (kg)."""
    frames = []
    for estimate in ["central", "high", "low"]:
        for variable in CORE_METRICS:
            r = totals["R"][estimate][variable]
            c = totals["C"][estimate][variable]
            t = totals["TOTAL"][estimate][variable]
            frames.append(pd.DataFrame({
                "year": totals["TOTAL"][estimate]["year"],
                "estimate": estimate.upper(),
                "variable": variable,
                "R": r,
                "C": c,
                "TOTAL": t,
                "C_share": c / t.where(t != 0),
            }))
    return pd.concat(frames, ignore_index=True)


def run_all_cases(project_root: Path, scenario: str = "BAU", s0_c_method: str = S0_C_METHOD) -> None:
    input_csv = project_root / "data" / "Basic_data.csv"
    dirs = get_output_dirs(project_root)
    output_dir = dirs["tables_national"]
    output_dir.mkdir(parents=True, exist_ok=True)

    base_df = load_basic_data(input_csv)
    phi = load_commercial_share(project_root / COMMERCIAL_SHARE_CSV)
    floorspace_c_1990 = load_cbecs_commercial_floorspace_1990(project_root / CBECS_FLOORSPACE_CSV)
    s0_scales = build_s0_scales(base_df, phi, floorspace_c_1990, s0_c_method)
    print(f"S0_C method: {s0_c_method}; s0_scale_C = {s0_scales['C']}")

    results = run_scenario_streams(base_df, scenario, phi, s0_scales)

    totals = {}
    for stream, suffix in STREAM_SUFFIXES.items():
        totals[stream] = save_stream_outputs(results[stream], output_dir, scenario, suffix)

    summary = build_rc_share_summary(totals)
    save_csv(summary, output_dir / f"R_C_share_summary_{scenario}.csv")
    print(f"Saved: R_C_share_summary_{scenario}.csv")
