from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import lognorm
from scipy.optimize import curve_fit

from src.config import MODEL_CONSTANTS, PFASConfig, CarpetConfig


def get_lifetime_distribution(carpet_config: CarpetConfig):

    # Fixed CV from your original assumption
    CV = 2.6 / 8.6

    mean = carpet_config.lifespan_mean
    sd = mean * CV

    sigma = np.sqrt(np.log(1 + (sd**2 / mean**2)))
    mu = np.log(mean) - 0.5 * sigma**2

    shape = sigma
    scale = np.exp(mu)
    loc = 0.0

    return lognorm(shape, loc=loc, scale=scale)


def logistic_nylon_share(t, r, t0, fmin, fmax):
    return fmin + (fmax - fmin) / (1 + np.exp(r * (t - t0)))


def fit_and_apply_nylon_share(
    df: pd.DataFrame,
    carpet_config: CarpetConfig
) -> pd.DataFrame:
    out = df.copy()

    observed_mask = (
        out["Percent_nylon"].notna()
        & (out["year"] >= 2000)
        & (out["year"] <= 2015)
    )

    obs = out.loc[observed_mask, ["year", "Percent_nylon"]].copy()

    years_obs = obs["year"].to_numpy(dtype=float)
    nylon_obs = obs["Percent_nylon"].to_numpy(dtype=float)

    fmin = carpet_config.nylon_fmin
    fmax = float(np.nanmax(nylon_obs))

    def fit_func(t, r, t0):
        return logistic_nylon_share(t, r, t0, fmin=fmin, fmax=fmax)

    params, _ = curve_fit(
        fit_func,
        years_obs,
        nylon_obs,
        p0=[0.10, 2009],
        bounds=([0.001, 1990], [1.0, 2030]),
        maxfev=10000,
    )

    r_fit, t0_fit = params

    out["Percent_nylon_fitted"] = logistic_nylon_share(
        out["year"].to_numpy(dtype=float),
        r_fit,
        t0_fit,
        fmin=fmin,
        fmax=fmax,
    )

    out["Percent_nylon_original"] = out["Percent_nylon"]

    out["Percent_nylon_source"] = np.where(
        out["Percent_nylon"].notna(),
        "historical",
        "logistic_fitted"
    )

    out["Percent_nylon"] = out["Percent_nylon"].fillna(
        out["Percent_nylon_fitted"]
    )

    out["nylon_fmin"] = fmin
    out["nylon_fmax"] = fmax
    out["nylon_r_fit"] = r_fit
    out["nylon_t0_fit"] = t0_fit

    return out


def compute_initial_stock_volume(
    df: pd.DataFrame,
    carpet_config: CarpetConfig
) -> float:
    c = MODEL_CONSTANTS
    return (
        (carpet_config.average_floor_area / c.initial_household_size)
        * c.P_cf
        * df.loc[0, "us_pop"]
        * df.loc[0, "Percent_nylon"]
    )


def build_yearly_eol_matrix(years, sales, carpet_config) -> np.ndarray:
    dist = get_lifetime_distribution(carpet_config)
    sales_year = years[:, None]
    target_year = years[None, :]
    age = target_year - sales_year

    probs = np.where(
        age >= 0,
        dist.cdf(age + 1) - dist.cdf(age),
        0.0,
    )
    return probs * sales[:, None]


def add_volume_side(df, carpet_config, s0_scale: float = 1.0) -> pd.DataFrame:
    years = df["year"].to_numpy()
    sales_base = df["V_sales_nylon"].to_numpy(dtype=float)

    # For phase-out, new market inflow should be reduced by PhaseOut_effective
    if "PhaseOut_effective" in df.columns:
        sales = sales_base * df["PhaseOut_effective"].to_numpy(dtype=float)
    else:
        sales = sales_base.copy()

    dist = get_lifetime_distribution(carpet_config)
    n = len(df)

    S0_volumn = compute_initial_stock_volume(df, carpet_config) * s0_scale

    yearly_eol = build_yearly_eol_matrix(years, sales, carpet_config)
    V_new_EoL = yearly_eol.sum(axis=0)

    i = np.arange(n, dtype=float)
    V_iniStock_EoL = (
        (dist.cdf(carpet_config.initial_stock_age + i + 1) - dist.cdf(carpet_config.initial_stock_age + i))
        / (1 - dist.cdf(carpet_config.initial_stock_age))
        * S0_volumn
    )

    V4 = V_iniStock_EoL + V_new_EoL
    V_delta_S = sales - V4

    V_stock = np.empty(n, dtype=float)
    V_stock[0] = S0_volumn + V_delta_S[0]
    for idx in range(1, n):
        V_stock[idx] = V_stock[idx - 1] + V_delta_S[idx]

    df["V_sales_effective"] = sales
    df["V_iniStock_EoL"] = V_iniStock_EoL
    df["V_new_EoL"] = V_new_EoL
    df["V4"] = V4
    df["V_delta_S"] = V_delta_S
    df["V_stock"] = V_stock
    return df


def add_sp_protector_block(df, config, carpet_config) -> pd.DataFrame:
    """
    Reproduce the original notebook logic for SP protector use (F10).

    Logic:
    1. Build yearly cohort EoL matrix from V_sales_nylon.
    2. Build yearly remaining stock matrix (df_yearly_left).
    3. From 1994 onward, sum remaining cohorts by parity:
       - even years sum even-index cohorts
       - odd years sum odd-index cohorts
    4. F10 = V_new_left * P_protector
    """
    if not config.use_protector:
        df["V_new_left"] = 0.0
        df["F10"] = 0.0
        return df

    years = df["year"].to_numpy()
    sales = df["V_sales_nylon"].to_numpy(dtype=float)
    n = len(df)

    # yearly_eol[row_i, col_j] = EoL from cohort i in year j
    yearly_eol = build_yearly_eol_matrix(years, sales, carpet_config)

    # Reconstruct df_yearly_left exactly like the notebook
    # rows = cohort index i
    # cols = calendar year
    df_yearly_left = pd.DataFrame(0.0, index=range(n), columns=years)

    for i in range(n):
        cohort_year = int(df.loc[i, "year"])
        for year in years:
            if year >= cohort_year:
                # sum yearly_eol from cohort year up to current year
                col_end = int(year - years[0]) + 1
                col_start = int(cohort_year - years[0])
                sum_pre_year = yearly_eol[i, col_start:col_end].sum()
                df_yearly_left.loc[i, year] = df.loc[i, "V_sales_nylon"] - sum_pre_year

    # Initialize V_new_left
    df["V_new_left"] = 0.0

    # Replicate original notebook logic:
    # b = 4 corresponds to row of year 1994
    b = int(np.where(years == 1994)[0][0]) if 1994 in years else n

    for year in range(1994, int(years[-1]) + 1):
        a = 0.0

        if year % 2 == 0:
            for i in range(0, n, 2):
                a += df_yearly_left.loc[i, year]
        else:
            for i in range(1, n, 2):
                a += df_yearly_left.loc[i, year]

        if b < n:
            df.loc[b, "V_new_left"] = a
            b += 1

    # PFAS input from protector
    df["F10"] = df["V_new_left"] * float(config.P_protector)

    return df


def add_flow_side(
    df: pd.DataFrame,
    config: PFASConfig,
    carpet_config: CarpetConfig,
    s0_scale: float = 1.0,
) -> pd.DataFrame:
    c = MODEL_CONSTANTS
    n = len(df)

    S0_volumn = compute_initial_stock_volume(df, carpet_config) * s0_scale
    S0_PFAS = (S0_volumn * config.P_production * c.initial_stock_retention) / 1000

    phase = df["PhaseOut_effective"].to_numpy(dtype=float) if "PhaseOut_effective" in df.columns else np.ones(n)
    rr = df["RR_effective"].to_numpy(dtype=float) if "RR_effective" in df.columns else df["RR_total"].to_numpy(dtype=float)

    df["F1"] = (
        df["V_import"] * config.P_import * df["Percent_nylon"]
        - df["V_export"] * config.P_export * df["Percent_nylon"]
    ) / 1000

    # Keep F1 unchanged unless your original phase-out notebook also phase-outs import/export
    # If later you want phase-out to affect imports/exports too, multiply F1 by phase.

    df["F_production"] = (
        df["V_production"] * config.P_production * df["Percent_nylon"] * phase
    ) / 1000

    df["F3"] = df["F_production"] * c.production_waste_fraction * 0.15

    df["F_to_air"] = config.P_yearlyloss_air * df["V_stock"] / 1000
    df["F_to_dust"] = config.P_yearlyloss_dust * df["V_stock"] / 1000
    df["F6"] = (df["F_to_air"] + df["F_to_dust"]) * 2  # x2: F_clean = F_air + F_dust (SI Eq. S15)

    df["F2"] = 0.0
    df["F4"] = 0.0
    df["F5"] = 0.0
    df["F_delta_S"] = 0.0
    df["F_stock"] = 0.0

    F10 = df["F10"].to_numpy(dtype=float) if "F10" in df.columns else np.zeros(n, dtype=float)

    df.loc[0, "F4"] = (df.loc[0, "V4"] / S0_volumn) * (S0_PFAS-df.loc[0, "F6"])
    df.loc[0, "F2"] = df.loc[0, "F1"] + df.loc[0, "F_production"] + df.loc[0, "F5"] - df.loc[0, "F3"]
    df.loc[0, "F_delta_S"] = df.loc[0, "F2"] + F10[0] - df.loc[0, "F4"] - df.loc[0, "F6"]
    df.loc[0, "F_stock"] = max(S0_PFAS + df.loc[0, "F_delta_S"], 0.0)

    for idx in range(1, n):
        prev_stock = max(df.loc[idx - 1, "F_stock"], 0.0)
        prev_v_stock = df.loc[idx - 1, "V_stock"]

        raw_f4 = (
            (df.loc[idx - 1, "V4"] / prev_v_stock) * (prev_stock-df.loc[idx-1, "F6"])
            if prev_v_stock > 0 else 0.0
        )
        df.loc[idx, "F4"] = raw_f4

        df.loc[idx, "F5"] = df.loc[idx - 1, "F4"] * rr[idx] * MODEL_CONSTANTS.R_to_carpet
        df.loc[idx, "F2"] = df.loc[idx, "F1"] + df.loc[idx, "F_production"] + df.loc[idx, "F5"] - df.loc[idx, "F3"]
        df.loc[idx, "F_delta_S"] = df.loc[idx, "F2"] + df.loc[idx, "F10"] - df.loc[idx, "F4"] - df.loc[idx, "F6"]
        df.loc[idx, "F_stock"] = max(prev_stock + df.loc[idx, "F_delta_S"], 0.0)

    df["F8"] = (df["F4"] + df["F3"]) * df["IncR"]
    df["F9"] = df["F_production"] * config.Prod_emi_R
    df["F_EoL_total"] = df["F3"] + df["F4"]
    df["landfill_ratio"] = 1 - rr - df["IncR"]
    df["F_landfill_yearly"] = df["F_EoL_total"] * df["landfill_ratio"]
    df["F_accu_landfill"] = df["F_landfill_yearly"].cumsum()

    return df


def add_leachate_block(df: pd.DataFrame, config: PFASConfig, carpet_config: CarpetConfig) -> pd.DataFrame:
    c = MODEL_CONSTANTS

    if not config.has_leachate:
        df["V_annual_landfill"] = 0.0
        df["V_accum_landfill"] = 0.0
        df["Mass_accum_landfill"] = 0.0
        df["PFAS_landfill_leachate"] = 0.0
        return df

    phase = df["PhaseOut_effective"].to_numpy(dtype=float) if "PhaseOut_effective" in df.columns else 1.0

    df["V_annual_landfill"] = (
            df["V4"] * df["landfill_ratio"]
            + df["V_production"] * df["Percent_nylon"] * c.production_scrap_to_landfill_fraction * phase
    )
    df["V_accum_landfill"] = df["V_annual_landfill"].cumsum()
    df["Mass_accum_landfill"] = df["V_accum_landfill"] * carpet_config.carpet_density
    df["PFAS_landfill_leachate"] = (
        df["Mass_accum_landfill"]
        * float(config.xt)
        * c.LG
        / c.SA
        * 365
        / 1_000_000
        * c.leachate_release_fraction
    )
    return df


def run_national_model(
    df: pd.DataFrame,
    config: PFASConfig,
    carpet_config: CarpetConfig,
    s0_scale: float = 1.0,
) -> pd.DataFrame:
    out = df.copy()

    out = fit_and_apply_nylon_share(out, carpet_config)

    out = add_volume_side(out, carpet_config, s0_scale)
    out = add_sp_protector_block(out, config, carpet_config)
    out = add_flow_side(out, config, carpet_config, s0_scale)
    out = add_leachate_block(out, config, carpet_config)

    return out