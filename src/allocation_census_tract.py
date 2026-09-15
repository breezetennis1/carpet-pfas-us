from pathlib import Path
import pandas as pd
import numpy as np


def _read_csv_case_insensitive(path_without_suffix: Path) -> pd.DataFrame:
    """Read .csv or .CSV file."""
    for suffix in [".csv", ".CSV"]:
        path = path_without_suffix.with_suffix(suffix)
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"Cannot find {path_without_suffix}.csv or .CSV")


def _check_required_columns(df: pd.DataFrame, required_cols: list[str], file_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{file_name} is missing required columns: {missing}")


def update_total_central_bau(project_root: Path) -> pd.DataFrame:
    """
    Add F_landfill_yearly and V_annual_landfill into
    TOTAL_CENTRAL_ALL_SUBCLASSES_BAU.csv.

    F_landfill_yearly = NP + SP + PFAA
    V_annual_landfill = carpet volume, so use one consistent subclass file.
    """

    national_dir = project_root / "outputs" / "tables" / "national"

    np_df = pd.read_csv(national_dir / "NP_CENTRAL_BAU.csv")
    sp_df = pd.read_csv(national_dir / "SP_CENTRAL_BAU.csv")
    pfaa_df = pd.read_csv(national_dir / "PFAA_CENTRAL_BAU.csv")
    total_df = pd.read_csv(national_dir / "TOTAL_CENTRAL_ALL_SUBCLASSES_BAU.csv")

    for name, df in {
        "NP_CENTRAL_BAU.csv": np_df,
        "SP_CENTRAL_BAU.csv": sp_df,
        "PFAA_CENTRAL_BAU.csv": pfaa_df,
        "TOTAL_CENTRAL_ALL_SUBCLASSES_BAU.csv": total_df,
    }.items():
        _check_required_columns(df, ["year"], name)
        df["year"] = df["year"].astype(int)

    for name, df in {
        "NP_CENTRAL_BAU.csv": np_df,
        "SP_CENTRAL_BAU.csv": sp_df,
        "PFAA_CENTRAL_BAU.csv": pfaa_df,
    }.items():
        _check_required_columns(
            df,
            ["year", "F_landfill_yearly", "V_annual_landfill"],
            name,
        )

    landfill_yearly = (
        np_df[["year", "F_landfill_yearly"]]
        .rename(columns={"F_landfill_yearly": "NP_F_landfill_yearly"})
        .merge(
            sp_df[["year", "F_landfill_yearly"]]
            .rename(columns={"F_landfill_yearly": "SP_F_landfill_yearly"}),
            on="year",
            how="inner",
        )
        .merge(
            pfaa_df[["year", "F_landfill_yearly"]]
            .rename(columns={"F_landfill_yearly": "PFAA_F_landfill_yearly"}),
            on="year",
            how="inner",
        )
    )

    landfill_yearly["F_landfill_yearly"] = (
        landfill_yearly["NP_F_landfill_yearly"]
        + landfill_yearly["SP_F_landfill_yearly"]
        + landfill_yearly["PFAA_F_landfill_yearly"]
    )

    # V_annual_landfill is carpet volume, not PFAS subclass mass.
    # Therefore, do NOT sum NP + SP + PFAA.
    volume = np_df[["year", "V_annual_landfill"]].copy()

    add_cols = landfill_yearly[["year", "F_landfill_yearly"]].merge(
        volume,
        on="year",
        how="inner",
    )

    total_df = total_df.drop(
        columns=[
            c for c in ["F_landfill_yearly", "V_annual_landfill"]
            if c in total_df.columns
        ]
    )

    total_updated = total_df.merge(add_cols, on="year", how="left")

    out_file = national_dir / "TOTAL_CENTRAL_ALL_SUBCLASSES_BAU_updated.csv"
    total_updated.to_csv(out_file, index=False)

    print(f"Saved updated national file: {out_file}")

    return total_updated


def allocate_national_to_census_tracts(project_root: Path) -> None:
    """
    Allocate national central BAU carpet PFAS results to census tracts.

    Outputs:
    - outputs/tables/census_tracts/F_stock/
    - outputs/tables/census_tracts/F6/
    - outputs/tables/census_tracts/F_landfill_yearly/
    - outputs/tables/census_tracts/V_annual_landfill/
    - outputs/tables/census_tracts/combined/
    """

    total_df = update_total_central_bau(project_root)

    weights_dir = project_root / "data" / "census_tracts" / "allocation_weights"
    pop_dir = project_root / "data" / "census_tracts" / "population"

    output_dir = project_root / "outputs" / "tables" / "census_tracts"
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    allocation_vars = {
        "F_stock": "F_stock",
        "F6": "F6",
        "F_landfill_yearly": "F_landfill_yearly",
        "V_annual_landfill": "V_annual_landfill",
    }

    required_total_cols = ["year"] + list(allocation_vars.keys())
    _check_required_columns(
        total_df,
        required_total_cols,
        "TOTAL_CENTRAL_ALL_SUBCLASSES_BAU_updated.csv",
    )

    all_years = range(1990, 2061)
    combined_records = []
    qa_records = []

    for year in all_years:
        weights_file_stem = weights_dir / f"CT_W_Area_{year}"
        ct_df = _read_csv_case_insensitive(weights_file_stem)

        _check_required_columns(
            ct_df,
            ["Portion"],
            f"CT_W_Area_{year}.csv",
        )

        if "GEOID" not in ct_df.columns:
            print(f"Warning: GEOID not found in CT_W_Area_{year}.csv")

        portion_sum = ct_df["Portion"].sum()

        if not np.isclose(portion_sum, 1.0, rtol=1e-5, atol=1e-8):
            print(
                f"Warning: Portion sum for {year} = {portion_sum:.8f}, not exactly 1."
            )

        year_values = total_df.loc[total_df["year"] == year]

        if year_values.empty:
            raise ValueError(f"No national value found for year {year}")

        year_values = year_values.iloc[0]

        df_combined_year = ct_df.copy()
        df_combined_year["year"] = year

        for national_col, output_folder_name in allocation_vars.items():
            national_value = year_values[national_col]

            df_out = ct_df.copy()
            df_out[str(year)] = df_out["Portion"] * national_value

            var_dir = output_dir / output_folder_name
            var_dir.mkdir(parents=True, exist_ok=True)

            out_file = var_dir / f"CT_{output_folder_name}_{year}.csv"
            df_out.to_csv(out_file, index=False)

            df_combined_year[national_col] = df_out[str(year)]

            allocated_sum = df_out[str(year)].sum()
            diff = allocated_sum - national_value

            qa_records.append(
                {
                    "year": year,
                    "variable": national_col,
                    "national_value": national_value,
                    "allocated_sum": allocated_sum,
                    "difference": diff,
                    "portion_sum": portion_sum,
                }
            )

        # Optional per-capita calculations if population file exists
        pop_file_csv = pop_dir / f"CT_Population_{year}.csv"
        pop_file_CSV = pop_dir / f"CT_Population_{year}.CSV"

        if pop_file_csv.exists() or pop_file_CSV.exists():
            pop_df = _read_csv_case_insensitive(pop_dir / f"CT_Population_{year}")

            pop_col = None
            for candidate in [str(year), "population", "Population", "POP", "pop"]:
                if candidate in pop_df.columns:
                    pop_col = candidate
                    break

            if pop_col is not None:
                if "GEOID" in df_combined_year.columns and "GEOID" in pop_df.columns:
                    df_combined_year = df_combined_year.merge(
                        pop_df[["GEOID", pop_col]],
                        on="GEOID",
                        how="left",
                    )
                else:
                    df_combined_year[pop_col] = pop_df[pop_col].values

                population = df_combined_year[pop_col].replace(0, np.nan)

                # F_stock usually tonnes; convert to g/cap
                df_combined_year["F_stock_g_per_cap"] = (
                    df_combined_year["F_stock"] * 1_000_000 / population
                )

                # F6 usually tonnes/year; convert to mg/cap/year
                df_combined_year["F6_mg_per_cap_yr"] = (
                    df_combined_year["F6"] * 1_000_000_000 / population
                )

        combined_records.append(df_combined_year)

    combined_df = pd.concat(combined_records, ignore_index=True)

    combined_file = combined_dir / "CT_allocation_CENTRAL_BAU_1990_2060_long.csv"
    combined_df.to_csv(combined_file, index=False)

    qa_df = pd.DataFrame(qa_records)
    qa_file = combined_dir / "CT_allocation_QA_check.csv"
    qa_df.to_csv(qa_file, index=False)

    print(f"Saved combined census tract allocation file: {combined_file}")
    print(f"Saved QA check file: {qa_file}")
    print("Finished census tract allocation.")