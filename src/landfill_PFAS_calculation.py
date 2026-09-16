from pathlib import Path
import pandas as pd


START_YEAR = 1990
END_YEAR = 2060
YEARS = [str(y) for y in range(START_YEAR, END_YEAR + 1)]

# Carpet areal density, central estimate (SI Table S1: low/central/high = 2.2/2.35/2.5 kg/m^2),
# matching CarpetConfig.carpet_density in src/config.py. Was previously 23.9 * 1000 (10x too high).
CARPET_MASS_FACTOR = 2.35 * 1000
SA = 140
LG_DAYS = 365
# Not documented in the SI as of the current draft (SI_v10.docx) - Eq. S16 (F7 = xT * WIP/SA * LG)
# has no release/active-fraction term. Must be added to Eq. S16 during the revision.
LEACHATE_ACTIVE_FRACTION = 0.35

XT_NP = 10.6
XT_PFAA = 4.75


def read_yearly_assigned_tables(input_dir: Path, file_prefix: str, landfill_type: str = "MLF") -> pd.DataFrame:
    merged = None

    for year in range(START_YEAR, END_YEAR + 1):
        file_path = input_dir / f"{file_prefix}_{year}_{landfill_type}_pfas_assigned.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        df_year = pd.read_csv(file_path)

        if "facility_id" not in df_year.columns:
            raise ValueError(f"'facility_id' not found in {file_path}")

        year_col = str(year)

        if year_col not in df_year.columns:
            numeric_cols = df_year.select_dtypes(include="number").columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != "facility_id"]

            if not numeric_cols:
                raise ValueError(f"No yearly numeric value column found in {file_path}")

            value_col = numeric_cols[-1]
            df_year = df_year[["facility_id", value_col]].rename(columns={value_col: year_col})
        else:
            df_year = df_year[["facility_id", year_col]]

        merged = df_year if merged is None else pd.merge(merged, df_year, on="facility_id", how="outer")

    merged[YEARS] = merged[YEARS].fillna(0)
    return merged


def attach_landfill_info(yearly_df: pd.DataFrame, landfill_info_path: Path) -> pd.DataFrame:
    landfill_info = pd.read_csv(landfill_info_path, encoding="latin-1")

    if "facility_id" not in landfill_info.columns:
        raise ValueError("'facility_id' not found in landfill information file.")

    output = pd.merge(landfill_info, yearly_df, on="facility_id", how="left")
    output[YEARS] = output[YEARS].fillna(0)
    return output


def calculate_landfill_accumulation(yearly_pfas_df: pd.DataFrame) -> pd.DataFrame:
    output = yearly_pfas_df.copy()
    output[YEARS] = output[YEARS].fillna(0).cumsum(axis=1)
    return output


def calculate_landfill_mass_from_volume(yearly_volume_df: pd.DataFrame) -> pd.DataFrame:
    output = yearly_volume_df.copy()
    output[YEARS] = output[YEARS].fillna(0).cumsum(axis=1)
    output[YEARS] = output[YEARS] * CARPET_MASS_FACTOR
    return output


def align_lg_to_landfills(landfill_mass_df: pd.DataFrame, landfill_lg_path: Path) -> pd.DataFrame:
    lg_df = pd.read_csv(landfill_lg_path)

    if "facility_id" not in lg_df.columns:
        raise ValueError("'facility_id' not found in Landfill_LG.csv.")

    missing_years = [y for y in YEARS if y not in lg_df.columns]
    if missing_years:
        raise ValueError(f"Missing year columns in Landfill_LG.csv: {missing_years[:5]}...")

    lg_aligned = pd.merge(
        landfill_mass_df[["facility_id"]],
        lg_df[["facility_id"] + YEARS],
        on="facility_id",
        how="left",
    )

    lg_aligned[YEARS] = lg_aligned[YEARS].fillna(0)
    return lg_aligned


def calculate_landfill_leachate_components(
    landfill_mass_df: pd.DataFrame,
    landfill_lg_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lg_aligned = align_lg_to_landfills(landfill_mass_df, landfill_lg_path)

    leachate_np = landfill_mass_df.copy()
    leachate_pfaa = landfill_mass_df.copy()
    leachate_total = landfill_mass_df.copy()

    leachate_np[YEARS] = (
        landfill_mass_df[YEARS]
        * lg_aligned[YEARS]
        * XT_NP
        / SA
        * LG_DAYS
        / 1_000_000
        * LEACHATE_ACTIVE_FRACTION
    )

    leachate_pfaa[YEARS] = (
        landfill_mass_df[YEARS]
        * lg_aligned[YEARS]
        * XT_PFAA
        / SA
        * LG_DAYS
        / 1_000_000
        * LEACHATE_ACTIVE_FRACTION
    )

    leachate_total[YEARS] = leachate_np[YEARS] + leachate_pfaa[YEARS]

    return leachate_np, leachate_pfaa, leachate_total


def calculate_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output[YEARS] = output[YEARS].fillna(0).cumsum(axis=1)
    return output


def calculate_landfill_accumulation_without_leachate(
    landfill_accum_df: pd.DataFrame,
    leachate_accum_total_df: pd.DataFrame,
) -> pd.DataFrame:
    output = landfill_accum_df.copy()

    leachate_aligned = pd.merge(
        output[["facility_id"]],
        leachate_accum_total_df[["facility_id"] + YEARS],
        on="facility_id",
        how="left",
    )

    leachate_aligned[YEARS] = leachate_aligned[YEARS].fillna(0)
    output[YEARS] = output[YEARS].fillna(0) - leachate_aligned[YEARS]

    return output


def run_landfill_central_bau(project_root: Path, landfill_type: str = "MLF") -> None:
    transport_dir = project_root / "outputs" / "tables" / "transportation_assignment"

    pfas_input_dir = transport_dir / "annual_PFAS_accum_central_BAU"
    volume_input_dir = transport_dir / "annual_carpt_volume_central_BAU"

    landfill_input_dir = project_root / "data" / "landfill_central_BAU"

    landfill_info_path = landfill_input_dir / "Landfill_Final_W_CountyInform.csv"
    landfill_lg_path = landfill_input_dir / "Landfill_LG.csv"

    output_dir = project_root / "outputs" / "tables" / "landfill_central_BAU"
    output_dir.mkdir(parents=True, exist_ok=True)

    yearly_pfas = read_yearly_assigned_tables(
        input_dir=pfas_input_dir,
        file_prefix="annual_PFAS_accum_central_BAU",
        landfill_type=landfill_type,
    )

    yearly_pfas = attach_landfill_info(yearly_pfas, landfill_info_path)
    yearly_pfas.to_csv(
        output_dir / f"landfill_yearly_PFAS_central_BAU_{landfill_type}.csv",
        index=False,
    )

    landfill_accum = calculate_landfill_accumulation(yearly_pfas)
    landfill_accum.to_csv(
        output_dir / f"landfill_accum_central_BAU_{landfill_type}.csv",
        index=False,
    )

    yearly_volume = read_yearly_assigned_tables(
        input_dir=volume_input_dir,
        file_prefix="annual_carpt_volume_central_BAU",
        landfill_type=landfill_type,
    )

    yearly_volume = attach_landfill_info(yearly_volume, landfill_info_path)
    yearly_volume.to_csv(
        output_dir / f"landfill_yearly_carpet_volume_central_BAU_{landfill_type}.csv",
        index=False,
    )

    landfill_mass = calculate_landfill_mass_from_volume(yearly_volume)
    landfill_mass.to_csv(
        output_dir / f"landfill_mass_accum_central_BAU_{landfill_type}.csv",
        index=False,
    )

    leachate_np, leachate_pfaa, leachate_total = calculate_landfill_leachate_components(
        landfill_mass_df=landfill_mass,
        landfill_lg_path=landfill_lg_path,
    )

    leachate_np.to_csv(
        output_dir / f"landfill_leachate_NP_central_BAU_{landfill_type}.csv",
        index=False,
    )
    leachate_pfaa.to_csv(
        output_dir / f"landfill_leachate_PFAA_central_BAU_{landfill_type}.csv",
        index=False,
    )
    leachate_total.to_csv(
        output_dir / f"landfill_leachate_total_central_BAU_{landfill_type}.csv",
        index=False,
    )

    leachate_accum_np = calculate_cumulative(leachate_np)
    leachate_accum_pfaa = calculate_cumulative(leachate_pfaa)
    leachate_accum_total = calculate_cumulative(leachate_total)

    leachate_accum_np.to_csv(
        output_dir / f"landfill_leachate_accum_NP_central_BAU_{landfill_type}.csv",
        index=False,
    )
    leachate_accum_pfaa.to_csv(
        output_dir / f"landfill_leachate_accum_PFAA_central_BAU_{landfill_type}.csv",
        index=False,
    )
    leachate_accum_total.to_csv(
        output_dir / f"landfill_leachate_accum_total_central_BAU_{landfill_type}.csv",
        index=False,
    )

    landfill_accum_without_leachate = calculate_landfill_accumulation_without_leachate(
        landfill_accum_df=landfill_accum,
        leachate_accum_total_df=leachate_accum_total,
    )

    landfill_accum_without_leachate.to_csv(
        output_dir / f"landfill_accum_without_leachate_central_BAU_{landfill_type}.csv",
        index=False,
    )

    print("Finished landfill PFAS central BAU calculation.")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    run_landfill_central_bau(project_root=project_root, landfill_type="MLF")