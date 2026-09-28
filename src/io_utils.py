from typing import Union
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = {
    "year",
    "us_pop",
    "Percent_nylon",
    "V_import",
    "V_export",
    "V_production",
    "V_sales_nylon",
    "RR_total",
    "IncR",
}


def load_basic_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.copy()
    df["year"] = df["year"].astype(int)
    return df


def load_commercial_share(
    csv_path: Union[str, Path],
    start_year: int = 1990,
    end_year: int = 2060,
) -> pd.Series:
    """Commercial share phi_k of residential + commercial carpet sales, indexed by year."""
    df = pd.read_csv(csv_path)
    if "commercial_share" not in df.columns:
        raise ValueError(f"Column 'commercial_share' not found in {csv_path}")

    years = df["year"].astype(int).tolist()
    if years != list(range(start_year, end_year + 1)):
        raise ValueError(f"Expected contiguous years {start_year}-{end_year} in {csv_path}")

    phi = pd.Series(df["commercial_share"].to_numpy(dtype=float), index=years, name="commercial_share")
    phi.index.name = "year"
    if phi.isna().any():
        raise ValueError(f"NaN commercial_share in {csv_path}: years {phi[phi.isna()].index.tolist()}")
    if not ((phi > 0) & (phi < 1)).all():
        raise ValueError(f"commercial_share outside (0, 1) in {csv_path}")
    return phi


SQFT_TO_M2 = 0.09290304

# CBECS 1992, Table A1 (million ft2): all buildings, and floorspace built 1990-1992.
CBECS_1992_TOTAL_FLOORSPACE = 67876
CBECS_1992_BUILT_1990_1992 = 2502

# Expected commercial floor area in 1990 (million m2), used as a guard on the input file.
EXPECTED_COMMERCIAL_FLOORSPACE_1990 = 4403.7
FLOORSPACE_REL_TOL = 1e-3


def load_cbecs_commercial_floorspace_1990(csv_path: Union[str, Path]) -> float:
    """
    Carpet-relevant commercial floor area in 1990 (million m2), from CBECS 1992:
    floorspace of building activities with treatment == "included", converted to m2 and
    scaled to the share of 1992 floorspace built before 1990.
    """
    df = pd.read_csv(csv_path)

    total = df.loc[df["treatment"] == "total", "floorspace_million_sqft"]
    if len(total) != 1 or float(total.iloc[0]) != CBECS_1992_TOTAL_FLOORSPACE:
        raise ValueError(
            f"Expected one 'total' row of {CBECS_1992_TOTAL_FLOORSPACE} million ft2 in {csv_path}"
        )

    included_sqft = float(df.loc[df["treatment"] == "included", "floorspace_million_sqft"].sum())
    pre_1990_share = (CBECS_1992_TOTAL_FLOORSPACE - CBECS_1992_BUILT_1990_1992) / CBECS_1992_TOTAL_FLOORSPACE
    floorspace_m2 = included_sqft * SQFT_TO_M2 * pre_1990_share

    rel_diff = abs(floorspace_m2 / EXPECTED_COMMERCIAL_FLOORSPACE_1990 - 1)
    if rel_diff > FLOORSPACE_REL_TOL:
        raise ValueError(
            f"Commercial floor area 1990 = {floorspace_m2:.1f} million m2, expected "
            f"{EXPECTED_COMMERCIAL_FLOORSPACE_1990} within {FLOORSPACE_REL_TOL:.1%}"
        )
    return floorspace_m2


def save_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)