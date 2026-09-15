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


def save_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)