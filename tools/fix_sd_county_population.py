#!/usr/bin/env python3
r"""
Fix a one-row shift in the South Dakota county populations for 1990-2009 in

    data/county/US population 1990-2060.csv          (source of the v2025 file for 1990-2019)
    data/county/US population 1990-2060_v2025.csv    (used by src/allocation_county.py)

Shannon County (46113) was renamed Oglala Lakota County (46102) in 2015. In both files the
1990-2009 values of six South Dakota counties sit one row off: each row holds the series of the
next county in the old FIPS order, and Oglala Lakota's own series sits in the Sanborn row, e.g.
Pennington (46103) has 2,943 people in 2009 and 101,253 in 2010. The rows are:

    row 46102 (Oglala Lakota) holds Pennington (46103)      row 46107 (Potter)  holds Roberts       (46109)
    row 46103 (Pennington)    holds Perkins    (46105)      row 46109 (Roberts) holds Sanborn       (46111)
    row 46105 (Perkins)       holds Potter     (46107)      row 46111 (Sanborn) holds Oglala Lakota (46102)

The 1990 values then match the 1990 Census counts (Pennington 81,343; Shannon 9,902; Perkins
3,932; Potter 3,190; Roberts 9,914; Sanborn 2,833), and every 2009 -> 2010 step is continuous.
2010-2060 are correct and unchanged. The script only moves values between these six rows, so the
state and national totals are unchanged in every year.

The script is idempotent: it checks the 2009 -> 2010 continuity first and does nothing if the
file is already fixed. The original file is copied to data/county/backup/ before the first change.

Run from the repository root:

    python tools\fix_sd_county_population.py
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

FILES = ["US population 1990-2060.csv", "US population 1990-2060_v2025.csv"]
SHIFTED_YEARS = [str(y) for y in range(1990, 2010)]

# county: the row that currently holds its 1990-2009 series
ROW_HOLDS = {
    46103: 46102,  # Pennington    <- values in the Oglala Lakota row
    46105: 46103,  # Perkins       <- Pennington row
    46107: 46105,  # Potter        <- Perkins row
    46109: 46107,  # Roberts       <- Potter row
    46111: 46109,  # Sanborn       <- Roberts row
    46102: 46111,  # Oglala Lakota <- Sanborn row
}
CONTINUITY = (0.8, 1.25)  # plausible 2009 -> 2010 population ratio


def continuous(df: pd.DataFrame, mapping: dict) -> bool:
    ratios = [df.loc[county, "2010"] / df.loc[row, "2009"] for county, row in mapping.items()]
    return all(CONTINUITY[0] <= r <= CONTINUITY[1] for r in ratios)


def fix_file(path: Path) -> None:
    df = pd.read_csv(path)
    df["GeoID"] = df["GeoID"].astype(int)
    if not df["GeoID"].is_unique:
        raise ValueError(f"Duplicate GeoIDs in {path.name}")
    df = df.set_index("GeoID")

    identity = {county: county for county in ROW_HOLDS}
    if continuous(df, identity):
        print(f"{path.name}: already consistent, nothing changed")
        return
    if not continuous(df, ROW_HOLDS):
        raise RuntimeError(f"{path.name}: the South Dakota rows do not show the expected shift; not changed")

    backup = path.parent / "backup" / f"{path.stem}_pre_SD_fix.csv"
    backup.parent.mkdir(exist_ok=True)
    if not backup.exists():
        shutil.copy2(path, backup)

    sd_total_before = df.loc[df.index // 1000 == 46, SHIFTED_YEARS].sum()
    fixed = df.copy()
    for county, row in ROW_HOLDS.items():
        fixed.loc[county, SHIFTED_YEARS] = df.loc[row, SHIFTED_YEARS].to_numpy()

    assert continuous(fixed, identity)
    assert fixed.loc[fixed.index // 1000 == 46, SHIFTED_YEARS].sum().equals(sd_total_before)
    assert fixed.drop(columns=SHIFTED_YEARS).equals(df.drop(columns=SHIFTED_YEARS))

    fixed.reset_index().to_csv(path, index=False)
    print(f"{path.name}: fixed 1990-2009 for {len(ROW_HOLDS)} counties (backup: {backup.relative_to(path.parents[2])})")
    for county in sorted(ROW_HOLDS):
        print(f"    {county} {df.loc[county, 'County,State']:38s} 1990 {df.loc[county, '1990']:>7} -> {fixed.loc[county, '1990']:>7}"
              f" | 2009 {df.loc[county, '2009']:>7} -> {fixed.loc[county, '2009']:>7} | 2010 {fixed.loc[county, '2010']:>7}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path("."))
    args = p.parse_args()
    for name in FILES:
        fix_file(args.repo / "data" / "county" / name)


if __name__ == "__main__":
    main()
