"""
Build tract-level commercial (carpet-relevant) floor area from the Hazus extract (R2.1 / E7).

Input : hazus_floor_area_m2_by_occupancy_tract.csv  (from extract_hazus_sqft_tract.py; m2)
Output: census_commercial_floorarea_2022.csv
        Tract, state_fips, county_fips, the included occupancy columns (m2), and
          C_included_m2 = sum of INCLUDED classes
Excluded classes (COM2 warehouse, COM10 parking, IND1-6, AGR1) and residential RES1-3
are not written. Floor-area reference year: NSI 2022 basis (Hazus Inventory National Database, FEMA 2023).

Usage : python build_commercial_floorarea_tract.py
        python build_commercial_floorarea_tract.py --in <csv> --out <csv>
"""
import argparse
from pathlib import Path

import pandas as pd

DEFAULT_IN = Path("data/raw/hazus/hazus_floor_area_m2_by_occupancy_tract.csv")
DEFAULT_OUT = Path("data/processed/census_commercial_floorarea_2022.csv")

# Carpet-relevant classification (judgement; see SI). Edit here only.
INCLUDED = ["COM1", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",  # retail, personal/repair, offices, banks,
                                                                           # hospitals, clinics, entertainment, theaters
            "EDU1", "EDU2",                                                # schools, colleges/universities
            "GOV1", "GOV2", "REL1",                                        # government offices, emergency response, churches
            "RES4", "RES5", "RES6"]                                        # hotels, dormitories, nursing homes
EXCLUDED = ["COM2", "COM10", "IND1", "IND2", "IND3", "IND4", "IND5", "IND6", "AGR1"]
RESIDENTIAL = ["RES1", "RES2", "RES3A", "RES3B", "RES3C", "RES3D", "RES3E", "RES3F"]  # allocated via Berrill key

ID_COLS = ["Tract", "state_fips", "county_fips"]


def col(occ):
    return f"{occ}_m2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    df = pd.read_csv(args.inp, dtype={c: str for c in ID_COLS})

    # every occupancy class must be classified exactly once
    all_occ = INCLUDED + EXCLUDED + RESIDENTIAL
    assert len(all_occ) == len(set(all_occ)), "a class is listed in more than one group"
    data_occ = {c[:-3] for c in df.columns if c.endswith("_m2")}
    unclassified = data_occ - set(all_occ)
    missing = set(all_occ) - data_occ
    if unclassified or missing:
        raise ValueError(f"unclassified: {sorted(unclassified)}; missing in CSV: {sorted(missing)}")

    keep = [col(o) for o in INCLUDED]
    out = df[ID_COLS + keep].copy()
    out["C_included_m2"] = out[keep].sum(axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # summary (50 states + DC only)
    states_dc = out[~out["state_fips"].isin(["60", "66", "69", "72", "78"])]
    print(f"Wrote {args.out}  ({len(out):,} tracts; {states_dc['county_fips'].nunique():,} counties in 50 states + DC)")
    print(f"National C_included : {states_dc['C_included_m2'].sum() / 1e9:.2f} billion m2")
    print(f"Tracts with C_included = 0: {(states_dc['C_included_m2'] == 0).sum():,}")
    print("\nBy class (billion m2):")
    for o in INCLUDED:
        print(f"  {o:<5} {states_dc[col(o)].sum() / 1e9:6.2f}")


if __name__ == "__main__":
    main()
