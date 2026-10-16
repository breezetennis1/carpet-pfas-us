"""
Aggregate tract-level commercial floor area to county (R2.1 / E7).

Input : census_commercial_floorarea_2022.csv   (from build_commercial_floorarea_tract.py)
Output: county_commercial_floorarea_2022.csv
        county_fips, state_fips, n_tracts, C_included_m2
County FIPS follow 2020 census geography (e.g. Connecticut's 8 legacy counties, not planning regions).

Usage : python build_commercial_floorarea_county.py
        python build_commercial_floorarea_county.py --in <csv> --out <csv>
"""
import argparse
from pathlib import Path

import pandas as pd

DEFAULT_IN = Path("data/processed/census_commercial_floorarea_2022.csv")
DEFAULT_OUT = Path("data/processed/county_commercial_floorarea_2022.csv")
VALUE_COLS = ["C_included_m2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    tr = pd.read_csv(args.inp, dtype={"Tract": str, "state_fips": str, "county_fips": str},
                     usecols=["Tract", "state_fips", "county_fips"] + VALUE_COLS)

    cty = (tr.groupby(["county_fips", "state_fips"], as_index=False)
             .agg(n_tracts=("Tract", "size"), **{c: (c, "sum") for c in VALUE_COLS}))

    # checks: totals preserved, one state per county
    for c in VALUE_COLS:
        assert abs(cty[c].sum() - tr[c].sum()) <= 1e-6 * max(1.0, tr[c].sum()), f"{c} total not preserved"
    assert cty["county_fips"].is_unique, "a county_fips maps to more than one state_fips"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cty.to_csv(args.out, index=False)

    states_dc = cty[~cty["state_fips"].isin(["60", "66", "69", "72", "78"])]
    print(f"Wrote {args.out}  ({len(cty):,} counties; {len(states_dc):,} in 50 states + DC)")
    for c in VALUE_COLS:
        print(f"National {c:<17}: {states_dc[c].sum() / 1e9:.2f} billion m2 "
              f"| counties with 0: {(states_dc[c] == 0).sum()}")
    ct = sorted(states_dc.loc[states_dc["state_fips"] == "09", "county_fips"])
    print(f"Connecticut county FIPS in file: {ct}")


if __name__ == "__main__":
    main()
