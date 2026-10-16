"""
Residential vs commercial floor-area rankings, county and tract level (R2.1 / E7).

Test A (what was asked): are the top-20 units by residential area the same as by commercial area?
Test B (what the allocation actually depends on): do the top-20 units by allocated carpet change
        when the residential-only key is replaced by the composite key
            s_i = (1 - phi) * R_i / sum(R) + phi * C_i / sum(C)
        and how is r_i = (C_i/R_i) / (sum C / sum R) distributed?  s_i / s_i^R = 1 + phi * (r_i - 1).

Residential R : Hazus RES1-3 (default; same source and geography as C), or the model's own
                Berrill-based key via --res-county / --res-tract (CSV with an ID column and an area column).
Commercial C  : C_included_m2.
Scope         : 50 states + DC.

Usage : python compare_res_com_rankings.py
        python compare_res_com_rankings.py --phi 0.30 --top 20
        python compare_res_com_rankings.py --res-county model_R_county.csv --res-county-cols county_fips R_m2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/hazus/hazus_floor_area_m2_by_occupancy_tract.csv")
TRACT_C = Path("data/processed/census_commercial_floorarea_2022.csv")
COUNTY_C = Path("data/processed/county_commercial_floorarea_2022.csv")
OUT_DIR = Path("results/robustness_commercial")
RES_COLS = [f"{o}_m2" for o in ["RES1", "RES2", "RES3A", "RES3B", "RES3C", "RES3D", "RES3E", "RES3F"]]
C_COLS = ["C_included_m2"]
TERRITORIES = {"60", "66", "69", "72", "78"}


def load_res(level, custom, cols):
    """Residential area by unit: Hazus RES1-3 unless a model file is supplied."""
    if custom is not None:
        idc, valc = cols
        r = pd.read_csv(custom, dtype={idc: str})[[idc, valc]]
        r.columns = ["id", "R_m2"]
        r["id"] = r["id"].str.zfill(5 if level == "county" else 11)
        return r, f"model ({Path(custom).name}:{valc})"
    raw = pd.read_csv(RAW, dtype={"Tract": str, "state_fips": str, "county_fips": str},
                      usecols=["Tract", "state_fips", "county_fips"] + RES_COLS)
    # rebuild IDs from Tract so leading zeros cannot be lost upstream
    raw["Tract"] = raw["Tract"].str.strip().str.zfill(11)
    raw["state_fips"], raw["county_fips"] = raw["Tract"].str[:2], raw["Tract"].str[:5]
    raw = raw[~raw["state_fips"].isin(TERRITORIES)]
    raw["R_m2"] = raw[RES_COLS].sum(axis=1)
    key = "county_fips" if level == "county" else "Tract"
    r = raw.groupby(key, as_index=False)["R_m2"].sum().rename(columns={key: "id"})
    return r, "Hazus RES1-3"


def load_com(level):
    if level == "county":
        c = pd.read_csv(COUNTY_C, dtype={"county_fips": str, "state_fips": str})
        c["id"] = c["county_fips"].str.strip().str.zfill(5)
        c["state_fips"] = c["id"].str[:2]
    else:
        c = pd.read_csv(TRACT_C, dtype={"Tract": str, "state_fips": str, "county_fips": str},
                        usecols=["Tract", "state_fips"] + C_COLS)
        c["id"] = c["Tract"].str.strip().str.zfill(11)
        c["state_fips"] = c["id"].str[:2]
    return c[~c["state_fips"].isin(TERRITORIES)][["id"] + C_COLS]


def topn(s, n):
    return list(s.sort_values(ascending=False).index[:n])


def compare(level, r, c, phi, n, res_label):
    m = r.merge(c, on="id", how="outer", indicator=True)
    unmatched = m["_merge"].value_counts().to_dict()
    for side in ("left_only", "right_only"):
        ids = m.loc[m["_merge"] == side, "id"].head(10).tolist()
        if ids:
            print(f"  [{level}] {side} examples (residential-only / commercial-only): {ids}")
    m = m[m["_merge"] == "both"].drop(columns="_merge").set_index("id")
    m = m[m["R_m2"] > 0]  # composite ratio undefined without residential area

    rows, tables = [], {}
    for ccol in C_COLS:
        tag = ccol.replace("C_", "").replace("_m2", "")
        R, C = m["R_m2"], m[ccol]
        sR = R / R.sum()
        sC = C / C.sum()
        s_comp = (1 - phi) * sR + phi * sC
        ratio = s_comp / sR                     # = 1 + phi * (r_i - 1)
        r_i = (C / R) / (C.sum() / R.sum())

        # Test A: residential vs commercial ranking
        tR, tC = topn(R, n), topn(C, n)
        # Test B: residential-only vs composite allocation ranking
        tS = topn(s_comp, n)
        rows.append({
            "level": level, "commercial": tag, "residential": res_label, "phi": phi, "units": len(m),
            f"A_top{n}_overlap_R_vs_C": len(set(tR) & set(tC)),
            "A_spearman_R_vs_C": R.corr(C, method="spearman"),
            f"B_top{n}_overlap_Rkey_vs_composite": len(set(tR) & set(tS)),
            "B_spearman_Rkey_vs_composite": sR.corr(s_comp, method="spearman"),
            "r_P10": r_i.quantile(0.10), "r_median": r_i.median(), "r_P90": r_i.quantile(0.90),
            "r_CV": r_i.std() / r_i.mean(),
            "share_change_P10_%": 100 * (ratio.quantile(0.10) - 1),
            "share_change_median_%": 100 * (ratio.median() - 1),
            "share_change_P90_%": 100 * (ratio.quantile(0.90) - 1),
            "abs_share_change_>10%_units": int((ratio.sub(1).abs() > 0.10).sum()),
            "abs_share_change_>25%_units": int((ratio.sub(1).abs() > 0.25).sum()),
        })
        tables[tag] = pd.DataFrame({
            "rank": range(1, n + 1),
            "top_by_residential": tR,
            "top_by_commercial": tC,
            "commercial_in_res_top": [x in tR for x in tC],
            "top_by_composite_allocation": tS,
            "composite_in_res_top": [x in tR for x in tS],
            "composite_share_change_%": [100 * (ratio[x] - 1) for x in tS],
        })
    return rows, tables, unmatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi", type=float, default=0.30, help="commercial share of carpet (FCW end-use split)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--res-county", default=None)
    ap.add_argument("--res-county-cols", nargs=2, default=["county_fips", "R_m2"])
    ap.add_argument("--res-tract", default=None)
    ap.add_argument("--res-tract-cols", nargs=2, default=["Tract", "R_m2"])
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for level, custom, cols in [("county", args.res_county, args.res_county_cols),
                                ("tract", args.res_tract, args.res_tract_cols)]:
        r, label = load_res(level, custom, cols)
        c = load_com(level)
        rows, tables, unmatched = compare(level, r, c, args.phi, args.top, label)
        all_rows += rows
        print(f"\n[{level}] residential = {label}; merge: {unmatched}")
        for tag, t in tables.items():
            f = OUT_DIR / f"top{args.top}_{level}_{tag}.csv"
            t.to_csv(f, index=False)
            print(f"  wrote {f}")

    summary = pd.DataFrame(all_rows)
    f = OUT_DIR / "ranking_summary.csv"
    summary.to_csv(f, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.precision", 3):
        print("\n", summary.T)
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()
