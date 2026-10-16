#!/usr/bin/env python3
r"""
Update county population 2020-2025 with U.S. Census Bureau Vintage 2025 estimates and
re-anchor the 2026-2060 Hauer/SSP2 projection, for BOTH population files used by
src/allocation_county.py:

    data/county/US population 1990-2060.csv          3142 rows, key column "GeoID"
    data/county/Contiguous US population 2020-2060.csv   3108 rows, key column "County"
                                                         (which holds GeoID values)

Both must come from the same source, because allocation_county.py computes

    county_floorarea = county_pop x (occ_stock / contig_pop)

so the two population terms partly cancel. Updating only one would create a spurious
1-2% signal in per-capita floor area.

Source: https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/counties/totals/co-est2025-alldata.csv
Release March 2026, reference date 1 July.

Run from the repository root:

    python tools\update_county_population.py --repo . --reanchor
"""

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

URL = ("https://www2.census.gov/programs-surveys/popest/datasets/"
       "2020-2025/counties/totals/co-est2025-alldata.csv")
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
PROJ = [str(y) for y in range(2026, 2061)]

CT_LEGACY = [9001, 9003, 9005, 9007, 9009, 9011, 9013, 9015]
CT_PLANNING = [9110, 9120, 9130, 9140, 9150, 9160, 9170, 9180, 9190]

# Census FIPS that the model files may still carry under an older code.
# Keys are the CURRENT Census code, values the code used in the model files.
RECODE_TO_MODEL = {
    2063: 2261,   # Chugach Census Area  -> Valdez-Cordova (split 2019)
    2066: 2261,   # Copper River Census Area -> Valdez-Cordova
    2158: 2270,   # Kusilvak -> Wade Hampton (renamed 2015)
    46102: 46113,  # Oglala Lakota -> Shannon (renamed 2015)
}

FILES = [
    ("US population 1990-2060.csv", "GeoID"),
    ("Contiguous US population 2020-2060.csv", "County"),
]


def load_census(local):
    if local:
        raw = Path(local).read_bytes()
    else:
        import requests
        print("downloading %s" % URL)
        r = requests.get(URL, timeout=300)
        r.raise_for_status()
        raw = r.content
        Path("co-est2025-alldata.csv").write_bytes(raw)
        print("  saved co-est2025-alldata.csv (%.1f MB)" % (len(raw) / 1e6))
    df = pd.read_csv(io.StringIO(raw.decode("latin-1")), low_memory=False)
    df = df[df["SUMLEV"] == 50].copy()
    df["GeoID"] = df["STATE"] * 1000 + df["COUNTY"]
    df = df[["GeoID"] + ["POPESTIMATE%d" % y for y in YEARS]]
    df.columns = ["GeoID"] + [str(y) for y in YEARS]
    print("  %d counties in the Census file" % len(df))
    return df


def split_connecticut(census, shares):
    pr = census[census["GeoID"].isin(CT_PLANNING)]
    if not len(pr):
        print("  no CT planning regions found - leaving Connecticut as delivered")
        return census
    tot = pr[[str(y) for y in YEARS]].sum()
    rows = {"GeoID": CT_LEGACY}
    for y in YEARS:
        rows[str(y)] = [round(shares[g] * tot[str(y)]) for g in CT_LEGACY]
    print("  Connecticut: %d planning regions -> 8 legacy counties" % len(pr))
    for g in CT_LEGACY:
        print("     %d  share %.4f" % (g, shares[g]))
    census = census[~census["GeoID"].isin(CT_PLANNING)]
    return pd.concat([census, pd.DataFrame(rows)], ignore_index=True)


def ct_shares(path, key, share_year):
    df = pd.read_csv(path)
    sub = df[df[key].isin(CT_LEGACY)]
    if not len(sub):
        sys.exit("no Connecticut legacy counties found in %s" % path)
    if str(share_year) not in df.columns:
        sys.exit("column %d not in %s - pass a different --ct-share-year"
                 % (share_year, path))
    s = sub.set_index(key)[str(share_year)]
    return (s / s.sum()).to_dict()


def update_file(path, key, census, reanchor, cap, out_suffix, report_dir):
    df = pd.read_csv(path)
    n0 = len(df)
    lookup = census.set_index("GeoID")

    # align Census codes to whatever the model file uses
    ids = set(df[key])
    ren = {cur: old for cur, old in RECODE_TO_MODEL.items()
           if old in ids and cur not in ids}
    if ren:
        lookup = lookup.rename(index=ren)
        lookup = lookup.groupby(level=0).sum()
        print("  recoded to model FIPS: %s" % ren)

    have = [str(y) for y in YEARS if str(y) in df.columns]
    before = df.set_index(key)[have].copy()

    missing = sorted(ids - set(lookup.index))
    if missing:
        print("  NOT MATCHED (%d): %s" % (len(missing), missing[:15]))

    for y in have:
        s = df[key].map(lookup[y])
        df[y] = s.fillna(df[y]).round().astype("int64")

    if reanchor and "2025" in df.columns:
        proj = [c for c in PROJ if c in df.columns]
        f = (df["2025"] / before["2025"].reindex(df[key]).values)
        if cap > 0:
            hi, lo = cap, 1.0 / cap
            nclip = int(((f > hi) | (f < lo)).sum())
            f = f.clip(lo, hi)
            if nclip:
                print("  re-anchor: clipped %d counties to [%.3f, %.3f]" % (nclip, lo, hi))
        for c in proj:
            df[c] = (df[c] * f).round().astype("int64")
        q = f.quantile([0.01, 0.10, 0.50, 0.90, 0.99]).values
        print("  re-anchor factor  p1 %.4f  p10 %.4f  median %.4f  p90 %.4f  p99 %.4f" % tuple(q))

    out = path.with_name(path.stem + out_suffix + path.suffix)
    df.to_csv(out, index=False)

    rep = before.join(df.set_index(key)[have], lsuffix="_old", rsuffix="_new")
    for y in have:
        rep["d%s_pct" % y] = 100 * (rep["%s_new" % y] / rep["%s_old" % y] - 1)
    rname = report_dir / ("report_" + path.stem.replace(" ", "_") + ".csv")
    rep.to_csv(rname)

    print("  rows %d -> %d   wrote %s" % (n0, len(df), out.name))
    for y in have:
        o, n = before[y].sum() / 1e6, df[y].sum() / 1e6
        print("    %s  %8.3fM -> %8.3fM  %+6.2f%%" % (y, o, n, 100 * (n / o - 1)))
    if "2026" in df.columns:
        print("    2025->2026 continuity: %+.3f%%"
              % (100 * (df["2026"].sum() / df["2025"].sum() - 1)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=".", help="repository root")
    p.add_argument("--census-file", default=None,
                   help="local co-est2025-alldata.csv; downloaded if omitted")
    p.add_argument("--reanchor", action="store_true")
    p.add_argument("--reanchor-cap", type=float, default=0.0)
    p.add_argument("--ct-share-year", type=int, default=2019)
    p.add_argument("--suffix", default="_v2025")
    a = p.parse_args()

    root = Path(a.repo)
    cdir = root / "data" / "county"
    odir = root / "outputs"
    odir.mkdir(parents=True, exist_ok=True)

    census = load_census(a.census_file)
    shares = ct_shares(cdir / FILES[0][0], FILES[0][1], a.ct_share_year)
    census = split_connecticut(census, shares)

    for name, key in FILES:
        path = cdir / name
        if not path.exists():
            sys.exit("missing %s" % path)
        print("\n%s  (key column: %s)" % (name, key))
        update_file(path, key, census, a.reanchor, a.reanchor_cap, a.suffix, odir)

    print("\nBoth files written with suffix '%s'. Update src/allocation_county.py to "
          "read them, or rename once you are satisfied with the reports." % a.suffix)


if __name__ == "__main__":
    main()
