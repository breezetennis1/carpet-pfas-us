"""
Extract the Hazus Inventory National Database tract-level
"Square Footage by Occupancy" table to CSV (R2.1 / E7).

Source : FEMA Hazus Inventory National Database (May 2023; NSI 2022, 2020 tracts)
         https://www.loc.gov/item/2024595033
Units  : source is thousand sq ft (fields <OCCUPANCY>F, e.g. COM4F);
         output is converted to square metres (fields <OCCUPANCY>_m2, e.g. COM4_m2)
Needs  : pyogrio (preferred) or fiona
Usage  : python extract_hazus_sqft_tract.py
         python extract_hazus_sqft_tract.py --gdb <path> --out-dir <path>
"""
import argparse
import datetime as dt
import hashlib
from pathlib import Path

import pandas as pd

DEFAULT_GDB = Path(r"C:\Users\mchen48\Downloads\master-gdc-gdcdatasets-2024595033-2024595033"
                   r"\2024595033\National.gdb")
DEFAULT_OUT = Path("data/raw/hazus")
SOURCE_URL = "https://www.loc.gov/item/2024595033"
KSQFT_TO_M2 = 1000 * 0.09290304  # 1 ft = 0.3048 m exactly -> 1 thousand sq ft = 92.90304 m2

# Fields expected in "Square Footage by Occupancy at Census Tract Level" (data dictionary 6.10)
OCC = ["AGR1", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "COM10",
       "EDU1", "EDU2", "GOV1", "GOV2", "IND1", "IND2", "IND3", "IND4", "IND5", "IND6", "REL1",
       "RES1", "RES2", "RES3A", "RES3B", "RES3C", "RES3D", "RES3E", "RES3F",
       "RES4", "RES5", "RES6"]
SQFT_FIELDS = [f"{o}F" for o in OCC]
M2_FIELDS = [f"{o}_m2" for o in OCC]
VALID_STATE_FIPS = {f"{i:02d}" for i in [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19,
                                         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                         34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48,
                                         49, 50, 51, 53, 54, 55, 56]}  # 50 states + DC


def list_layers(gdb):
    try:
        import pyogrio
        return [str(row[0]) for row in pyogrio.list_layers(str(gdb))]
    except ImportError:
        import fiona
        return fiona.listlayers(str(gdb))


def layer_fields(gdb, layer):
    try:
        import pyogrio
        info = pyogrio.read_info(str(gdb), layer=layer)
        return [str(f) for f in info["fields"]], int(info["features"])
    except ImportError:
        import fiona
        with fiona.open(str(gdb), layer=layer) as src:
            return list(src.schema["properties"].keys()), len(src)


def read_table(gdb, layer):
    try:
        import pyogrio
        df = pyogrio.read_dataframe(str(gdb), layer=layer, read_geometry=False)
    except ImportError:
        import fiona
        with fiona.open(str(gdb), layer=layer) as src:
            df = pd.DataFrame([dict(f["properties"]) for f in src])
    return pd.DataFrame(df).drop(columns="geometry", errors="ignore")


def find_tract_sqft_layer(gdb):
    """Identify the table by its fields, not by name: has 'Tract' and all <OCC>F fields."""
    hits = []
    for lyr in list_layers(gdb):
        fields, n = layer_fields(gdb, lyr)
        if "Tract" in fields and "CensusBlock" not in fields and set(SQFT_FIELDS) <= set(fields):
            hits.append((lyr, n))
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly 1 tract square-footage table, found: {hits}")
    return hits[0]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gdb", type=Path, default=DEFAULT_GDB)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if not args.gdb.exists():
        raise FileNotFoundError(args.gdb)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Layers in gdb:")
    for lyr in list_layers(args.gdb):
        print("  ", lyr)

    layer, n_src = find_tract_sqft_layer(args.gdb)
    print(f"\nUsing layer: {layer} ({n_src:,} rows)")

    df = read_table(args.gdb, layer)
    df = df[["Tract"] + SQFT_FIELDS].copy()
    df["Tract"] = df["Tract"].astype(str).str.strip().str.zfill(11)
    df.insert(1, "county_fips", df["Tract"].str[:5])
    df.insert(1, "state_fips", df["Tract"].str[:2])

    # thousand sq ft -> m2 (float64 to avoid float32 rounding from the source "Single" type)
    df[SQFT_FIELDS] = df[SQFT_FIELDS].astype("float64") * KSQFT_TO_M2
    df = df.rename(columns=dict(zip(SQFT_FIELDS, M2_FIELDS)))

    # ---------------- checks ----------------
    issues = []
    if len(df) != n_src:
        issues.append(f"row count {len(df)} != source {n_src}")
    if df["Tract"].str.len().ne(11).any() or not df["Tract"].str.isdigit().all():
        issues.append("non-11-digit / non-numeric Tract IDs present")
    dup = df["Tract"].duplicated().sum()
    if dup:
        issues.append(f"{dup} duplicated Tract IDs")
    na = int(df[M2_FIELDS].isna().sum().sum())
    neg = int((df[M2_FIELDS] < 0).sum().sum())
    if na:
        issues.append(f"{na} missing values in floor-area fields")
    if neg:
        issues.append(f"{neg} negative values in floor-area fields")
    missing_states = sorted(VALID_STATE_FIPS - set(df["state_fips"]))
    if missing_states:
        issues.append(f"missing state FIPS: {missing_states}")
    other = sorted(set(df["state_fips"]) - VALID_STATE_FIPS)  # territories, kept but flagged

    # national totals (billion m2) for plausibility; CBECS 2018 commercial ~96.4 billion sq ft ~ 8.96 billion m2
    grp = {
        "RES1-3 (Berrill-comparable)": [f for f in M2_FIELDS if f[:4] in ("RES1", "RES2", "RES3")],
        "RES4-6": ["RES4_m2", "RES5_m2", "RES6_m2"],
        "COM": [f for f in M2_FIELDS if f.startswith("COM")],
        "EDU+GOV+REL": ["EDU1_m2", "EDU2_m2", "GOV1_m2", "GOV2_m2", "REL1_m2"],
        "IND+AGR": [f for f in M2_FIELDS if f.startswith(("IND", "AGR"))],
    }
    states_dc = df[df["state_fips"].isin(VALID_STATE_FIPS)]
    totals = {k: states_dc[v].to_numpy().sum() / 1e9 for k, v in grp.items()}

    # ---------------- write ----------------
    out_csv = args.out_dir / "hazus_floor_area_m2_by_occupancy_tract.csv"
    df.to_csv(out_csv, index=False)
    csv_sha = sha256_of(out_csv)

    lines = [
        "# Hazus tract-level square footage by occupancy (source record)",
        "",
        f"- Source: FEMA Hazus Inventory National Database, data dictionary dated May 2023 ({SOURCE_URL})",
        "- Basis: NSI 2022 building inventory; 2020 census tract boundaries",
        f"- Local source path: `{args.gdb}`",
        f"- Table (layer) name in National.gdb: `{layer}`",
        f"- Extracted: {dt.date.today().isoformat()} by extract_hazus_sqft_tract.py",
        f"- Output: `{out_csv.name}` ({len(df):,} rows), SHA-256 `{csv_sha}`",
        "- Units: source **thousand sq ft** (fields `<OCCUPANCY>F`), converted to **m2** (fields `<OCCUPANCY>_m2`) "
        f"with factor {KSQFT_TO_M2} m2 per thousand sq ft; added `state_fips`, `county_fips` = Tract[:2], Tract[:5]",
        f"- Non-state FIPS kept (territories): {other if other else 'none'}",
        "",
        "## National totals, 50 states + DC (billion m2; CBECS 2018 commercial ~8.96 billion m2 for comparison)",
        *[f"- {k}: {v:,.2f}" for k, v in totals.items()],
        "",
        "## Checks",
        *([f"- WARNING: {i}" for i in issues] or ["- all passed"]),
    ]
    (args.out_dir / "README_source.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
