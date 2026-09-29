from pathlib import Path
import numpy as np
import pandas as pd


YEAR_START = 1990
YEAR_END = 2060
YEAR_COLS = [str(y) for y in range(YEAR_START, YEAR_END + 1)]


PROJECTION_YEARS = [str(y) for y in range(2020, YEAR_END + 1)]

COUNTY_POP_CSV = "US population 1990-2060_v2025.csv"
CONTIG_POP_CSV = "Contiguous US population 2020-2060_v2025.csv"  # key column "County" holds the GeoID


def read_by_geoid(path: Path, key: str = "GeoID") -> pd.DataFrame:
    """Read a county table indexed by its integer GeoID; all joins below are by GeoID, never by row order."""
    df = pd.read_csv(path).rename(columns={key: "GeoID"})
    df["GeoID"] = df["GeoID"].astype(int)
    if not df["GeoID"].is_unique:
        raise ValueError(f"Duplicate GeoIDs in {path.name}")
    return df.set_index("GeoID")


def require_geoids(table: pd.DataFrame, geoids: pd.Index, name: str) -> None:
    missing = geoids.difference(table.index)
    if len(missing):
        raise ValueError(f"{name}: {len(missing)} counties missing, e.g. {missing[:10].tolist()}")


def build_pc_floorarea(project_root: Path) -> pd.DataFrame:
    county_dir = project_root / "data" / "county"

    df_sf = read_by_geoid(county_dir / "OccStock_SF_m2.csv")
    df_mf = read_by_geoid(county_dir / "OccStock_MF_m2.csv")
    df_mh = read_by_geoid(county_dir / "OccStock_MH_m2.csv")
    df_contig_pop = read_by_geoid(county_dir / CONTIG_POP_CSV, key="County")
    df_county_pop = read_by_geoid(county_dir / COUNTY_POP_CSV)
    df_county_info = read_by_geoid(county_dir / "County_W_CapArea.csv")

    # Floor-area counties; every other input must cover them
    geoids = df_sf.index
    for name, table in [("OccStock_MF", df_mf), ("OccStock_MH", df_mh), ("contiguous population", df_contig_pop),
                        ("county population", df_county_pop)]:
        require_geoids(table, geoids, name)
    require_geoids(df_county_info, df_county_pop.index, "County_W_CapArea")

    # Combine occupied residential floor area: SF + MF + MH
    df_occ_combined = (
        df_sf[PROJECTION_YEARS]
        + df_mf.loc[geoids, PROJECTION_YEARS]
        + df_mh.loc[geoids, PROJECTION_YEARS]
    )

    # Per capita floor area for 2020-2060
    df_pc_floorarea_contig = df_occ_combined / df_contig_pop.loc[geoids, PROJECTION_YEARS]

    # For 1990-2019, use 2020 per capita floor area
    for year in range(1990, 2020):
        df_pc_floorarea_contig[str(year)] = df_pc_floorarea_contig["2020"]
    df_pc_floorarea_contig = df_pc_floorarea_contig[YEAR_COLS]

    # All counties in the population file; counties without floor area (Alaska, Hawaii) get the annual mean
    pc_floorarea = df_pc_floorarea_contig.reindex(df_county_pop.index)
    pc_floorarea = pc_floorarea.fillna(pc_floorarea.mean())

    # Add County and State columns
    pc_floorarea.insert(0, "County", df_county_info.loc[pc_floorarea.index, "County"])
    pc_floorarea.insert(1, "State", df_county_info.loc[pc_floorarea.index, "State"])
    pc_floorarea = pc_floorarea.reset_index()

    # Save generated intermediate file
    pc_floorarea.to_csv(
        county_dir / "PC_FloorArea_generated.csv",
        index=False
    )

    return pc_floorarea


# Streams: residential (R) allocated by residential floor area, commercial (C) by commercial
# floor area; the R + C total keeps the original file names.
STREAM_SUFFIXES = {"TOTAL": "", "R": "_R", "C": "_C"}

COMMERCIAL_FLOORAREA_CSV = Path("data") / "processed" / "county_commercial_floorarea_2022.csv"

METRICS = {
    "F_stock": ("County_InUseStock_CENTRAL_BAU_kg", "County_PC_InUseStock_CENTRAL_BAU_g_cap", 1000),
    "F6": ("County_InUseEmission_CENTRAL_BAU_kg_year", "County_PC_InUseEmission_CENTRAL_BAU_mg_cap_year", 1_000_000),
}


def load_contiguous_geoids(project_root: Path) -> pd.Series:
    """The 3108 contiguous-US counties covered by the residential floor-area data."""
    return read_by_geoid(project_root / "data" / "county" / "OccStock_SF_m2.csv").index


def build_residential_share(project_root: Path, county_pop: pd.DataFrame) -> pd.DataFrame:
    """
    Residential share per county and year (unchanged method): county population x per capita
    residential floor area, normalised over the counties in county_pop (indexed by GeoID).
    """
    pc_floorarea = build_pc_floorarea(project_root).set_index("GeoID")
    require_geoids(pc_floorarea, county_pop.index, "per capita floor area")

    floorarea = county_pop[YEAR_COLS] * pc_floorarea.loc[county_pop.index, YEAR_COLS]
    return floorarea / floorarea.sum(axis=0)


def build_commercial_share(project_root: Path, geoids: pd.Index) -> pd.Series:
    """
    Commercial share per county from the 2022 commercial floor area (single year, so the
    shares are held constant for 1990-2060), normalised over the given counties.
    """
    com = read_by_geoid(project_root / COMMERCIAL_FLOORAREA_CSV, key="county_fips")
    require_geoids(com, geoids, "commercial floor area")

    floorarea = com.loc[geoids, "C_included_m2"]
    if floorarea.isna().any() or (floorarea < 0).any():
        raise ValueError("Missing or negative commercial floor area")
    return floorarea / floorarea.sum()


def load_national(project_root: Path, suffix: str) -> pd.DataFrame:
    """National central BAU all-subclass totals (kg, kg/year) for one stream."""
    path = (
        project_root / "outputs" / "tables" / "national"
        / f"TOTAL_CENTRAL_ALL_SUBCLASSES_BAU{suffix}.csv"
    )
    return pd.read_csv(path).set_index("year")[list(METRICS)]


def allocate_central_bau_to_counties(project_root: Path) -> None:
    county_dir = project_root / "data" / "county"
    output_dir = project_root / "outputs" / "tables" / "county"
    output_dir.mkdir(parents=True, exist_ok=True)

    geoids = load_contiguous_geoids(project_root)

    # All tables are indexed by GeoID; the output rows follow the population file order
    county_pop_all = read_by_geoid(county_dir / COUNTY_POP_CSV)
    require_geoids(county_pop_all, geoids, "county population")
    county_pop = county_pop_all[county_pop_all.index.isin(geoids)]

    years = list(range(YEAR_START, YEAR_END + 1))
    commercial_share = build_commercial_share(project_root, county_pop.index)
    shares = {
        "R": build_residential_share(project_root, county_pop),
        "C": pd.DataFrame({y: commercial_share for y in YEAR_COLS}),
    }
    national = {stream: load_national(project_root, suffix) for stream, suffix in STREAM_SUFFIXES.items()}

    # Allocate: county value = stream share x national stream value (kg, kg/year)
    allocated = {"R": {}, "C": {}, "TOTAL": {}}
    for metric in METRICS:
        for stream in ["R", "C"]:
            values = shares[stream].values * national[stream].loc[years, metric].values
            allocated[stream][metric] = pd.DataFrame(values, index=shares[stream].index, columns=YEAR_COLS)
        allocated["TOTAL"][metric] = allocated["R"][metric] + allocated["C"][metric]

    # Save county and per capita outputs with the original layout (GeoID, StID, County,State, years)
    id_cols = county_pop.drop(columns=YEAR_COLS)
    pop = county_pop[YEAR_COLS].replace(0, np.nan)

    for stream, suffix in STREAM_SUFFIXES.items():
        for metric, (county_name, pc_name, pc_factor) in METRICS.items():
            county_values = allocated[stream][metric]
            per_capita = county_values * pc_factor / pop  # kg -> g/cap; kg/year -> mg/cap/year

            for name, table in [(county_name, county_values), (pc_name, per_capita)]:
                if not table.index.sort_values().equals(id_cols.index.sort_values()):
                    raise ValueError(f"{name}{suffix}: counties do not match the population file")
                out = id_cols.join(table[YEAR_COLS], how="left").reset_index()
                out.to_csv(output_dir / f"{name}{suffix}.csv", index=False)

    # Validation
    check_all = validate_allocation(project_root, county_pop_all, geoids, shares, national, allocated)
    for stream, suffix in STREAM_SUFFIXES.items():
        check_all[stream].to_csv(output_dir / f"County_Allocation_Check_CENTRAL_BAU{suffix}.csv", index=False)

    print("County allocation finished.")
    print("Saved outputs to:", output_dir)


def validate_allocation(project_root, county_pop_all, geoids, shares, national, allocated) -> dict:
    years = list(range(YEAR_START, YEAR_END + 1))
    ok = True

    def report(label, value, tol):
        nonlocal ok
        passed = value <= tol
        ok = ok and passed
        print(f"  {label}: {value:.3e} (tol {tol:g}) -> {'PASS' if passed else 'FAIL'}")

    print("Validation")
    print(f"  counties: {len(geoids)} contiguous, unique GeoIDs = {geoids.is_unique}")
    for stream in ["R", "C"]:
        s = shares[stream]
        report(f"{stream} shares: max |sum over counties - 1|", float((s.sum(axis=0) - 1).abs().max()), 1e-12)
        report(f"{stream} shares: negative or NaN entries", float(((s < 0) | s.isna()).to_numpy().sum()), 0)
    report("C shares identical in every year", float(shares["C"].sub(shares["C"]["1990"], axis=0).abs().to_numpy().max()), 0)

    # Residential method unchanged: the new R shares are the original 3142-county shares, renormalised
    pc_floorarea = build_pc_floorarea(project_root).set_index("GeoID")
    pop_all = county_pop_all
    fa_all = pop_all[YEAR_COLS] * pc_floorarea.loc[pop_all.index, YEAR_COLS]
    old_share = fa_all / fa_all.sum(axis=0)
    renorm = old_share.loc[geoids] / old_share.loc[geoids].sum(axis=0)
    report("R shares vs original 3142-county shares renormalised to 3108: max rel diff",
           float((shares["R"].loc[geoids] / renorm - 1).abs().to_numpy().max()), 1e-12)
    kept = old_share.loc[geoids].sum(axis=0)
    print(f"  original residential share of the 3108 counties: {kept.min():.5f} - {kept.max():.5f} "
          f"(Alaska + Hawaii {1 - kept.max():.4%} - {1 - kept.min():.4%} now allocated to the contiguous US)")

    checks = {}
    for stream in ["R", "C", "TOTAL"]:
        check = pd.DataFrame({"year": years})
        for metric, unit in [("F_stock", "kg"), ("F6", "kg_year")]:
            nat = national[stream].loc[years, metric].to_numpy()
            alloc = allocated[stream][metric][YEAR_COLS].sum(axis=0).to_numpy()
            check[f"national_{metric}_{unit}"] = nat
            check[f"allocated_{metric}_{unit}"] = alloc
            check[f"{metric}_difference_{unit}"] = alloc - nat
            check[f"{metric}_rel_difference"] = (alloc - nat) / nat
            report(f"{stream:5s} {metric:7s}: max |allocated - national| / national",
                   float(np.abs((alloc - nat) / nat).max()), 1e-12)
        checks[stream] = check

    # R + C = TOTAL at county level, and national R + C = national TOTAL
    for metric in METRICS:
        t = allocated["TOTAL"][metric]
        diff = (allocated["R"][metric] + allocated["C"][metric] - t).abs().to_numpy().max()
        report(f"county R + C - TOTAL, {metric}: max abs (kg)", float(diff), 0)
        nat_diff = (national["R"][metric] + national["C"][metric] - national["TOTAL"][metric]).abs()
        report(f"national R + C vs TOTAL file, {metric}: max rel",
               float((nat_diff / national["TOTAL"][metric].abs()).max()), 1e-12)
        report(f"negative county values, {metric}", float((t < 0).to_numpy().sum()), 0)

    print(f"Validation: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    if not ok:
        raise RuntimeError("County allocation validation failed")
    return checks


