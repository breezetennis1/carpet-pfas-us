from pathlib import Path
import numpy as np
import pandas as pd

from src.allocation_county import (
    COUNTY_POP_CSV,
    YEAR_COLS,
    build_residential_share,
    load_contiguous_geoids,
    read_by_geoid,
    require_geoids,
)


# Census tracts: 2020 boundaries for every year (Hazus floor area exists only for 2020 tracts).
BASE_YEAR = 2020
HAZUS_CSV = Path("data") / "raw" / "hazus" / "hazus_floor_area_m2_by_occupancy_tract.csv"
TRACT_COMMERCIAL_CSV = Path("data") / "processed" / "census_commercial_floorarea_2022.csv"
TRACT_POP_DIR = Path("data") / "census_tracts" / "population"

# Hazus residential occupancies comparable to Berrill SF + MH + MF (RES4-6 are in the commercial key)
RES_COLS = [f"{o}_m2" for o in ["RES1", "RES2", "RES3A", "RES3B", "RES3C", "RES3D", "RES3E", "RES3F"]]

STREAM_SUFFIXES = {"TOTAL": "", "R": "_R", "C": "_C"}
SUBCLASSES = ["NP", "PFAA", "SP"]

# National variables (units): F_stock kg, F6 kg/yr, F_landfill_yearly kg/yr,
# V_annual_landfill million m2 of carpet area landfilled per year
VARIABLES = ["F_stock", "F6", "F_landfill_yearly", "V_annual_landfill"]
ID_COLS = ["GEO_ID", "CensusTract", "County", "State"]
PER_CAPITA = {"F_stock": ("F_stock_g_per_cap", 1000), "F6": ("F6_mg_per_cap_yr", 1_000_000)}  # from kg, kg/yr


def update_total_central_bau(project_root: Path, suffix: str = "") -> pd.DataFrame:
    """
    National central BAU inputs for one stream (suffix "", "_R", "_C"): F_stock and F6 from the
    all-subclass total, F_landfill_yearly = NP + PFAA + SP, and V_annual_landfill (carpet area,
    identical for every subclass that has a landfill block; taken from NP).
    Saved as TOTAL_CENTRAL_ALL_SUBCLASSES_BAU_updated{suffix}.csv.
    """
    national_dir = project_root / "outputs" / "tables" / "national"
    total_df = pd.read_csv(national_dir / f"TOTAL_CENTRAL_ALL_SUBCLASSES_BAU{suffix}.csv").set_index("year")
    subclass = {s: pd.read_csv(national_dir / f"{s}_CENTRAL_BAU{suffix}.csv").set_index("year") for s in SUBCLASSES}

    landfill_yearly = sum(subclass[s]["F_landfill_yearly"] for s in SUBCLASSES)
    area = subclass["NP"]["V_annual_landfill"]
    if not np.allclose(area, subclass["PFAA"]["V_annual_landfill"], rtol=1e-12, atol=0):
        raise ValueError("V_annual_landfill differs between NP and PFAA")

    total_updated = total_df.drop(columns=[c for c in ["F_landfill_yearly", "V_annual_landfill"] if c in total_df])
    total_updated["F_landfill_yearly"] = landfill_yearly
    total_updated["V_annual_landfill"] = area
    total_updated = total_updated.reset_index()

    out_file = national_dir / f"TOTAL_CENTRAL_ALL_SUBCLASSES_BAU_updated{suffix}.csv"
    total_updated.to_csv(out_file, index=False)
    print(f"Saved updated national file: {out_file}")
    return total_updated.set_index("year")


def read_tract_population(project_root: Path, year: int) -> pd.Series:
    df = pd.read_csv(project_root / TRACT_POP_DIR / f"CT_Population_{year}.csv", dtype={"GEO_ID": str})
    df.index = df["GEO_ID"].str[-11:]
    if not df.index.is_unique:
        raise ValueError(f"Duplicate tracts in CT_Population_{year}.csv")
    return df[str(year)].astype(float)


def load_tracts(project_root: Path, county_geoids: pd.Index) -> pd.DataFrame:
    """2020 tracts of the contiguous counties with Hazus residential and commercial floor area (m2)."""
    hz = pd.read_csv(project_root / HAZUS_CSV, dtype={"Tract": str, "state_fips": str, "county_fips": str})
    hz = hz[hz["Tract"].str.fullmatch(r"\d{11}")].set_index("Tract")  # drops the 2 'None' rows (state 00)
    com = pd.read_csv(project_root / TRACT_COMMERCIAL_CSV, dtype={"Tract": str, "state_fips": str, "county_fips": str})
    com = com[com["Tract"].str.fullmatch(r"\d{11}")].set_index("Tract")
    for name, table in [("Hazus", hz), ("commercial floor area", com)]:
        if not table.index.is_unique:
            raise ValueError(f"Duplicate tracts in {name} table")

    ids = pd.read_csv(project_root / TRACT_POP_DIR / f"CT_Population_{BASE_YEAR}.csv", dtype={"GEO_ID": str})
    ids.index = ids["GEO_ID"].str[-11:]

    tracts = pd.DataFrame(index=hz.index[hz.index.str[:5].astype(int).isin(county_geoids)])
    tracts["county"] = tracts.index.str[:5].astype(int)
    require_geoids(ids, tracts.index, f"CT_Population_{BASE_YEAR}")
    require_geoids(com, tracts.index, "commercial floor area")
    tracts = tracts.join(ids[ID_COLS])
    tracts["R_m2"] = hz.loc[tracts.index, RES_COLS].sum(axis=1)
    tracts["C_m2"] = com.loc[tracts.index, "C_included_m2"]

    missing_counties = county_geoids.difference(tracts["county"].unique())
    if len(missing_counties):
        raise ValueError(f"Counties without tracts: {missing_counties.tolist()[:10]}")
    if (tracts[["R_m2", "C_m2"]] < 0).any().any() or tracts[["R_m2", "C_m2"]].isna().any().any():
        raise ValueError("Negative or missing Hazus floor area")
    return tracts.sort_index()


def build_tract_population(project_root: Path, tracts: pd.DataFrame, county_pop: pd.DataFrame) -> tuple:
    """
    Tract population on 2020 tracts, scaled to the county population of each year (sum over the
    tracts of a county = county population). Before 2020 the 2020 tract distribution is used.
    Also reports the population-file tracts that are dropped because they are not in Hazus (water or
    special land-use tracts); their population is carried by the other tracts of the county.
    """
    raw = pd.DataFrame(index=tracts.index, columns=YEAR_COLS, dtype=float)
    dropped = {"max_share": 0.0, "tracts_with_population": set()}
    for year in range(BASE_YEAR, int(YEAR_COLS[-1]) + 1):
        pop = read_tract_population(project_root, year)
        require_geoids(pop, tracts.index, f"CT_Population_{year}")
        in_contiguous = pop.index.str[:5].astype(int).isin(county_pop.index)
        lost = pop[in_contiguous & ~pop.index.isin(tracts.index)]
        dropped["max_share"] = max(dropped["max_share"], float(lost.sum() / pop[in_contiguous].sum()))
        dropped["tracts_with_population"] |= set(lost.index[lost > 0])
        raw[str(year)] = pop.loc[tracts.index]
    for year in range(int(YEAR_COLS[0]), BASE_YEAR):
        raw[str(year)] = raw[str(BASE_YEAR)]

    county_sum = raw.groupby(tracts["county"]).sum()
    target = county_pop.loc[county_sum.index, YEAR_COLS]
    if ((county_sum == 0) & (target > 0)).any().any():
        raise ValueError("A county with population has zero tract population")
    factor = (target / county_sum.replace(0, np.nan)).fillna(0.0)
    population = raw * factor.loc[tracts["county"]].to_numpy()
    return population, dropped


def build_residential_within_county(tracts: pd.DataFrame, population: pd.DataFrame) -> tuple:
    """
    Within-county residential weight of each tract: P_t(y) x a_t, normalised within the county,
    with a_t = Hazus residential area / P_t(2020). Tracts with no 2020 population use the county
    average a_c = county Hazus residential area / county 2020 population.
    """
    p0 = population[str(BASE_YEAR)]
    county_a = tracts["R_m2"].groupby(tracts["county"]).sum() / p0.groupby(tracts["county"]).sum()
    a_t = (tracts["R_m2"] / p0.where(p0 > 0)).fillna(tracts["county"].map(county_a))
    weight = population.mul(a_t, axis=0)
    county_total = weight.groupby(tracts["county"]).sum()
    if (county_total == 0).any().any():
        raise ValueError("A county has zero residential weight in some year")
    within = weight / county_total.loc[tracts["county"]].to_numpy()
    return within, a_t


def allocate_national_to_census_tracts(project_root: Path) -> None:
    """
    Allocate the national central BAU results (residential R, commercial C, R + C) to 2020 census
    tracts of the contiguous US: one folder per variable, one CSV per year and stream, 1990-2060.
      R: national R x county residential share (county allocation) x within-county share
         (tract population x tract per capita Hazus residential floor area)
      C: national C x tract share of Hazus commercial floor area (static, 2022 snapshot)
    """
    county_dir = project_root / "data" / "county"
    output_dir = project_root / "outputs" / "tables" / "census_tracts"
    output_dir.mkdir(parents=True, exist_ok=True)

    national = {stream: update_total_central_bau(project_root, suffix) for stream, suffix in STREAM_SUFFIXES.items()}

    county_geoids = load_contiguous_geoids(project_root)
    county_pop_all = read_by_geoid(county_dir / COUNTY_POP_CSV)
    county_pop = county_pop_all[county_pop_all.index.isin(county_geoids)]

    tracts = load_tracts(project_root, county_pop.index)
    population, dropped = build_tract_population(project_root, tracts, county_pop)
    within, a_t = build_residential_within_county(tracts, population)

    # National shares of each tract (tract x year)
    county_share_R = build_residential_share(project_root, county_pop)
    share = {
        "R": within * county_share_R.loc[tracts["county"], YEAR_COLS].to_numpy(),
        "C": pd.DataFrame(
            np.repeat((tracts["C_m2"] / tracts["C_m2"].sum()).to_numpy()[:, None], len(YEAR_COLS), axis=1),
            index=tracts.index, columns=YEAR_COLS,
        ),
    }

    # Static weights table for documentation
    weights = tracts[["GEO_ID", "county", "R_m2", "C_m2"]].copy()
    weights["population_2020"] = population[str(BASE_YEAR)]
    weights["a_t_m2_per_cap"] = a_t
    weights["a_t_source"] = np.where(population[str(BASE_YEAR)] > 0, "tract", "county_mean")
    weights["C_share"] = share["C"][YEAR_COLS[0]]
    weights.to_csv(output_dir / "CT_allocation_weights_2020tracts.csv", index_label="Tract")

    qa_records = []
    county_sums = {stream: {v: {} for v in VARIABLES} for stream in STREAM_SUFFIXES}
    for year in YEAR_COLS:
        values = {}
        for stream in ["R", "C"]:
            values[stream] = pd.DataFrame(
                {v: share[stream][year] * national[stream].loc[int(year), v] for v in VARIABLES}
            )
        values["TOTAL"] = values["R"] + values["C"]

        pop = population[year].replace(0, np.nan)
        for stream, suffix in STREAM_SUFFIXES.items():
            for v in VARIABLES:
                nat = national[stream].loc[int(year), v]

                # One folder per variable, one file per year and stream (original layout + population)
                out = tracts[ID_COLS].copy()
                out["population"] = population[year]
                out["Portion"] = values[stream][v] / nat if nat else 0.0
                out[year] = values[stream][v]
                if v in PER_CAPITA:
                    name, factor = PER_CAPITA[v]
                    out[name] = values[stream][v] * factor / pop
                var_dir = output_dir / v
                var_dir.mkdir(exist_ok=True)
                out.to_csv(var_dir / f"CT_{v}_{year}{suffix}.csv", index=False)

                alloc = values[stream][v].sum()
                qa_records.append({
                    "year": int(year), "stream": stream, "variable": v,
                    "national_value": nat, "allocated_sum": alloc, "difference": alloc - nat,
                    "rel_difference": (alloc - nat) / nat if nat else np.nan,
                })
                county_sums[stream][v][year] = values[stream][v].groupby(tracts["county"]).sum()

    qa = pd.DataFrame(qa_records)
    qa.to_csv(output_dir / "CT_allocation_QA_check.csv", index=False)

    validate_tract_allocation(project_root, tracts, population, county_pop, within, share, qa, county_sums,
                              dropped, a_t)
    print(f"Saved {len(VARIABLES)} folders x {len(YEAR_COLS) * len(STREAM_SUFFIXES)} yearly tract files to {output_dir}")
    print("Finished census tract allocation.")


def validate_tract_allocation(project_root, tracts, population, county_pop, within, share, qa, county_sums,
                              dropped, a_t) -> None:
    ok = True

    def report(label, value, tol):
        nonlocal ok
        passed = value <= tol
        ok = ok and passed
        print(f"  {label}: {value:.3e} (tol {tol:g}) -> {'PASS' if passed else 'FAIL'}")

    print("Validation")
    p0 = population[str(BASE_YEAR)]
    print(f"  tracts: {len(tracts)} (2020 boundaries) in {tracts['county'].nunique()} contiguous counties")
    print(f"  tracts with population > 0 but no Hazus residential area (weight 0): "
          f"{int(((p0 > 0) & (tracts['R_m2'] == 0)).sum())}, {100 * p0[tracts['R_m2'] == 0].sum() / p0.sum():.3f} % of 2020 population")
    print(f"  tracts with zero 2020 population using the county-average a_c: {int((p0 == 0).sum())}")
    print(f"  population-file tracts not in Hazus with population > 0 (dropped; county total kept): "
          f"{sorted(dropped['tracts_with_population'])}")
    report("population of dropped tracts, max share of contiguous population", dropped["max_share"], 1e-6)

    county_of = tracts["county"]
    pop_sum = population.groupby(county_of).sum()
    report("tract population summed by county vs county population: max rel diff",
           float((pop_sum / county_pop.loc[pop_sum.index, YEAR_COLS] - 1).abs().max().max()), 1e-12)
    report("within-county residential shares sum to 1: max |sum - 1|",
           float((within.groupby(county_of).sum() - 1).abs().max().max()), 1e-12)
    for stream in ["R", "C"]:
        report(f"{stream} national shares sum to 1: max |sum - 1|", float((share[stream].sum() - 1).abs().max()), 1e-12)
        report(f"{stream} shares negative or NaN", float(((share[stream] < 0) | share[stream].isna()).to_numpy().sum()), 0)

    # Commercial floor area aggregates to the county file used by the county allocation
    cty_c = read_by_geoid(project_root / "data" / "processed" / "county_commercial_floorarea_2022.csv", key="county_fips")
    agg = tracts["C_m2"].groupby(county_of).sum()
    report("tract commercial area summed by county vs county commercial file: max rel diff",
           float((agg / cty_c.loc[agg.index, "C_included_m2"] - 1).abs().max()), 1e-12)

    report("sum over tracts vs national, all years, streams, variables: max |rel diff|",
           float(qa["rel_difference"].abs().max()), 1e-12)

    # Tract results summed by county equal the county allocation outputs (F_stock, F6)
    county_dir = project_root / "outputs" / "tables" / "county"
    files = {"F_stock": "County_InUseStock_CENTRAL_BAU_kg", "F6": "County_InUseEmission_CENTRAL_BAU_kg_year"}
    for stream, suffix in STREAM_SUFFIXES.items():
        for v, name in files.items():
            county = pd.read_csv(county_dir / f"{name}{suffix}.csv").set_index("GeoID")[YEAR_COLS]
            tract = pd.DataFrame(county_sums[stream][v])[YEAR_COLS]
            rel = (tract / county.loc[tract.index] - 1).abs().max().max()
            report(f"{stream:5s} {v:7s}: tracts summed by county vs county allocation, max rel diff", float(rel), 1e-12)

    print(f"Validation: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    if not ok:
        raise RuntimeError("Census tract allocation validation failed")
