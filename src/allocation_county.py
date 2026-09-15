from pathlib import Path
import numpy as np
import pandas as pd


YEAR_START = 1990
YEAR_END = 2060
YEAR_COLS = [str(y) for y in range(YEAR_START, YEAR_END + 1)]


def build_pc_floorarea(project_root: Path) -> pd.DataFrame:
    county_dir = project_root / "data" / "county"

    df_sf = pd.read_csv(county_dir / "OccStock_SF_m2.csv")
    df_mf = pd.read_csv(county_dir / "OccStock_MF_m2.csv")
    df_mh = pd.read_csv(county_dir / "OccStock_MH_m2.csv")

    df_contig_pop = pd.read_csv(
        county_dir / "Contiguous US population 2020-2060.csv"
    )

    df_county_pop = pd.read_csv(
        county_dir / "US population 1990-2060.csv"
    )

    df_county_info = pd.read_csv(
        county_dir / "County_W_CapArea.csv"
    )

    # Combine occupied residential floor area: SF + MF + MH
    df_occ_combined = df_sf.copy()

    for year in range(2020, 2061):
        y = str(year)
        df_occ_combined[y] = df_sf[y] + df_mf[y] + df_mh[y]

    # Per capita floor area for 2020-2060
    df_pc_floorarea_contig = df_occ_combined.copy()

    for year in range(2020, 2061):
        y = str(year)
        df_pc_floorarea_contig[y] = (
            df_occ_combined[y] / df_contig_pop[y]
        )

    # For 1990-2019, use 2020 per capita floor area
    insert_pos = 1
    for year in range(1990, 2020):
        df_pc_floorarea_contig.insert(
            insert_pos,
            str(year),
            df_pc_floorarea_contig["2020"]
        )
        insert_pos += 1

    # Merge to all counties
    pc_floorarea = pd.DataFrame()
    pc_floorarea["GeoID"] = df_county_pop["GeoID"]

    pc_floorarea = pd.merge(
        pc_floorarea,
        df_pc_floorarea_contig,
        how="left",
        on="GeoID"
    )

    # Fill counties without matched floor area using annual mean
    for year in YEAR_COLS:
        pc_floorarea[year] = pc_floorarea[year].fillna(
            pc_floorarea[year].mean()
        )

    # Add County and State columns
    pc_floorarea.insert(1, "County", df_county_info["County"])
    pc_floorarea.insert(2, "State", df_county_info["State"])

    # Save generated intermediate file
    pc_floorarea.to_csv(
        county_dir / "PC_FloorArea_generated.csv",
        index=False
    )

    return pc_floorarea


def allocate_central_bau_to_counties(project_root: Path) -> None:
    county_dir = project_root / "data" / "county"

    national_csv = (
        project_root
        / "outputs"
        / "tables"
        / "national"
        / "TOTAL_CENTRAL_ALL_SUBCLASSES_BAU.csv"
    )

    output_dir = (
        project_root
        / "outputs"
        / "tables"
        / "county"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    national = pd.read_csv(national_csv)
    county_pop = pd.read_csv(county_dir / "US population 1990-2060.csv")

    pc_floorarea = build_pc_floorarea(project_root)

    # National results are kg and kg/year
    national = national[["year", "F_stock", "F6"]].copy()
    national_stock = national.set_index("year")["F_stock"]
    national_emission = national.set_index("year")["F6"]

    # County total floor area = county population × per capita floor area
    county_floorarea = county_pop.copy()

    for year in YEAR_COLS:
        county_floorarea[year] = county_pop[year] * pc_floorarea[year]

    total_floorarea = county_floorarea[YEAR_COLS].sum(axis=0)

    county_ratio = county_floorarea.copy()

    for year in YEAR_COLS:
        county_ratio[year] = county_floorarea[year] / total_floorarea[year]

    # Allocate national results to counties
    county_stock = county_pop.copy()
    county_emission = county_pop.copy()

    for year in range(YEAR_START, YEAR_END + 1):
        y = str(year)

        county_stock[y] = county_ratio[y] * national_stock.loc[year]
        county_emission[y] = county_ratio[y] * national_emission.loc[year]

    # Units:
    # county_stock = kg
    # county_emission = kg/year

    # Per capita results
    pc_stock = county_stock.copy()
    pc_emission = county_emission.copy()

    for year in YEAR_COLS:
        pop = county_pop[year].replace(0, np.nan)

        # kg -> g/cap
        pc_stock[year] = county_stock[year] * 1000 / pop

        # kg/year -> mg/cap/year
        pc_emission[year] = county_emission[year] * 1_000_000 / pop

    # Save outputs
    county_stock.to_csv(
        output_dir / "County_InUseStock_CENTRAL_BAU_kg.csv",
        index=False
    )

    county_emission.to_csv(
        output_dir / "County_InUseEmission_CENTRAL_BAU_kg_year.csv",
        index=False
    )

    pc_stock.to_csv(
        output_dir / "County_PC_InUseStock_CENTRAL_BAU_g_cap.csv",
        index=False
    )

    pc_emission.to_csv(
        output_dir / "County_PC_InUseEmission_CENTRAL_BAU_mg_cap_year.csv",
        index=False
    )

    # Validation check
    check = pd.DataFrame({"year": range(YEAR_START, YEAR_END + 1)})

    check["national_F_stock_kg"] = check["year"].map(national_stock)
    check["allocated_F_stock_kg"] = [
        county_stock[str(y)].sum()
        for y in range(YEAR_START, YEAR_END + 1)
    ]

    check["national_F6_kg_year"] = check["year"].map(national_emission)
    check["allocated_F6_kg_year"] = [
        county_emission[str(y)].sum()
        for y in range(YEAR_START, YEAR_END + 1)
    ]

    check["stock_difference_kg"] = (
        check["allocated_F_stock_kg"]
        - check["national_F_stock_kg"]
    )

    check["emission_difference_kg_year"] = (
        check["allocated_F6_kg_year"]
        - check["national_F6_kg_year"]
    )

    check.to_csv(
        output_dir / "County_Allocation_Check_CENTRAL_BAU.csv",
        index=False
    )

    print("County allocation finished.")
    print("Saved outputs to:", output_dir)
    print("Max stock difference kg:", check["stock_difference_kg"].abs().max())
    print("Max emission difference kg/year:", check["emission_difference_kg_year"].abs().max())