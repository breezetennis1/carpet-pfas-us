"""
Landfill-level PFAS accumulation and leachate (central BAU), from the census tract -> landfill
transport assignment (MLF files; waste routed through transfer stations is already included).

Leachate follows the national model (src/model.py, add_leachate_block): first order in the PFAS
inventory of each landfill, with the landfill's own leachate generation rate LG_i(t):
    k_L,i(t)   = [xt x LG_i(t) / SA x 365 / 1e6 x release fraction] / [PFAS per tonne of new carpet]
    leachate_i(t) = k_L,i(t) x inventory_i(t-1)
    inventory_i(t) = inventory_i(t-1) + deposit_i(t) - leachate_i(t)
NP and PFAA deposits of each landfill are its total deposit x the national subclass share of the
landfilled PFAS in that year (SP has no leachate).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CARPET_CONFIGS, MODEL_CONSTANTS, PFAS_CONFIGS


START_YEAR = 1990
END_YEAR = 2060
YEARS = [str(y) for y in range(START_YEAR, END_YEAR + 1)]

ESTIMATE = "central"
SUBCLASSES = ["NP", "PFAA", "SP"]
LEACHATE_SUBCLASSES = [s for s in SUBCLASSES if PFAS_CONFIGS[(s, ESTIMATE)].has_leachate]

LANDFILL_DIR = Path("data") / "landfill_central_BAU"
PFAS_PREFIX = "annual_PFAS_accum_central_BAU"        # PFAS landfilled, kg/yr
AREA_PREFIX = "annual_carpt_volume_central_BAU"      # carpet AREA landfilled, million m2/yr

TOL = 1e-9


def read_assigned_tables(input_dir: Path, file_prefix: str, landfill_type: str = "MLF") -> pd.DataFrame:
    """Landfill x year table from the yearly assignment files; the value column is named by the year."""
    columns = []
    for year in YEARS:
        file_path = input_dir / f"{file_prefix}_{year}_{landfill_type}_pfas_assigned.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")
        df = pd.read_csv(file_path, dtype={"facility_id": str})
        if df.columns[-1] != year:
            raise ValueError(f"Last column of {file_path.name} is '{df.columns[-1]}', expected '{year}'")
        if not df["facility_id"].is_unique:
            raise ValueError(f"Duplicate facility_id in {file_path.name}")
        columns.append(df.set_index("facility_id")[year])
    return pd.concat(columns, axis=1).fillna(0.0)


def read_landfill_info(project_root: Path) -> tuple:
    info = pd.read_csv(project_root / LANDFILL_DIR / "Landfill_Final_W_CountyInform.csv",
                       encoding="latin-1", dtype={"facility_id": str})
    lg = pd.read_csv(project_root / LANDFILL_DIR / "Landfill_LG.csv", dtype={"facility_id": str})
    for name, df in [("landfill info", info), ("Landfill_LG", lg)]:
        if not df["facility_id"].is_unique:
            raise ValueError(f"Duplicate facility_id in {name}")
    return info.set_index("facility_id"), lg.set_index("facility_id")[YEARS].astype(float)


def national_landfill_inputs(project_root: Path) -> tuple:
    """National R + C landfilled PFAS and carpet area per year, and subclass shares of the landfilled PFAS."""
    national_dir = project_root / "outputs" / "tables" / "national"
    total = pd.read_csv(national_dir / "TOTAL_CENTRAL_ALL_SUBCLASSES_BAU_updated.csv").set_index("year")
    by_subclass = pd.DataFrame({
        s: pd.read_csv(national_dir / f"{s}_{ESTIMATE.upper()}_BAU.csv").set_index("year")["F_landfill_yearly"]
        for s in SUBCLASSES
    })
    shares = by_subclass.div(by_subclass.sum(axis=1), axis=0)
    shares.index = shares.index.astype(str)
    total.index = total.index.astype(str)
    return total, shares


def leachate_rate(subclass: str, lg: pd.DataFrame) -> pd.DataFrame:
    """First-order leachate rate k_L (1/yr) per landfill and year, as in the national model but with local LG."""
    c = MODEL_CONSTANTS
    cfg = PFAS_CONFIGS[(subclass, ESTIMATE)]
    carpet = CARPET_CONFIGS[ESTIMATE]
    leachate_per_tonne = float(cfg.xt) * lg / c.SA * 365 / 1_000_000 * c.leachate_release_fraction
    pfas_per_tonne_new = cfg.P_production * c.initial_stock_retention / carpet.carpet_density / 1000
    k = leachate_per_tonne / pfas_per_tonne_new
    if ((k < 0) | (k >= 1)).any().any():
        raise ValueError(f"{subclass}: leachate rate outside [0, 1)")
    return k


def first_order_inventory(deposits: pd.DataFrame, k: pd.DataFrame) -> tuple:
    """Landfill PFAS inventory and leachate (rate applied to the inventory at the start of the year)."""
    inventory = pd.DataFrame(0.0, index=deposits.index, columns=YEARS)
    leachate = pd.DataFrame(0.0, index=deposits.index, columns=YEARS)
    prev = np.zeros(len(deposits))
    for year in YEARS:
        leach = k[year].to_numpy() * prev
        prev = prev + deposits[year].to_numpy() - leach
        leachate[year] = leach
        inventory[year] = prev
    return inventory, leachate


def run_landfill_central_bau(project_root: Path, transport_dir: Path, landfill_type: str = "MLF") -> None:
    """
    transport_dir: folder with the collaborator's assignment results, containing the subfolders
    annual_PFAS_accum_central_BAU/ and annual_carpt_volume_central_BAU/.
    """
    output_dir = project_root / "outputs" / "tables" / "landfill_central_BAU"
    output_dir.mkdir(parents=True, exist_ok=True)

    info, lg_all = read_landfill_info(project_root)
    deposits = read_assigned_tables(transport_dir / PFAS_PREFIX, PFAS_PREFIX, landfill_type)
    area = read_assigned_tables(transport_dir / AREA_PREFIX, AREA_PREFIX, landfill_type)

    # All landfills of the information table (zeros where nothing is landfilled)
    for name, table in [("PFAS assignment", deposits), ("area assignment", area)]:
        unknown = table.index.difference(info.index)
        if len(unknown):
            raise ValueError(f"{name}: {len(unknown)} facility_id not in the landfill information table, e.g. {unknown[:5].tolist()}")
    missing_lg = info.index.difference(lg_all.index)
    if len(missing_lg):
        raise ValueError(f"No LG for {len(missing_lg)} landfills, e.g. {missing_lg[:5].tolist()}")
    deposits = deposits.reindex(info.index).fillna(0.0)[YEARS]
    area = area.reindex(info.index).fillna(0.0)[YEARS]
    lg = lg_all.loc[info.index, YEARS]

    national, shares = national_landfill_inputs(project_root)

    # Subclass deposits, inventories and leachate
    leachate = {}
    inventory_total = pd.DataFrame(0.0, index=info.index, columns=YEARS)
    for subclass in SUBCLASSES:
        dep_s = deposits * shares.loc[YEARS, subclass].to_numpy()
        if subclass in LEACHATE_SUBCLASSES:
            inv_s, leachate[subclass] = first_order_inventory(dep_s, leachate_rate(subclass, lg))
        else:
            inv_s = dep_s.cumsum(axis=1)
        inventory_total += inv_s
    leachate_total = sum(leachate.values())

    carpet_mass = area.cumsum(axis=1) * CARPET_CONFIGS[ESTIMATE].carpet_density  # million m2 x g/m2 = tonnes

    def save(table: pd.DataFrame, name: str) -> None:
        info.join(table).reset_index().to_csv(output_dir / f"{name}_central_BAU_{landfill_type}.csv", index=False)

    save(deposits, "landfill_yearly_PFAS")                         # kg/yr
    save(deposits.cumsum(axis=1), "landfill_accum")                # kg, cumulative deposits
    save(area, "landfill_yearly_carpet_volume")                    # million m2/yr (carpet area)
    save(carpet_mass, "landfill_mass_accum")                       # tonnes carpet
    for subclass in LEACHATE_SUBCLASSES:
        save(leachate[subclass], f"landfill_leachate_{subclass}")  # kg/yr
        save(leachate[subclass].cumsum(axis=1), f"landfill_leachate_accum_{subclass}")
    save(leachate_total, "landfill_leachate_total")
    save(leachate_total.cumsum(axis=1), "landfill_leachate_accum_total")
    save(inventory_total, "landfill_accum_without_leachate")      # kg, inventory net of leachate

    qa = validate(deposits, area, inventory_total, leachate_total, national, project_root)
    qa.to_csv(output_dir / f"landfill_QA_check_central_BAU_{landfill_type}.csv", index_label="year")

    print("Finished landfill PFAS central BAU calculation.")
    print(f"Output folder: {output_dir}")


def validate(deposits, area, inventory, leachate, national, project_root) -> pd.DataFrame:
    ok = True

    def report(label, value, tol):
        nonlocal ok
        passed = value <= tol
        ok = ok and passed
        print(f"  {label}: {value:.3e} (tol {tol:g}) -> {'PASS' if passed else 'FAIL'}")

    national_dir = project_root / "outputs" / "tables" / "national"
    nat_leach = pd.read_csv(national_dir / "TOTAL_CENTRAL_ALL_SUBCLASSES_BAU.csv").set_index("year")["PFAS_landfill_leachate"]
    nat_leach.index = nat_leach.index.astype(str)

    qa = pd.DataFrame(index=YEARS)
    qa["national_F_landfill_yearly_kg"] = national.loc[YEARS, "F_landfill_yearly"]
    qa["landfills_F_landfill_yearly_kg"] = deposits.sum()
    qa["national_V_annual_landfill_million_m2"] = national.loc[YEARS, "V_annual_landfill"]
    qa["landfills_V_annual_landfill_million_m2"] = area.sum()
    qa["national_leachate_kg"] = nat_leach.loc[YEARS]
    qa["landfills_leachate_kg"] = leachate.sum()
    qa["leachate_ratio_landfills_to_national"] = qa["landfills_leachate_kg"] / qa["national_leachate_kg"].where(qa["national_leachate_kg"] > 0)

    print("Validation")
    print(f"  landfills: {len(deposits)} in the information table, {int((deposits.sum(axis=1) > 0).sum())} receive carpet")
    report("sum over landfills vs national landfilled PFAS: max |rel diff|",
           float((qa["landfills_F_landfill_yearly_kg"] / qa["national_F_landfill_yearly_kg"] - 1).abs().max()), TOL)
    report("sum over landfills vs national landfilled carpet area: max |rel diff|",
           float((qa["landfills_V_annual_landfill_million_m2"] / qa["national_V_annual_landfill_million_m2"] - 1).abs().max()), TOL)

    prev = inventory.shift(1, axis=1).fillna(0.0)
    balance = (inventory - prev) - (deposits - leachate)
    report("landfill mass balance: max |residual| / max cumulative deposit",
           float(balance.abs().to_numpy().max() / deposits.cumsum(axis=1).to_numpy().max()), TOL)
    report("leachate > inventory of the previous year (count)", float((leachate > prev * (1 + 1e-12) + 1e-12).to_numpy().sum()), 0)
    report("negative inventory or leachate (count)", float(((inventory < 0) | (leachate < 0)).to_numpy().sum()), 0)
    report("cumulative leachate > cumulative deposits (count)",
           float((leachate.cumsum(axis=1) > deposits.cumsum(axis=1) * (1 + 1e-12)).to_numpy().sum()), 0)

    ratio = qa["leachate_ratio_landfills_to_national"].dropna()
    print(f"  leachate, sum over landfills / national (differs through local LG vs national LG "
          f"{MODEL_CONSTANTS.LG}): {ratio.min():.3f} - {ratio.max():.3f}; 2020 {ratio.get('2020', np.nan):.3f}, 2060 {ratio.get('2060', np.nan):.3f}")
    print(f"Validation: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    if not ok:
        raise RuntimeError("Landfill PFAS validation failed")
    return qa


if __name__ == "__main__":
    raise SystemExit("Run 05_run_landfill_PFAS_calculation.py (it sets the transport assignment folder).")
