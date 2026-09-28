#!/usr/bin/env python3
r"""
Verification of the national-model mass-balance fixes, relative to baseline B1
(outputs/tables/national_baseline_pre_massfix/, commit e287d94, SP central P_import = 448).

Run from the repository root with the project environment:

    C:\Users\mchen48\anaconda_envs\carpet-pfas-rev\python.exe tools\verify_massfix.py --phase 1 --work-dir <scratch>

Model variants (src/model.py holds only the final code; the earlier steps are rebuilt here):
  B1 : model.py at e287d94, P_import(SP, central) = 448      -> must reproduce the B1 files exactly
  A  : model.py at e287d94, current config (P_import = 488)
  B  : A + phase-out applied to the net trade flow F1
  C  : B + first-order in-use emission             (= D with the F10 phase-out reverted)
  D  : C + phase-out applied to the post-sale protector input F10   (= E with the old leachate block)
  E  : D + mass-conserving landfill leachate (working-tree model.py) = final
All outputs are written to --work-dir only.
"""

import argparse
import contextlib
import dataclasses
import filecmp
import subprocess
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import src.model as model  # noqa: E402
import src.runner as runner  # noqa: E402
from src.config import CARPET_CONFIGS, MODEL_CONSTANTS, PFAS_CONFIGS  # noqa: E402
from src.io_utils import load_basic_data, load_cbecs_commercial_floorspace_1990, load_commercial_share  # noqa: E402
from src.postprocess import CORE_METRICS  # noqa: E402

sys.path.insert(0, str(REPO / "tools"))
from verify_rc_streams import check_t2, check_t4, read_csv  # noqa: E402

B1_COMMIT = "e287d94"
B1_DIR = REPO / "outputs" / "tables" / "national_baseline_pre_massfix"
SCENARIOS = ["BAU", "PHASE_OUT", "RECYCLE_24"]
STEPS = ["B1", "A", "B", "C", "D", "E"]
FINAL = "E"

F1_UNPHASED = '''    df["F1"] = (
        df["V_import"] * config.P_import * df["Percent_nylon"]
        - df["V_export"] * config.P_export * df["Percent_nylon"]
    ) / 1000'''
F1_PHASED = F1_UNPHASED.replace(") / 1000", ") * phase / 1000")

F10_PHASED = '''    # PFAS input from protector; protector products are phased out with the product
    phase = df["PhaseOut_effective"].to_numpy(dtype=float) if "PhaseOut_effective" in df.columns else 1.0
    df["F10"] = df["V_new_left"] * float(config.P_protector) * phase'''
F10_UNPHASED = '''    # PFAS input from protector
    df["F10"] = df["V_new_left"] * float(config.P_protector)'''

# Step E -> D: the no-leachate branch loses the deposited column, and the leachate block is restored to the
# carpet-mass formula. The E block runs from LEACH_E_START to LEACH_E_END (inclusive) in model.py.
LEACH_E_NOLEACH = '''        df["PFAS_landfill_leachate"] = 0.0
        df["F_accu_landfill_deposited"] = df["F_accu_landfill"].copy()
        return df'''
LEACH_D_NOLEACH = '''        df["PFAS_landfill_leachate"] = 0.0
        return df'''
LEACH_E_START = '    df["Mass_accum_landfill"] = df["V_accum_landfill"] * carpet_config.carpet_density  # tonnes carpet\n'
LEACH_E_END = '    df["F_accu_landfill"] = inventory  # landfill PFAS inventory net of leachate\n    return df\n'
LEACH_D_BLOCK = '''    df["Mass_accum_landfill"] = df["V_accum_landfill"] * carpet_config.carpet_density
    df["PFAS_landfill_leachate"] = (
        df["Mass_accum_landfill"]
        * float(config.xt)
        * c.LG
        / c.SA
        * 365
        / 1_000_000
        * c.leachate_release_fraction
    )
    return df
'''


def committed_model_source() -> str:
    return subprocess.run(
        ["git", "show", f"{B1_COMMIT}:src/model.py"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout


def patched(source: str, old: str, new: str, what: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"{what} block not found exactly once")
    return source.replace(old, new)


def revert_leachate(source: str) -> str:
    source = patched(source, LEACH_E_NOLEACH, LEACH_D_NOLEACH, "no-leachate branch")
    if source.count(LEACH_E_START) != 1 or source.count(LEACH_E_END) != 1:
        raise RuntimeError("leachate block markers not found exactly once")
    start = source.index(LEACH_E_START)
    end = source.index(LEACH_E_END) + len(LEACH_E_END)
    return source[:start] + LEACH_D_BLOCK + source[end:]


def build_model(step: str) -> types.ModuleType:
    if step == FINAL:
        return model
    if step in ("C", "D"):
        source = revert_leachate(Path(model.__file__).read_text(encoding="utf-8"))
        if step == "C":
            source = patched(source, F10_PHASED, F10_UNPHASED, "F10")
    else:
        source = committed_model_source()
    if step == "B":
        if source.count(F1_UNPHASED) != 1:
            raise RuntimeError("F1 block not found in the committed model.py")
        source = source.replace(F1_UNPHASED, F1_PHASED)
    module = types.ModuleType(f"model_step_{step}")
    exec(compile(source, f"model_step_{step}.py", "exec"), module.__dict__)
    return module


@contextlib.contextmanager
def use_step(step: str):
    """Route runner.run_national_model to the step's model; B1 also restores P_import(SP, central) = 448."""
    variant = build_model(step)
    original_model = runner.run_national_model
    original_cfg = PFAS_CONFIGS[("SP", "central")]
    runner.run_national_model = variant.run_national_model
    if step == "B1":
        PFAS_CONFIGS[("SP", "central")] = dataclasses.replace(original_cfg, P_import=448)
    try:
        yield variant
    finally:
        runner.run_national_model = original_model
        PFAS_CONFIGS[("SP", "central")] = original_cfg


def run_step(step: str, out_dir: Path) -> Path:
    original_dirs = runner.get_output_dirs
    runner.get_output_dirs = lambda root: {"tables_national": out_dir}
    try:
        with use_step(step), contextlib.redirect_stdout(None):
            for scenario in SCENARIOS:
                runner.run_all_cases(REPO, scenario=scenario)
    finally:
        runner.get_output_dirs = original_dirs
    return out_dir


def check_phase1(work_dir: Path) -> None:
    print(f"SP central P_import in working-tree config: {PFAS_CONFIGS[('SP', 'central')].P_import}")
    for step in STEPS:
        out = run_step(step, work_dir / f"step_{step}")
        n = len(list(out.glob("*.csv")))
        print(f"step {step}: {n} files written to {out}")

    b1 = work_dir / "step_B1"
    names = sorted(p.name for p in b1.glob("*.csv"))
    same = [n for n in names if filecmp.cmp(b1 / n, B1_DIR / n, shallow=False)]
    print(f"B1 reproduction: {len(same)} / {len(names)} files byte-identical to {B1_DIR.name}/")
    assert len(same) == len(names) == 120, "harness does not reproduce B1"


# ---------------------------------------------------------------------------------------------------------
# Phase 2: V1-V6 on step C
# ---------------------------------------------------------------------------------------------------------
STREAMS = {"TOTAL": "", "R": "_R", "C": "_C"}
V1_TOL = 1e-9
YEARS = [1990, 2000, 2010, 2020, 2040, 2060]
YEARS_HL = [2020, 2060]


def step_dir(work_dir: Path, step: str) -> Path:
    out = work_dir / f"step_{step}"
    if not (out / "R_C_share_summary_BAU.csv").exists():
        run_step(step, out)
    return out


def s0_pfas_by_stream() -> dict:
    """Initial PFAS stock (kg) per (subclass, estimate, stream), exactly as add_flow_side computes it."""
    base_df = load_basic_data(REPO / "data" / "Basic_data.csv")
    phi = load_commercial_share(REPO / runner.COMMERCIAL_SHARE_CSV)
    f_c = load_cbecs_commercial_floorspace_1990(REPO / runner.CBECS_FLOORSPACE_CSV)
    scales = runner.build_s0_scales(base_df, phi, f_c, runner.S0_C_METHOD)
    out = {}
    for (subclass, estimate), cfg in PFAS_CONFIGS.items():
        cc = CARPET_CONFIGS[estimate]
        s0_vol = model.compute_initial_stock_volume(model.fit_and_apply_nylon_share(base_df, cc), cc)
        per = {"R": scales["R"][estimate], "C": scales["C"][estimate]}
        per["TOTAL"] = per["R"] + per["C"]
        for stream, scale in per.items():
            out[(subclass, estimate, stream)] = (
                s0_vol * scale * cfg.P_production * MODEL_CONSTANTS.initial_stock_retention / 1000,
                s0_vol * scale,
            )
    return out


def iter_cases(folder: Path):
    for scenario in SCENARIOS:
        for subclass, estimate in PFAS_CONFIGS:
            for stream, suffix in STREAMS.items():
                name = f"{subclass}_{estimate.upper()}_{scenario}{suffix}.csv"
                yield scenario, subclass, estimate, stream, name, read_csv(folder / name)


def check_phase2(work_dir: Path, step: str = FINAL) -> None:
    final = step_dir(work_dir, step)
    print(f"Phase 2 checks on step {step}")
    s0 = s0_pfas_by_stream()
    ok = True

    # V1 mass balance, V2 non-negativity, V3 F6_t <= F_stock_{t-1}
    worst_v1, floor_cases, v2_viol, worst_v3 = (0.0, ""), [], [], (-np.inf, "")
    n_cases = 0
    for scenario, subclass, estimate, stream, name, df in iter_cases(final):
        n_cases += 1
        fs = df["F_stock"].to_numpy()
        prev = np.concatenate([[s0[(subclass, estimate, stream)][0]], fs[:-1]])
        flows = (df["F2"] + df["F10"] - df["F4"] - df["F6"]).to_numpy()
        resid = (fs - prev) - flows
        scale = max(np.abs(fs).max(), np.abs(prev).max())
        rel = np.abs(resid).max() / scale
        if rel > worst_v1[0]:
            worst_v1 = (rel, f"{name} year {int(df['year'][np.abs(resid).argmax()])}")
        if rel > V1_TOL:
            ok = False
        binds = (fs <= 0) | (resid > V1_TOL * scale)
        if binds.any():
            floor_cases.append(f"{name}: F_stock = 0 in {int(binds.sum())} years, first {int(df['year'][binds].iloc[0])}")
        for col in ["F4", "F6", "F_landfill_yearly", "F2"]:
            neg = df[col] < 0
            if neg.any():
                i = df[col].idxmin()
                v2_viol.append(f"{name}: {col} < 0 in {int(neg.sum())} years "
                               f"({int(df.loc[neg, 'year'].iloc[0])}-{int(df.loc[neg, 'year'].iloc[-1])}), "
                               f"min {df.loc[i, col]:.4g} kg in {int(df.loc[i, 'year'])}")
        margin = (df["F6"].to_numpy() - prev)
        if margin.max() > worst_v3[0]:
            worst_v3 = (margin.max(), f"{name} year {int(df['year'][margin.argmax()])}")

    print(f"V1  mass balance F_stock_t - F_stock_(t-1) = F2 + F10 - F4 - F6 ({n_cases} cases, "
          f"F_stock_(-1) = S0_PFAS): max |residual| / max|F_stock| = {worst_v1[0]:.2e} ({worst_v1[1]}), "
          f"tol {V1_TOL:g} -> {'PASS' if worst_v1[0] <= V1_TOL else 'FAIL'}")
    print(f"    zero floor binds in {len(floor_cases)} cases" + (":" if floor_cases else ""))
    for c in floor_cases:
        print(f"      {c}")
    print(f"V2  non-negativity of F4, F6, F_landfill_yearly, F2: {len(v2_viol)} case/column violations"
          + (":" if v2_viol else " -> PASS"))
    for c in v2_viol:
        print(f"      {c}")
    print(f"V3  F6_t <= F_stock_(t-1): max (F6_t - F_stock_(t-1)) = {worst_v3[0]:.4g} kg ({worst_v3[1]}) -> "
          f"{'PASS' if worst_v3[0] <= 0 else 'FAIL'}")
    ok = ok and worst_v3[0] <= 0

    print(f"\nV4  additivity (T2 logic on step {step} files)")
    ok = check_t2(final) and ok

    if "F_accu_landfill_deposited" in read_csv(final / "NP_CENTRAL_BAU.csv").columns:
        ok = check_landfill(final) and ok

    # V5 F6_1990 new / old
    b1 = B1_DIR
    print(f"\nV5  F6_1990 new (step {step}) / old (B1), TOTAL stream; expected 0.85 x S0_vol / V_stock_1990")
    print("    subclass estimate   F6_old_kg     F6_new_kg     new/old    expected   |  R: new/old  C: new/old")
    for subclass, estimate in PFAS_CONFIGS:
        ratios = {}
        for stream, suffix in STREAMS.items():
            name = f"{subclass}_{estimate.upper()}_BAU{suffix}.csv"
            new, old = read_csv(final / name).iloc[0], read_csv(b1 / name).iloc[0]
            ratios[stream] = (old["F6"], new["F6"], new["F6"] / old["F6"] if old["F6"] else np.nan,
                              0.85 * s0[(subclass, estimate, stream)][1] / new["V_stock"])
        o, n, r, e = ratios["TOTAL"]
        print(f"    {subclass:5s}   {estimate:8s} {o:11.4g}   {n:11.4g}   {r:9.4f}   {e:9.4f}   |  "
              f"{ratios['R'][2]:9.4f}   {ratios['C'][2]:9.4f}")

    # V6 leachate plausibility
    print("\nV6  cumulative PFAS_landfill_leachate 1990-2060 vs F_accu_landfill(2060), central BAU (report only)")
    for subclass in ["NP", "PFAA"]:
        for label, folder in [("B1", b1), (step + " ", final)]:
            df = read_csv(folder / f"{subclass}_CENTRAL_BAU.csv").set_index("year")
            cum = df["PFAS_landfill_leachate"].sum()
            acc = df.loc[2060, "F_accu_landfill"]
            print(f"    {subclass:4s} {label}: cumulative leachate = {cum:10.4g} kg, F_accu_landfill(2060) = "
                  f"{acc:10.4g} kg, ratio = {cum / acc:.3f}")

    print(f"\nPhase 2: {'ALL PASS' if ok and not v2_viol else 'CHECK FAILURES ABOVE'}")
    if not ok:
        sys.exit(1)


def check_landfill(folder: Path) -> bool:
    """Step E landfill checks: inventory balance, leachate <= previous inventory, cumulative leachate <= deposited."""
    worst_bal, worst_leach, worst_cum, worst_dep, n_neg, n = 0.0, -np.inf, -np.inf, 0.0, 0, 0
    where = {}
    for scenario, subclass, estimate, stream, name, df in iter_cases(folder):
        n += 1
        inv = df["F_accu_landfill"].to_numpy()
        prev = np.concatenate([[0.0], inv[:-1]])
        leach = df["PFAS_landfill_leachate"].to_numpy()
        dep_y = df["F_landfill_yearly"].to_numpy()
        dep = df["F_accu_landfill_deposited"].to_numpy()
        scale = max(np.abs(dep).max(), 1e-300)
        bal = np.abs((inv - prev) - (dep_y - leach)).max() / scale
        dep_err = np.abs(dep - np.cumsum(dep_y)).max() / scale
        leach_margin = (leach - prev).max()
        cum_margin = (np.cumsum(leach) - dep).max()
        n_neg += int((inv < 0).sum())
        for key, val, worst in [("bal", bal, worst_bal), ("leach", leach_margin, worst_leach),
                                ("cum", cum_margin, worst_cum), ("dep", dep_err, worst_dep)]:
            if val > worst:
                where[key] = name
        worst_bal, worst_leach = max(worst_bal, bal), max(worst_leach, leach_margin)
        worst_cum, worst_dep = max(worst_cum, cum_margin), max(worst_dep, dep_err)

    ok = worst_bal <= V1_TOL and worst_leach <= 0 and worst_cum <= 0 and worst_dep <= V1_TOL and n_neg == 0
    print(f"\nV7  landfill ({n} cases): inventory_t - inventory_(t-1) = F_landfill_yearly - leachate: "
          f"max |residual| / max deposited = {worst_bal:.2e} ({where.get('bal', '')})")
    print(f"    F_accu_landfill_deposited = cumsum(F_landfill_yearly): max rel error = {worst_dep:.2e}")
    print(f"    leachate_t <= inventory_(t-1): max (leachate - inventory_prev) = {worst_leach:.4g} kg "
          f"({where.get('leach', '')})")
    print(f"    cumulative leachate <= cumulative deposited (every year): max (cum leachate - deposited) = "
          f"{worst_cum:.4g} kg ({where.get('cum', '')})")
    print(f"    negative inventory values: {n_neg}")
    print(f"V7: {'PASS' if ok else 'FAIL'}")
    return ok


def k_L_from_config(subclass: str, estimate: str) -> float:
    c, cfg, cc = MODEL_CONSTANTS, PFAS_CONFIGS[(subclass, estimate)], CARPET_CONFIGS[estimate]
    leachate_per_tonne = float(cfg.xt) * c.LG / c.SA * 365 / 1_000_000 * c.leachate_release_fraction
    pfas_per_tonne_new = cfg.P_production * c.initial_stock_retention / cc.carpet_density / 1000
    return leachate_per_tonne, pfas_per_tonne_new, leachate_per_tonne / pfas_per_tonne_new


def report_step_e(work_dir: Path) -> None:
    d_dir, e_dir = step_dir(work_dir, "D"), step_dir(work_dir, "E")
    leach_subclasses = [s for s, e in PFAS_CONFIGS if e == "central" and PFAS_CONFIGS[(s, e)].has_leachate]

    print("k_L = [leachate per tonne landfilled carpet per yr] / [PFAS per tonne of new carpet]")
    print("    subclass estimate  leachate_kg/t/yr   PFAS_new_kg/t     k_L (1/yr)   implied k_L median 1995-2060 (step D)")
    for subclass in leach_subclasses:
        for estimate in ESTIMATES_UP:
            num, den, k = k_L_from_config(subclass, estimate.lower())
            d = read_csv(d_dir / f"{subclass}_{estimate}_BAU.csv").set_index("year")
            implied = (d["PFAS_landfill_leachate"] / d["F_accu_landfill"]).loc[1995:2060].median()
            print(f"    {subclass:5s}   {estimate:8s}  {num:13.4e}   {den:13.4e}   {k:11.5f}   {implied:11.5f}")

    print("\nCumulative leachate 1990-2060 and 2060 landfill inventory (TOTAL stream), step D vs step E")
    print("    scenario    subclass estimate   cum_leach_D   cum_leach_E    change %   | inventory_2060_D"
          "  inventory_2060_E  change %   deposited_E==D")
    for scenario in SCENARIOS:
        for subclass in leach_subclasses:
            for estimate in ESTIMATES_UP:
                name = f"{subclass}_{estimate}_{scenario}.csv"
                d, e = read_csv(d_dir / name).set_index("year"), read_csv(e_dir / name).set_index("year")
                cd, ce = d["PFAS_landfill_leachate"].sum(), e["PFAS_landfill_leachate"].sum()
                idd, ie = d.loc[2060, "F_accu_landfill"], e.loc[2060, "F_accu_landfill"]
                same_dep = np.array_equal(e["F_accu_landfill_deposited"].to_numpy(), d["F_accu_landfill"].to_numpy())
                print(f"    {scenario:11s} {subclass:5s}    {estimate:8s} {cd:11.4g}   {ce:11.4g}   {pct(ce, cd):+8.2f}   | "
                      f"{idd:13.4g}     {ie:13.4g}   {pct(ie, idd):+8.2f}    {same_dep}")

    print("\nAll-subclass F_accu_landfill (inventory, step E) and leachate, step D vs step E")
    print("    scenario    estimate  year   F_accu_D      F_accu_E    change %  |  leach_D     leach_E    change %")
    for scenario in SCENARIOS:
        for estimate in ESTIMATES_UP:
            name = f"TOTAL_{estimate}_ALL_SUBCLASSES_{scenario}.csv"
            d, e = read_csv(d_dir / name).set_index("year"), read_csv(e_dir / name).set_index("year")
            for year in YEARS_HL:
                a, b = d.loc[year, "F_accu_landfill"], e.loc[year, "F_accu_landfill"]
                la, lb = d.loc[year, "PFAS_landfill_leachate"], e.loc[year, "PFAS_landfill_leachate"]
                print(f"    {scenario:11s} {estimate:8s}  {year}  {a:11.4g}   {b:11.4g}   {pct(b, a):+8.2f}  | "
                      f"{la:9.4g}   {lb:9.4g}   {pct(lb, la):+8.2f}")


ESTIMATES_UP = ["HIGH", "CENTRAL", "LOW"]


# ---------------------------------------------------------------------------------------------------------
# Phase 3: impact of A, B, C vs B1
# ---------------------------------------------------------------------------------------------------------
def pct(x, base):
    return 100 * (x - base) / base if base != 0 else np.nan


def load_series(folder: Path, name: str, metric: str) -> pd.Series:
    return read_csv(folder / name).set_index("year")[metric]


DECOMP_STEPS = ["A", "B", "C", "D", "E"]
DECOMP_LABELS = {"A": "P_import 448 -> 488", "B": "phase-out on F1", "C": "first-order F6",
                 "D": "phase-out on F10", "E": "mass-conserving leachate"}


def decomposition_rows(dirs: dict, name: str, metric: str, years: list, label: dict) -> list:
    s = {step: load_series(folder, name, metric) for step, folder in dirs.items()}
    rows = []
    points = [(y, {k: v.loc[y] for k, v in s.items()}) for y in years]
    if metric in ("F6", "PFAS_landfill_leachate"):
        points.append(("cum", {k: v.sum() for k, v in s.items()}))
    for year, v in points:
        b = v["B1"]
        row = {**label, "metric": metric, "year": year, "B1_kg": b, "final_kg": v[FINAL],
               "total_pct": pct(v[FINAL], b)}
        prev = "B1"
        for step in DECOMP_STEPS:
            row[f"{step}_pct"] = pct(v[step], b) - pct(v[prev], b) if b else np.nan
            prev = step
        row["E_vs_D_pct"] = pct(v["E"], v["D"])
        rows.append(row)
    return rows


def fmt(v, spec="+8.2f"):
    return "     n/a" if pd.isna(v) else format(v, spec)


def print_rows(rows: list, title: str) -> None:
    print(f"\n  {title}")
    print("    metric                   year      B1_kg    final_kg     total       (A)       (B)       (C)"
          "       (D)       (E)  | E vs D %")
    for r in rows:
        steps = "  ".join(fmt(r[f"{step}_pct"]) for step in DECOMP_STEPS)
        print(f"    {r['metric']:24s} {str(r['year']):>5s} {r['B1_kg']:11.4g} {r['final_kg']:11.4g} "
              f"{fmt(r['total_pct'])}  {steps}  | {fmt(r['E_vs_D_pct'], '+9.3f')}")


def check_phase3(work_dir: Path) -> None:
    dirs = {"B1": B1_DIR, **{step: step_dir(work_dir, step) for step in DECOMP_STEPS}}
    print(f"Phase 3  % change of step {FINAL} vs B1 (commit e287d94, P_import = 448); "
          f"total = (A) + ... + (E), each in % of B1")
    print("  " + ";  ".join(f"({step}) {DECOMP_LABELS[step]}" for step in DECOMP_STEPS))
    all_rows = []
    for scenario in SCENARIOS:
        rows = []
        for metric in CORE_METRICS:
            rows += decomposition_rows(dirs, f"TOTAL_CENTRAL_ALL_SUBCLASSES_{scenario}.csv", metric, YEARS,
                                       {"scenario": scenario, "estimate": "CENTRAL", "subclass": "ALL"})
        print_rows(rows, f"{scenario}, central, all subclasses")
        all_rows += rows
    for estimate in ["HIGH", "LOW"]:
        for scenario in SCENARIOS:
            rows = []
            for metric in CORE_METRICS:
                rows += decomposition_rows(dirs, f"TOTAL_{estimate}_ALL_SUBCLASSES_{scenario}.csv", metric, YEARS_HL,
                                           {"scenario": scenario, "estimate": estimate, "subclass": "ALL"})
            print_rows([r for r in rows if r["year"] != "cum"] + [r for r in rows if r["year"] == "cum"],
                       f"{scenario}, {estimate.lower()}, all subclasses")
            all_rows += rows
    for scenario in SCENARIOS:
        for subclass in ["NP", "PFAA", "SP"]:
            rows = []
            for metric in ["F_stock", "F6"]:
                rows += decomposition_rows(dirs, f"{subclass}_CENTRAL_{scenario}.csv", metric, YEARS,
                                           {"scenario": scenario, "estimate": "CENTRAL", "subclass": subclass})
            print_rows(rows, f"{scenario}, central, {subclass}")
            all_rows += rows
    pd.DataFrame(all_rows).to_csv(work_dir / "phase3_decomposition.csv", index=False)

    print(f"\nCommercial share (step {FINAL})")
    check_t4(dirs[FINAL])

    print(f"\nNP F_stock and F6, central BAU, B1 (old) vs {FINAL} (new), kg")
    old = read_csv(B1_DIR / "NP_CENTRAL_BAU.csv").set_index("year")
    new = read_csv(dirs[FINAL] / "NP_CENTRAL_BAU.csv").set_index("year")
    traj = pd.DataFrame({"F_stock_old": old["F_stock"], "F_stock_new": new["F_stock"],
                         "F6_old": old["F6"], "F6_new": new["F6"]})
    traj.to_csv(work_dir / "NP_central_BAU_trajectory.csv")
    print("    year   F_stock_old   F_stock_new      F6_old      F6_new")
    for year in list(range(1990, 2061, 5)):
        r = traj.loc[year]
        print(f"    {year}  {r.F_stock_old:11.4g}   {r.F_stock_new:11.4g}  {r.F6_old:10.4g}  {r.F6_new:10.4g}")
    print(f"\n    tables saved: {work_dir / 'phase3_decomposition.csv'}, {work_dir / 'NP_central_BAU_trajectory.csv'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["1", "2", "3", "E"], required=True,
                   help="1: build steps; 2: checks on the final step; 3: decomposition vs B1; E: leachate report")
    p.add_argument("--work-dir", type=Path, required=True)
    args = p.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "1":
        check_phase1(args.work_dir)
    elif args.phase == "2":
        check_phase2(args.work_dir)
    elif args.phase == "3":
        check_phase3(args.work_dir)
    elif args.phase == "E":
        report_step_e(args.work_dir)


if __name__ == "__main__":
    main()
