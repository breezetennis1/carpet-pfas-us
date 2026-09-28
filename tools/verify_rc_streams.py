#!/usr/bin/env python3
r"""
Verification of the residential / commercial (R/C) two-stream national model (R2.1 / E7).

Run from the repository root with the project environment:

    C:\Users\mchen48\anaconda_envs\carpet-pfas-rev\python.exe tools\verify_rc_streams.py --phase 1

    ...\python.exe tools\verify_rc_streams.py --phase 3 --work-dir <scratch folder>

Phase 1: input checks for the commercial share phi_k and the CBECS 1990 commercial floor area.
Phase 3: T1 exact decomposition, T2 additivity of written files, T3 impact vs the pre-R/C baseline
         (outputs/tables/national_baseline_pre_RC/) and its decomposition, T4 commercial shares.
         All model outputs of the tests are written to --work-dir only; nothing is written to outputs/.
"""

import argparse
import contextlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.model as model  # noqa: E402
import src.runner as runner  # noqa: E402
from src.config import CARPET_CONFIGS, MODEL_CONSTANTS, PFAS_CONFIGS  # noqa: E402
from src.io_utils import (  # noqa: E402
    load_basic_data,
    load_cbecs_commercial_floorspace_1990,
    load_commercial_share,
)
from src.model import compute_initial_stock_volume, run_national_model  # noqa: E402
from src.postprocess import CORE_METRICS, build_contribution_long, build_total_output  # noqa: E402
from src.runner import (  # noqa: E402
    CBECS_FLOORSPACE_CSV,
    COMMERCIAL_SHARE_CSV,
    ESTIMATES,
    NON_ADDITIVE_COLUMNS,
    STREAM_SPLIT_COLUMNS,
    build_s0_scales,
    combine_streams,
    run_scenario_streams,
    split_streams,
)
from src.scenario import apply_scenario  # noqa: E402


PHI_SPOT_CHECKS = {
    1990: 0.345057,
    2012: 0.261622,
    2013: 0.287417,
    2014: 0.313212,
    2025: 0.378316,
    2060: 0.345057,
}
PHI_TOL = 1e-6


def check_phase1() -> None:
    phi = load_commercial_share(REPO / COMMERCIAL_SHARE_CSV)
    print(f"phi: {len(phi)} years {phi.index.min()}-{phi.index.max()}, "
          f"NaN = {int(phi.isna().sum())}, min = {phi.min():.6f} ({phi.idxmin()}), "
          f"max = {phi.max():.6f} ({phi.idxmax()})")

    for year, expected in PHI_SPOT_CHECKS.items():
        diff = abs(phi[year] - expected)
        status = "OK" if diff <= PHI_TOL else "FAIL"
        print(f"  phi({year}) = {phi[year]:.6f}  expected {expected:.6f}  |diff| = {diff:.1e}  {status}")
        assert diff <= PHI_TOL, f"phi({year}) = {phi[year]} != {expected}"

    base_df = load_basic_data(REPO / "data" / "Basic_data.csv")
    assert base_df["year"].tolist() == phi.index.tolist(), "Basic_data years do not match phi years"
    print("  Basic_data.csv years match phi years (1990-2060)")

    f_c = load_cbecs_commercial_floorspace_1990(REPO / CBECS_FLOORSPACE_CSV)
    print(f"F_C_1990 = {f_c:.2f} million m2 = {f_c * 1e6:.5e} m2 "
          f"(expected 4.4037e9 m2, rel diff {f_c / 4403.7 - 1:+.2e})")

    # Preview only: the model's residential S0 denominator per estimate, and the resulting S0_C scales.
    us_pop_1990 = float(base_df.loc[base_df["year"] == 1990, "us_pop"].iloc[0])
    phi_1990 = phi[1990]
    print(f"us_pop(1990) = {us_pop_1990} million; phi_1990/(1-phi_1990) = {phi_1990 / (1 - phi_1990):.4f}")
    for estimate, cc in CARPET_CONFIGS.items():
        res_area = cc.average_floor_area / MODEL_CONSTANTS.initial_household_size * us_pop_1990
        print(f"  {estimate:7s}: residential floor area 1990 = {res_area:.1f} million m2, "
              f"s0_scale_C (cbecs) = {f_c / res_area:.4f}")


SCENARIOS = ["BAU", "PHASE_OUT", "RECYCLE_24"]
BASELINE_DIR = REPO / "outputs" / "tables" / "national_baseline_pre_RC"
T3_YEARS = [1990, 2000, 2010, 2020, 2040, 2060]
T3_YEARS_OTHER_SCENARIOS = [2020, 2060]
T4_YEARS = [1990, 2000, 2020, 2060]

CONTRIBUTION_KEYS = ["year", "subclass", "estimate", "metric", "scenario"]
F4_YEAR0 = 'df.loc[0, "F4"] = (df.loc[0, "V4"] / S0_volumn) * (S0_PFAS-df.loc[0, "F6"])'


def read_csv(path: Path) -> pd.DataFrame:
    # round_trip: the default C parser can be 1 ulp off; tests compare against the exact written values
    return pd.read_csv(path, float_precision="round_trip")


def file_names(scenario: str, suffix: str = "") -> list[str]:
    names = [f"{subclass}_{estimate.upper()}_{scenario}{suffix}.csv" for subclass, estimate in PFAS_CONFIGS]
    names += [f"TOTAL_{estimate.upper()}_ALL_SUBCLASSES_{scenario}{suffix}.csv" for estimate in ESTIMATES]
    names.append(f"PFAS_subclass_contribution_long_{scenario}{suffix}.csv")
    return names


def split_additive(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    if "metric" in df.columns:
        additive = ["value"]
    else:
        additive = [c for c in df.columns if c not in NON_ADDITIVE_COLUMNS]
    return additive, [c for c in df.columns if c not in additive]


SCALED_TOL = 1e-12        # pass condition: |a - b| <= 1e-12 x max|b column|, every additive column
THRESHOLD_FRACTION = 1e-6  # thresholded relative test: entries with |b| >= 1e-6 x max|b column| ...
THRESHOLD_RTOL = 1e-9      # ... must agree to 1e-9 relative (reported, not the pass condition)


def diff_stats(a: pd.Series, b: pd.Series) -> dict:
    a = a.to_numpy(dtype=float)
    b = b.to_numpy(dtype=float)
    both_nan = np.isnan(a) & np.isnan(b)
    if (np.isnan(a) != np.isnan(b)).any():
        return {"nan_mismatch": True}
    d = np.where(both_nan, 0.0, np.abs(a - b))
    col_max = float(np.nanmax(np.abs(b))) if (~both_nan).any() else 0.0
    nz = (b != 0) & ~both_nan
    big = (np.abs(b) >= THRESHOLD_FRACTION * col_max) & nz
    return {
        "nan_mismatch": False,
        "max_abs": float(d.max()),
        "scaled": float(d.max() / col_max) if col_max > 0 else float(d.max()),
        "rel": float((d[nz] / np.abs(b[nz])).max()) if nz.any() else 0.0,
        "rel_thresholded": float((d[big] / np.abs(b[big])).max()) if big.any() else 0.0,
        "bitwise": bool((d == 0).all()),
    }


class Tally:
    def __init__(self):
        self.stats = {}
        self.failures = []

    def add(self, where: str, a: pd.Series, b: pd.Series) -> None:
        s = diff_stats(a, b)
        if s["nan_mismatch"]:
            self.failures.append(f"{where}: NaN pattern differs")
            return
        self.stats[where] = s
        if s["scaled"] > SCALED_TOL:
            self.failures.append(f"{where}: |diff| / max|column| = {s['scaled']:.2e}")

    def exact(self, where: str, a: pd.Series, b: pd.Series) -> None:
        if not a.reset_index(drop=True).equals(b.reset_index(drop=True)):
            self.failures.append(f"{where}: non-additive column differs")

    def report(self, label: str) -> bool:
        ok = not self.failures
        s = pd.DataFrame.from_dict(self.stats, orient="index")
        for key, text in [
            ("scaled", f"max |diff| / max|column|   (pass if <= {SCALED_TOL:g})"),
            ("max_abs", "max abs diff"),
            ("rel", "raw elementwise max rel diff"),
            ("rel_thresholded", f"max rel diff, |b| >= {THRESHOLD_FRACTION:g} x column max (tol {THRESHOLD_RTOL:g})"),
        ]:
            print(f"    {text:62s} = {s[key].max():.3e}   ({s[key].idxmax()})")
        n_thr_fail = int((s["rel_thresholded"] > THRESHOLD_RTOL).sum())
        print(f"    columns compared = {len(s)}, bitwise equal = {int(s['bitwise'].sum())}, "
              f"columns failing thresholded test = {n_thr_fail}")
        print(f"{label}: {'PASS' if ok else 'FAIL'}")
        for f in self.failures[:20]:
            print(f"    {f}")
        return ok


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(None):
        yield


@contextlib.contextmanager
def zero_initial_stock_guard():
    """
    Test-only: a stream with S0 = 0 divides by zero in F4 of year 0 (V4 / S0). For the T3 run without S0_C,
    use a copy of model.add_flow_side whose only change is F4[0] = 0 when S0 = 0. src/model.py is not modified.
    """
    source = inspect.getsource(model.add_flow_side)
    if source.count(F4_YEAR0) != 1:
        raise RuntimeError("add_flow_side changed; update F4_YEAR0 in the verification script")
    namespace = dict(vars(model))
    exec(source.replace(F4_YEAR0, F4_YEAR0 + " if S0_volumn > 0 else 0.0"), namespace)
    original = model.add_flow_side
    model.add_flow_side = namespace["add_flow_side"]
    try:
        yield
    finally:
        model.add_flow_side = original


def run_all_cases_to(work_dir: Path, s0_c_method: str = runner.S0_C_METHOD) -> None:
    original = runner.get_output_dirs
    runner.get_output_dirs = lambda root: {"tables_national": work_dir}
    try:
        with quiet():
            for scenario in SCENARIOS:
                runner.run_all_cases(REPO, scenario=scenario, s0_c_method=s0_c_method)
    finally:
        runner.get_output_dirs = original


def central_totals(base_df, scenario, phi, s0_R, s0_C, zero_s0=False) -> pd.DataFrame:
    """All-subclass totals (TOTAL = R + C) for the central estimate."""
    stream_inputs = split_streams(apply_scenario(base_df, scenario), phi)
    by_subclass = {}
    for (subclass, estimate), config in PFAS_CONFIGS.items():
        if estimate != "central":
            continue
        carpet_config = CARPET_CONFIGS[estimate]
        res_R = run_national_model(stream_inputs["R"], config, carpet_config, s0_scale=s0_R)
        guard = zero_initial_stock_guard() if zero_s0 else contextlib.nullcontext()
        with guard:
            res_C = run_national_model(stream_inputs["C"], config, carpet_config, s0_scale=s0_C)
        by_subclass[subclass] = combine_streams(res_R, res_C)
    return build_total_output(by_subclass).set_index("year")


def check_t1(base_df, phi, work_dir: Path) -> bool:
    c = float(phi.loc[1990])
    phi_const = pd.Series(c, index=phi.index)
    s0_scales = {"R": {e: 1 - c for e in ESTIMATES}, "C": {e: c for e in ESTIMATES}}
    t1_dir = work_dir / "T1_constant_phi"

    tally = Tally()
    n_files = 0
    for scenario in SCENARIOS:
        results = run_scenario_streams(base_df, scenario, phi_const, s0_scales)
        with quiet():
            runner.save_stream_outputs(results["TOTAL"], t1_dir, scenario, "")
        for name in file_names(scenario):
            test = read_csv(t1_dir / name)
            base = read_csv(BASELINE_DIR / name)
            assert list(test.columns) == list(base.columns) and len(test) == len(base), name
            additive, fixed = split_additive(base)
            for col in additive:
                tally.add(f"{name}:{col}", test[col], base[col])
            for col in fixed:
                tally.exact(f"{name}:{col}", test[col], base[col])
            n_files += 1

    print(f"\nT1  exact decomposition: phi = {c} for all years, s0_scale R = {1 - c:.6f}, C = {c:.6f}; "
          f"{n_files} files vs {BASELINE_DIR.name}/")
    return tally.report("T1")


def check_t2(final_dir: Path) -> bool:
    tally = Tally()
    n_files = 0
    for scenario in SCENARIOS:
        for name in file_names(scenario):
            stem = name[:-4]
            total = read_csv(final_dir / name)
            res = {s: read_csv(final_dir / f"{stem}_{s}.csv") for s in ["R", "C"]}
            for s, df in res.items():
                assert list(df.columns) == list(total.columns) and len(df) == len(total), f"{stem}_{s}"
            additive, fixed = split_additive(total)
            for col in additive:
                tally.add(f"{name}:{col}", res["R"][col] + res["C"][col], total[col])
            for col in fixed:
                tally.exact(f"{name}:{col} (R)", res["R"][col], total[col])
                tally.exact(f"{name}:{col} (C)", res["C"][col], total[col])
            n_files += 1

    print(f"\nT2  additivity of written files: R + C vs TOTAL, {n_files} file triplets in {final_dir}")
    return tally.report("T2")


def pct(x, base):
    return 100 * (x - base) / base


def check_t3(base_df, phi, f_c, final_dir: Path, work_dir: Path) -> None:
    c = float(phi.loc[1990])
    phi_const = pd.Series(c, index=phi.index)
    scales = {m: build_s0_scales(base_df, phi, f_c, m)["C"]["central"] for m in runner.S0_C_METHODS}

    print("\nT3  impact vs baseline (central, all-subclass totals, % of baseline)")
    print(f"    runs: B = baseline; A0 = constant phi {c}, S0_C = 0; A1 = constant phi, S0_C added; "
          f"F = time-varying phi, S0_C added")
    print(f"    s0_scale_C (central): cbecs = {scales['cbecs']:.6f}, sales_ratio = {scales['sales_ratio']:.6f}")
    print("    total = F - B = (0) A0 - B [stream structure] + (i) A1 - A0 [S0_C] + (ii) F - A1 [time-varying phi]")

    rows = []
    for scenario in SCENARIOS:
        B = read_csv(BASELINE_DIR / f"TOTAL_CENTRAL_ALL_SUBCLASSES_{scenario}.csv").set_index("year")
        A0 = central_totals(base_df, scenario, phi_const, 1.0, 0.0, zero_s0=True)
        runs = {"A0": A0}
        for m in runner.S0_C_METHODS:
            runs[f"A1_{m}"] = central_totals(base_df, scenario, phi_const, 1.0, scales[m])
            runs[f"F_{m}"] = central_totals(base_df, scenario, phi, 1.0, scales[m])

        written = read_csv(final_dir / f"TOTAL_CENTRAL_ALL_SUBCLASSES_{scenario}.csv").set_index("year")
        assert np.array_equal(written[CORE_METRICS].to_numpy(), runs["F_cbecs"][CORE_METRICS].to_numpy()), \
            f"final run {scenario} does not reproduce the written TOTAL file"

        years = T3_YEARS if scenario == "BAU" else T3_YEARS_OTHER_SCENARIOS
        for metric in CORE_METRICS:
            for year in years:
                b = B.loc[year, metric]
                row = {"scenario": scenario, "metric": metric, "year": year, "baseline_kg": b}
                for m in runner.S0_C_METHODS:
                    a0, a1, f = runs["A0"].loc[year, metric], runs[f"A1_{m}"].loc[year, metric], runs[f"F_{m}"].loc[year, metric]
                    row[f"{m}_final_kg"] = f
                    row[f"{m}_total_pct"] = pct(f, b)
                    row[f"{m}_(0)_pct"] = pct(a0, b)
                    row[f"{m}_(i)_S0C_pct"] = pct(a1, b) - pct(a0, b)
                    row[f"{m}_(ii)_phi_pct"] = pct(f, b) - pct(a1, b)
                rows.append(row)

    t3 = pd.DataFrame(rows)
    t3.to_csv(work_dir / "T3_impact_decomposition.csv", index=False)

    for scenario in SCENARIOS:
        print(f"\n  {scenario}")
        for metric in CORE_METRICS:
            sub = t3[(t3["scenario"] == scenario) & (t3["metric"] == metric)]
            print(f"    {metric}")
            print("      year  baseline_kg   |  cbecs: total    (0)     (i)S0_C  (ii)phi |  sales_ratio: total   (i)S0_C  (ii)phi")
            for _, r in sub.iterrows():
                print(f"      {int(r.year)}  {r.baseline_kg:11.4g}   |  "
                      f"{r['cbecs_total_pct']:+8.2f}  {r['cbecs_(0)_pct']:+7.3f}  {r['cbecs_(i)_S0C_pct']:+8.2f}  {r['cbecs_(ii)_phi_pct']:+7.2f} |  "
                      f"{r['sales_ratio_total_pct']:+8.2f}   {r['sales_ratio_(i)_S0C_pct']:+8.2f}  {r['sales_ratio_(ii)_phi_pct']:+7.2f}")
    print(f"\n    table saved: {work_dir / 'T3_impact_decomposition.csv'}")


def report_initial_stocks(base_df, phi, f_c) -> None:
    c = MODEL_CONSTANTS
    print(f"\nInitial stocks (million m2 nylon carpet), F_C_1990 = {f_c:.2f} million m2")
    print("    estimate  s0_scale_C(cbecs)  s0_scale_C(sales_ratio)   S0_R      S0_C(cbecs)  S0_C/S0_R   check F_C*P_cf*%nylon")
    scales = {m: build_s0_scales(base_df, phi, f_c, m)["C"] for m in runner.S0_C_METHODS}
    for estimate in ESTIMATES:
        cc = CARPET_CONFIGS[estimate]
        fitted = model.fit_and_apply_nylon_share(base_df, cc)
        s0_R = compute_initial_stock_volume(fitted, cc)
        s0_C = s0_R * scales["cbecs"][estimate]
        direct = f_c * c.P_cf * fitted.loc[0, "Percent_nylon"]
        print(f"    {estimate:8s}  {scales['cbecs'][estimate]:.6f}           {scales['sales_ratio'][estimate]:.6f}"
              f"                {s0_R:9.2f}  {s0_C:9.2f}    {s0_C / s0_R:.6f}   {direct:9.2f}")


def check_t4(final_dir: Path) -> None:
    summary = read_csv(final_dir / "R_C_share_summary_BAU.csv")
    central = summary[summary["estimate"] == "CENTRAL"]
    print("\nT4  C_share = C / TOTAL (BAU, central, all-subclass totals)")
    print("    variable                 " + "  ".join(f"{y:>6d}" for y in T4_YEARS) + "   min (year)       max (year)")
    for variable in CORE_METRICS:
        s = central[central["variable"] == variable].set_index("year")["C_share"]
        vals = "  ".join(f"{s.loc[y]:6.4f}" for y in T4_YEARS)
        print(f"    {variable:24s} {vals}   {s.min():.4f} ({s.idxmin()})   {s.max():.4f} ({s.idxmax()})")


def report_columns(final_dir: Path) -> None:
    df = read_csv(final_dir / "SP_CENTRAL_BAU.csv")
    additive, fixed = split_additive(df)
    print(f"\nSplit columns (x (1 - phi) for R, x phi for C): {STREAM_SPLIT_COLUMNS}")
    print(f"Non-additive, copied from R ({len(fixed)}): {fixed}")
    print(f"Additive, R + C ({len(additive)}): {additive}")


def check_file_counts(final_dir: Path) -> None:
    names = {p.name for p in final_dir.glob("*.csv")}
    total = [n for s in SCENARIOS for n in file_names(s)]
    for label, suffix in [("TOTAL", ""), ("_R", "_R"), ("_C", "_C")]:
        expected = [n[:-4] + suffix + ".csv" for n in total]
        print(f"    {label:5s}: {sum(n in names for n in expected)} / {len(expected)} files")
    summaries = [f"R_C_share_summary_{s}.csv" for s in SCENARIOS]
    print(f"    R_C_share_summary: {sum(n in names for n in summaries)} / 3; total files in dir: {len(names)}")


def check_phase3(work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    base_df = load_basic_data(REPO / "data" / "Basic_data.csv")
    phi = load_commercial_share(REPO / COMMERCIAL_SHARE_CSV)
    f_c = load_cbecs_commercial_floorspace_1990(REPO / CBECS_FLOORSPACE_CSV)

    t1_ok = check_t1(base_df, phi, work_dir)
    if not t1_ok:
        print("T1 FAILED - stopping.")
        sys.exit(1)

    final_dir = work_dir / "final_cbecs"
    run_all_cases_to(final_dir)
    print(f"\nFinal configuration (S0_C_METHOD = {runner.S0_C_METHOD!r}) written to {final_dir}")
    check_file_counts(final_dir)
    report_columns(final_dir)

    t2_ok = check_t2(final_dir)
    if not t2_ok:
        print("T2 FAILED - stopping.")
        sys.exit(1)

    report_initial_stocks(base_df, phi, f_c)
    check_t3(base_df, phi, f_c, final_dir, work_dir)
    check_t4(final_dir)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", type=int, choices=[1, 3], required=True)
    p.add_argument("--work-dir", type=Path, help="scratch folder for test outputs (phase 3)")
    args = p.parse_args()
    if args.phase == 1:
        check_phase1()
    elif args.phase == 3:
        if args.work_dir is None:
            p.error("--phase 3 requires --work-dir")
        check_phase3(args.work_dir)


if __name__ == "__main__":
    main()
