"""
Validation of the carpet waste allocation to landfills against EPA LMOP waste in place.

For the landfills that report waste in place for 2021 in the Landfill Methane Outreach Program (LMOP)
database, the modeled accumulated carpet mass in 2021 is compared with the reported waste in place on a
log-log scale (ln(1 + tonnes)), as in the original submission. The same test is run for the previous
allocation results (LFMass_Accum_StatusQuo.csv) and the revised results (landfill step 5), and the two
sets of modeled carpet mass are compared landfill by landfill.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

LMOP_YEAR = "2021"
SHORT_TON_TO_TONNE = 0.907184
LMOP_CSV = "lmopdata.csv"
PREVIOUS_MASS_CSV = "LFMass_Accum_StatusQuo.csv"  # previous results, carpet mass in tonnes
REVISED_MASS_CSV = Path("outputs") / "tables" / "landfill_central_BAU" / "landfill_mass_accum_central_BAU_MLF.csv"

# The revised allocation covers the contiguous U.S. only; both models are evaluated on the same domain
NON_CONTIGUOUS_STATES = ["AK", "HI", "PR", "GU", "VI", "AS", "MP"]
NON_CONTIGUOUS_STATEFP = [2, 15, 60, 66, 69, 72, 78]

RESULTS = {"previous": "Previous model", "revised": "Revised model"}
COLORS = {"previous": "#7F7F7F", "revised": "#1F77B4"}


def normalize_ids(ids: pd.Series) -> pd.Index:
    """Facility IDs as strings; numeric IDs without a decimal part ('2.0' -> '2'), other IDs unchanged."""
    s = ids.astype(str).str.strip()
    return pd.Index(s.str.replace(r"^(\d+)\.0+$", r"\1", regex=True))


def load_lmop(path: Path) -> pd.DataFrame:
    """LMOP landfills with waste in place reported for 2021 (metric tonnes), one row per Landfill ID."""
    lmop = pd.read_csv(path, encoding="latin-1")
    lmop = lmop.drop_duplicates(subset=["Landfill ID"], keep="first")  # one row per landfill (several projects)
    lmop = lmop[lmop["Waste in Place Year"] == int(LMOP_YEAR)].copy()
    wip = pd.to_numeric(lmop["Waste in Place (tons)"].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    lmop["wip_tonnes"] = wip * SHORT_TON_TO_TONNE  # short tons -> metric tonnes
    lmop = lmop.dropna(subset=["wip_tonnes"])
    lmop = lmop[~lmop["State"].isin(NON_CONTIGUOUS_STATES)]
    lmop.index = normalize_ids(lmop["Landfill ID"])
    return lmop[["Landfill Name", "State", "wip_tonnes"]]


def load_carpet_mass(path: Path) -> pd.Series:
    """Accumulated carpet mass (tonnes) in LMOP_YEAR by facility_id, contiguous U.S. landfills."""
    df = pd.read_csv(path, usecols=["facility_id", "STATEFP", LMOP_YEAR]).dropna(subset=["facility_id"])
    df = df[~df["STATEFP"].isin(NON_CONTIGUOUS_STATEFP)]
    df.index = normalize_ids(df["facility_id"])
    if not df.index.is_unique:
        raise ValueError(f"Duplicate facility_id in {path.name}")
    return df[LMOP_YEAR].astype(float)


def log_fit(carpet: pd.Series, wip: pd.Series) -> dict:
    """Pearson and Spearman correlation and OLS fit of ln(1 + waste in place) on ln(1 + carpet mass)."""
    x, y = np.log1p(carpet.to_numpy()), np.log1p(wip.to_numpy())
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    fit = stats.linregress(x, y)
    return {
        "n": len(x),
        "pearson_r": pearson.statistic,
        "pearson_p": pearson.pvalue,
        "spearman_rho": spearman.statistic,
        "spearman_p": spearman.pvalue,
        "slope": fit.slope,
        "intercept": fit.intercept,
        "r2": fit.rvalue ** 2,
        "x": x,
        "y": y,
    }


def run_landfill_validation(project_root: Path, validation_dir: Path) -> None:
    fig_dir = project_root / "outputs" / "figures" / "validation"
    table_dir = project_root / "outputs" / "tables" / "validation"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    lmop = load_lmop(validation_dir / LMOP_CSV)
    mass = {
        "previous": load_carpet_mass(validation_dir / PREVIOUS_MASS_CSV),
        "revised": load_carpet_mass(project_root / REVISED_MASS_CSV),
    }
    print(f"LMOP landfills in the contiguous U.S. with {LMOP_YEAR} waste in place: {len(lmop)}")

    # LMOP landfills present in the modeled landfill set (same matching as the original submission)
    matched = lmop.copy()
    for key, series in mass.items():
        matched[f"carpet_{key}_tonnes"] = series.reindex(matched.index)
        print(f"{RESULTS[key]}: {series.reindex(lmop.index).notna().sum()} LMOP landfills matched; "
              f"total modeled carpet mass {series.sum() / 1e6:.1f} million tonnes in {LMOP_YEAR}")
    matched = matched.dropna(subset=[f"carpet_{k}_tonnes" for k in mass])
    matched.index.name = "facility_id"
    matched.to_csv(table_dir / f"landfill_validation_LMOP_{LMOP_YEAR}.csv")

    # All matched landfills, and the same test without the landfills that receive no carpet in the
    # revised assignment (removed from both models so that they are compared on the same landfills)
    evaluate(matched, mass, table_dir, fig_dir, suffix="")
    zero_ids = matched.index[matched["carpet_revised_tonnes"] == 0]
    note = f"Excluding {len(zero_ids)} LMOP landfills with no carpet in the revised model"
    print(f"\n{note}: {', '.join(matched.loc[zero_ids, 'Landfill Name'].astype(str))}")
    mass_nonzero = {k: s.drop(mass['revised'].index[mass['revised'] == 0], errors="ignore") for k, s in mass.items()}
    results = evaluate(matched.drop(zero_ids), mass_nonzero, table_dir, fig_dir, suffix="_excl_zero", note=note)

    # SI figure: revised model only, landfills that receive carpet in the revised assignment
    plot_figure_s8(results["revised"], fig_dir / "Figure_S8_landfill_validation.png")


def evaluate(matched: pd.DataFrame, mass: dict, table_dir: Path, fig_dir: Path, suffix: str, note: str = "") -> dict:
    """Correlation statistics, summary table and figure for one set of LMOP landfills."""
    results = {k: log_fit(matched[f"carpet_{k}_tonnes"], matched["wip_tonnes"]) for k in mass}
    for k in mass:
        zero = int((matched[f"carpet_{k}_tonnes"] == 0).sum())
        results[k]["landfills_with_zero_carpet"] = zero
        results[k]["carpet_share_of_wip_median"] = float(
            (matched[f"carpet_{k}_tonnes"] / matched["wip_tonnes"]).median())
        # Sensitivity: landfills that receive carpet in the model only
        nonzero = matched[matched[f"carpet_{k}_tonnes"] > 0]
        nz = log_fit(nonzero[f"carpet_{k}_tonnes"], nonzero["wip_tonnes"])
        results[k].update({"n_nonzero": nz["n"], "pearson_r_nonzero": nz["pearson_r"],
                           "spearman_rho_nonzero": nz["spearman_rho"]})

    # Landfill-by-landfill agreement of the two sets of modeled carpet mass (all modeled landfills)
    both = pd.concat(mass, axis=1).dropna()
    both = both[(both > 0).any(axis=1)]
    lx, ly = np.log1p(both["previous"].to_numpy()), np.log1p(both["revised"].to_numpy())
    agreement = {
        "n": len(both),
        "pearson_r": stats.pearsonr(lx, ly).statistic,
        "spearman_rho": stats.spearmanr(both["previous"], both["revised"]).statistic,
        "total_ratio": both["revised"].sum() / both["previous"].sum(),
    }

    summary = pd.DataFrame(
        {RESULTS[k]: {m: v for m, v in r.items() if m not in ("x", "y")} for k, r in results.items()}
    )
    summary.loc["n_landfills_compared", "Revised vs previous"] = agreement["n"]
    summary.loc["pearson_r_log", "Revised vs previous"] = agreement["pearson_r"]
    summary.loc["spearman_rho", "Revised vs previous"] = agreement["spearman_rho"]
    summary.loc["total_mass_ratio", "Revised vs previous"] = agreement["total_ratio"]
    summary.to_csv(table_dir / f"landfill_validation_summary_{LMOP_YEAR}{suffix}.csv")
    print(summary.round(4).to_string())

    plot_validation(results, lx, ly, agreement, fig_dir / f"landfill_validation_LMOP_{LMOP_YEAR}{suffix}.png", note)
    return results


def plot_figure_s8(r: dict, path: Path) -> None:
    """Single-panel log-log comparison of modeled carpet mass and LMOP waste in place (SI figure)."""
    plt.rcParams.update({"font.family": "Arial", "font.size": 14})
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(r["x"], r["y"], color="blue", alpha=0.6, edgecolors="k", label="Landfills")
    xs = np.linspace(r["x"].min(), r["x"].max(), 100)
    ax.plot(xs, r["intercept"] + r["slope"] * xs, color="red", linewidth=2, label="Linear fit")
    ax.set_xlabel(f"Log-transformed predicted carpet mass accumulation, {LMOP_YEAR}\nln(1 + tonnes)")
    ax.set_ylabel(f"Log-transformed landfill waste in place from LMOP, {LMOP_YEAR}\nln(1 + tonnes)")
    ax.text(0.05, 0.95, f"Pearson r = {r['pearson_r']:.2f}\np-value = {r['pearson_p']:.3g}\nn = {r['n']}",
            transform=ax.transAxes, va="top", fontsize=14)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_validation(results: dict, lx: np.ndarray, ly: np.ndarray, agreement: dict, path: Path,
                    note: str = "") -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 14})
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    xmax = max(r["x"].max() for r in results.values()) * 1.05
    ymin = min(r["y"].min() for r in results.values()) * 0.95
    ymax = max(r["y"].max() for r in results.values()) * 1.03
    for ax, (key, r), label in zip(axes[:2], results.items(), ["(a)", "(b)"]):
        ax.scatter(r["x"], r["y"], s=22, color=COLORS[key], alpha=0.55, edgecolors="k", linewidths=0.4)
        xs = np.linspace(0, xmax, 100)
        ax.plot(xs, r["intercept"] + r["slope"] * xs, color="red", linewidth=2, label="Linear fit")
        ax.set_xlim(0, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"{RESULTS[key]}")
        ax.set_xlabel(f"Modeled carpet mass accumulation, {LMOP_YEAR}\nln(1 + tonnes)")
        ax.set_ylabel(f"LMOP landfill waste in place, {LMOP_YEAR}\nln(1 + tonnes)")
        ax.text(0.04, 0.96, f"{label}\nPearson r = {r['pearson_r']:.2f}\nSpearman ρ = {r['spearman_rho']:.2f}"
                f"\np = {r['pearson_p']:.2g}\nn = {r['n']}",
                transform=ax.transAxes, va="top", fontsize=13)
        ax.legend(loc="lower right", frameon=False)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    ax = axes[2]
    ax.scatter(lx, ly, s=14, color=COLORS["revised"], alpha=0.5, edgecolors="none")
    lim = max(lx.max(), ly.max()) * 1.03
    ax.plot([0, lim], [0, lim], color="k", linestyle="--", linewidth=1.2, label="1:1 line")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_title("Revised vs previous model")
    ax.set_xlabel(f"Previous carpet mass accumulation, {LMOP_YEAR}\nln(1 + tonnes)")
    ax.set_ylabel(f"Revised carpet mass accumulation, {LMOP_YEAR}\nln(1 + tonnes)")
    ax.text(0.04, 0.96, f"(c)\nPearson r = {agreement['pearson_r']:.2f}\nSpearman ρ = {agreement['spearman_rho']:.2f}"
            f"\nn = {agreement['n']}\nTotal revised / previous = {agreement['total_ratio']:.2f}",
            transform=ax.transAxes, va="top", fontsize=13)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    if note:
        fig.suptitle(note, fontsize=15, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
