"""
Shared model evaluation for the Monte Carlo and Sobol analyses (national BAU totals only).

Each evaluation runs the residential (R) and commercial (C) streams exactly as the national
runner does (phi split of the volume inputs, S0_C from the CBECS commercial floor area) for
NP, PFAA and SP, and returns the R + C all-subclass totals. Evaluations can run in parallel
worker processes; each worker builds the stream inputs once.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import PFASConfig, CarpetConfig
from src.io_utils import load_basic_data, load_cbecs_commercial_floorspace_1990, load_commercial_share
from src.model import run_national_model
from src.runner import (
    CBECS_FLOORSPACE_CSV,
    COMMERCIAL_SHARE_CSV,
    S0_C_METHOD,
    compute_s0_scale_C,
    split_streams,
)
from src.scenario import apply_scenario

SUBCLASSES = ["NP", "PFAA", "SP"]

# Model column -> total output column
TOTAL_OUTPUTS = {
    "F_stock": "F_stock_total",
    "F6": "F6_total",
    "F_accu_landfill": "F_accu_landfill_total",
    "PFAS_landfill_leachate": "PFAS_landfill_leachate_total",
}

Sample = Tuple[Dict[str, PFASConfig], CarpetConfig]

_CONTEXT: dict = {}


def build_context(project_root: Path, scenario: str = "BAU") -> dict:
    base_df = load_basic_data(project_root / "data" / "Basic_data.csv")
    phi = load_commercial_share(project_root / COMMERCIAL_SHARE_CSV)
    return {
        "inputs": split_streams(apply_scenario(base_df, scenario), phi),
        "us_pop_1990": float(base_df.loc[0, "us_pop"]),
        "phi_1990": float(phi.loc[int(base_df.loc[0, "year"])]),
        "floorspace_c_1990": load_cbecs_commercial_floorspace_1990(project_root / CBECS_FLOORSPACE_CSV),
    }


def evaluate_total(context: dict, configs: Dict[str, PFASConfig], carpet_config: CarpetConfig) -> pd.DataFrame:
    """R + C all-subclass totals (kg, kg/yr) for one parameter sample, summed as in the national runner."""
    s0_scale_C = compute_s0_scale_C(
        S0_C_METHOD, carpet_config, context["us_pop_1990"], context["phi_1990"], context["floorspace_c_1990"]
    )
    total = None
    for subclass in SUBCLASSES:
        res_R = run_national_model(context["inputs"]["R"], configs[subclass], carpet_config, s0_scale=1.0)
        res_C = run_national_model(context["inputs"]["C"], configs[subclass], carpet_config, s0_scale=s0_scale_C)
        if total is None:
            total = pd.DataFrame({"year": res_R["year"]})
            for col in TOTAL_OUTPUTS.values():
                total[col] = 0.0
        for model_col, out_col in TOTAL_OUTPUTS.items():
            total[out_col] += (res_R[model_col] + res_C[model_col]).fillna(0.0)
    return total


def _init_worker(project_root: str, scenario: str) -> None:
    _CONTEXT.update(build_context(Path(project_root), scenario))


def _evaluate_sample(sample: Sample) -> pd.DataFrame:
    configs, carpet_config = sample
    return evaluate_total(_CONTEXT, configs, carpet_config)


def evaluate_samples(
    project_root: Path,
    samples: Sequence[Sample],
    n_workers: int = 1,
    scenario: str = "BAU",
    progress_every: int = 500,
) -> List[pd.DataFrame]:
    """Evaluate samples in order (results are identical for any number of workers)."""
    if n_workers <= 1:
        context = build_context(project_root, scenario)
        return [evaluate_total(context, c, cc) for c, cc in samples]

    results = []
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker, initargs=(str(project_root), scenario)
    ) as pool:
        for i, res in enumerate(pool.map(_evaluate_sample, samples, chunksize=8)):
            results.append(res)
            if (i + 1) % progress_every == 0:
                print(f"{i + 1} / {len(samples)} evaluations finished", flush=True)
    return results


def default_workers() -> int:
    import os
    return max(1, (os.cpu_count() or 2) - 2)


def carpet_bounds() -> Dict[str, Tuple[float, float]]:
    """Carpet parameter ranges from the low / high estimates in config.py."""
    from src.config import CARPET_CONFIGS
    lo, hi = CARPET_CONFIGS["low"], CARPET_CONFIGS["high"]
    params = ["carpet_density", "lifespan_mean", "nylon_fmin", "average_floor_area", "initial_stock_age"]
    return {p: (float(min(getattr(lo, p), getattr(hi, p))), float(max(getattr(lo, p), getattr(hi, p)))) for p in params}


# First-order in-use emission rates (1/yr), sampled instead of P_yearlyloss: k = P_yearlyloss / P_production
EMISSION_RATES = {"k_air": "P_yearlyloss_air", "k_dust": "P_yearlyloss_dust"}


def emission_rate_bounds(low_cfg: PFASConfig, high_cfg: PFASConfig, rate: str) -> Tuple[float, float]:
    """Range of an emission rate between the low and high estimates of a subclass."""
    loss = EMISSION_RATES[rate]
    k = [getattr(c, loss) / c.P_production for c in (low_cfg, high_cfg)]
    return float(min(k)), float(max(k))


def check_nonnegative(results: Sequence[pd.DataFrame]) -> None:
    values = np.concatenate([r[list(TOTAL_OUTPUTS.values())].to_numpy() for r in results])
    if np.iscomplexobj(values) or values.dtype == object:
        raise ValueError("Non-real totals in the uncertainty runs")
    if np.isnan(values).any() or (values < 0).any():
        raise ValueError("NaN or negative totals in the uncertainty runs")
