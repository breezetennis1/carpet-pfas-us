import math
from dataclasses import dataclass, replace
from typing import Dict, Tuple, Optional


@dataclass(frozen=True)
class PFASConfig:
    subclass: str
    estimate: str
    P_import: float
    P_production: float
    P_export: float
    P_yearlyloss_air: float
    P_yearlyloss_dust: float
    Prod_emi_R: float
    xt: Optional[float] = None
    P_protector: Optional[float] = None
    has_leachate: bool = False
    use_protector: bool = False


@dataclass(frozen=True)
class CarpetConfig:
    estimate: str
    carpet_density: float
    lifespan_mean: float
    nylon_fmin: float
    average_floor_area: float
    initial_stock_age: float


@dataclass(frozen=True)
class ModelConstants:
    start_year: int = 1990
    end_year: int = 2060
    P_cf: float = 0.51
    initial_stock_retention: float = 0.85
    production_waste_fraction: float = 0.12
    production_scrap_to_landfill_fraction: float = 0.018
    SA: float = 140_000.0  # t/ha; Lang et al. 2017 Table 4: 0.14 Tg/ha (SI Table S3 mean 144,600 t/ha)
    LG: float = 3.26
    leachate_release_fraction: float = 0.35
    R_to_carpet: float = 0.36
    initial_household_size: float = 2.63


MODEL_CONSTANTS = ModelConstants()


CARPET_CONFIGS: Dict[str, CarpetConfig] = {
    "high": CarpetConfig(
        estimate="high",
        carpet_density=2.5 * 1000,
        lifespan_mean=11.25,
        nylon_fmin=0.4,
        average_floor_area=187,
        initial_stock_age=4.73,
    ),
    "central": CarpetConfig(
        estimate="central",
        carpet_density=2.35 * 1000,
        lifespan_mean=9,
        nylon_fmin=0.3,
        average_floor_area=167,
        initial_stock_age=4.3,
    ),
    "low": CarpetConfig(
        estimate="low",
        carpet_density=2.2 * 1000,
        lifespan_mean=7.0,
        nylon_fmin=0.2,
        average_floor_area=150,
        initial_stock_age=3.87,
    ),
}


# In-use losses (ug/m2/yr): P_yearlyloss_air = C_M (ng/m3) x 2.4 m x 12.7 /d x 365 / 1000;
# P_yearlyloss_dust = C_d (ug/g) x 7.3 g/m2/yr.
_PFAS_LOW_HIGH: Dict[Tuple[str, str], PFASConfig] = {
    ("NP", "high"): PFASConfig(
        subclass="NP",
        estimate="high",
        P_import=79,
        P_production=10500,
        P_export=10500,
        P_yearlyloss_air=729.813,
        P_yearlyloss_dust=39.42,
        Prod_emi_R=0.15,
        xt=20.6,
        has_leachate=True,
    ),
    ("NP", "low"): PFASConfig(
        subclass="NP",
        estimate="low",
        P_import=37,
        P_production=900,
        P_export=900,
        P_yearlyloss_air=2.225040,
        P_yearlyloss_dust=0.073,
        Prod_emi_R=0.05,
        xt=0.6,
        has_leachate=True,
    ),
    ("PFAA", "high"): PFASConfig(
        subclass="PFAA",
        estimate="high",
        P_import=72,
        P_production=9600,
        P_export=9600,
        P_yearlyloss_air=51.17592,
        P_yearlyloss_dust=33.58,
        Prod_emi_R=0.15,
        xt=8,
        has_leachate=True,
    ),
    ("PFAA", "low"): PFASConfig(
        subclass="PFAA",
        estimate="low",
        P_import=25,
        P_production=600,
        P_export=600,
        P_yearlyloss_air=5.5626,
        P_yearlyloss_dust=0.219,
        Prod_emi_R=0.05,
        xt=1.5,
        has_leachate=True,
    ),
    ("SP", "high"): PFASConfig(
        subclass="SP",
        estimate="high",
        P_import=664,
        P_production=88200,
        P_export=88200,
        P_yearlyloss_air=0,
        P_yearlyloss_dust=0,  # SP release during use not estimated (no carpet-specific data); 0 = not included
        Prod_emi_R=0.15,
        P_protector=0.01,
        use_protector=True,
    ),
    ("SP", "low"): PFASConfig(
        subclass="SP",
        estimate="low",
        P_import=312,
        P_production=7600,
        P_export=7600,
        P_yearlyloss_air=0,
        P_yearlyloss_dust=0,  # SP release during use not estimated (no carpet-specific data); 0 = not included
        Prod_emi_R=0.05,
        P_protector=0.007,
        use_protector=True,
    ),
}


# Parameters sampled log-uniform between low and high in the Monte Carlo analysis
# (the in-use losses enter through the rates k = P_yearlyloss / P_production, also log-uniform)
LOG_UNIFORM_PARAMS = frozenset({"P_import", "P_production", "xt", "P_protector"})

# Parameters whose central estimate is the geometric mean of low and high. This equals the median of the
# log-uniform Monte Carlo distribution; GM(P_yearlyloss) / GM(P_production) = GM(k).
GEOMETRIC_MEAN_PARAMS = LOG_UNIFORM_PARAMS | {"P_export", "P_yearlyloss_air", "P_yearlyloss_dust"}


def central_from_low_high(low: PFASConfig, high: PFASConfig) -> PFASConfig:
    """Central PFAS config: geometric mean of low and high (0 or None if either is), arithmetic mean
    for the uniformly sampled Prod_emi_R; the has_leachate / use_protector flags are shared."""
    if (low.has_leachate, low.use_protector) != (high.has_leachate, high.use_protector):
        raise ValueError(f"Low and high flags differ for {low.subclass}")
    values = {}
    for p in sorted(GEOMETRIC_MEAN_PARAMS):
        lo, hi = getattr(low, p), getattr(high, p)
        if lo is None or hi is None:
            values[p] = None
        elif lo == 0 or hi == 0:
            values[p] = 0.0
        else:
            values[p] = math.sqrt(lo * hi)
    return replace(high, estimate="central", Prod_emi_R=(low.Prod_emi_R + high.Prod_emi_R) / 2, **values)


PFAS_CONFIGS: Dict[Tuple[str, str], PFASConfig] = {}
for _subclass in ["NP", "PFAA", "SP"]:
    _low, _high = _PFAS_LOW_HIGH[(_subclass, "low")], _PFAS_LOW_HIGH[(_subclass, "high")]
    PFAS_CONFIGS[(_subclass, "high")] = _high
    PFAS_CONFIGS[(_subclass, "central")] = central_from_low_high(_low, _high)
    PFAS_CONFIGS[(_subclass, "low")] = _low
