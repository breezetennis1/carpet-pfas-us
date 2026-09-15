from dataclasses import dataclass
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
    SA: float = 140.0
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


PFAS_CONFIGS: Dict[Tuple[str, str], PFASConfig] = {
    ("NP", "high"): PFASConfig(
        subclass="NP",
        estimate="high",
        P_import=79,
        P_production=10500,
        P_export=10500,
        P_yearlyloss_air=729.813,
        P_yearlyloss_dust=0.03942,
        Prod_emi_R=0.15,
        xt=20.6,
        has_leachate=True,
    ),
    ("NP", "central"): PFASConfig(
        subclass="NP",
        estimate="central",
        P_import=58,
        P_production=5700,
        P_export=5700,
        P_yearlyloss_air=366.019,
        P_yearlyloss_dust=0.01975,
        Prod_emi_R=0.10,
        xt=10.6,
        has_leachate=True,
    ),
    ("NP", "low"): PFASConfig(
        subclass="NP",
        estimate="low",
        P_import=37,
        P_production=900,
        P_export=900,
        P_yearlyloss_air=2.225040,
        P_yearlyloss_dust=0.000073,
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
        P_yearlyloss_dust=0.03358,
        Prod_emi_R=0.15,
        xt=8,
        has_leachate=True,
    ),
    ("PFAA", "central"): PFASConfig(
        subclass="PFAA",
        estimate="central",
        P_import=48,
        P_production=5100,
        P_export=5100,
        P_yearlyloss_air=28.3693,
        P_yearlyloss_dust=0.0169,
        Prod_emi_R=0.10,
        xt=4.75,
        has_leachate=True,
    ),
    ("PFAA", "low"): PFASConfig(
        subclass="PFAA",
        estimate="low",
        P_import=25,
        P_production=600,
        P_export=600,
        P_yearlyloss_air=5.5626,
        P_yearlyloss_dust=0.000219,
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
        P_yearlyloss_dust=1.752,
        Prod_emi_R=0.15,
        P_protector=0.01,
        use_protector=True,
    ),
    ("SP", "central"): PFASConfig(
        subclass="SP",
        estimate="central",
        P_import=448,
        P_production=47900,
        P_export=47900,
        P_yearlyloss_air=0,
        P_yearlyloss_dust=1.314,
        Prod_emi_R=0.10,
        P_protector=0.0085,
        use_protector=True,
    ),
    ("SP", "low"): PFASConfig(
        subclass="SP",
        estimate="low",
        P_import=312,
        P_production=7600,
        P_export=7600,
        P_yearlyloss_air=0,
        P_yearlyloss_dust=0.876,
        Prod_emi_R=0.05,
        P_protector=0.007,
        use_protector=True,
    ),
}