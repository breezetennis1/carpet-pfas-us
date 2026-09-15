from __future__ import annotations

import pandas as pd


VALID_SCENARIOS = {"BAU", "RECYCLE_24", "PHASE_OUT"}


def apply_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """
    Prepare scenario-specific columns for the model.
    Keeps the original columns unchanged and creates effective columns:
      - RR_effective
      - PhaseOut_effective
      - scenario
    """
    scenario = scenario.upper()
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"Unsupported scenario: {scenario}")

    out = df.copy()
    out["scenario"] = scenario

    # Default values
    out["RR_effective"] = out["RR_total"]
    out["PhaseOut_effective"] = 1.0

    if scenario == "BAU":
        pass

    elif scenario == "RECYCLE_24":
        if "RR_total_24" not in out.columns:
            raise ValueError("RECYCLE_24 scenario requires column 'RR_total_24' in Basic_data.csv")
        out["RR_effective"] = out["RR_total_24"]

    elif scenario == "PHASE_OUT":
        if "PhaseOut" not in out.columns:
            raise ValueError("PHASE_OUT scenario requires column 'PhaseOut' in Basic_data.csv")
        out["PhaseOut_effective"] = out["PhaseOut"]

    return out