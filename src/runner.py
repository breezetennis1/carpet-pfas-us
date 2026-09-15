from __future__ import annotations

from pathlib import Path
import pandas as pd
from src.paths import get_output_dirs

from src.config import PFAS_CONFIGS, CARPET_CONFIGS
from src.io_utils import load_basic_data, save_csv
from src.model import run_national_model
from src.postprocess import build_total_output, build_contribution_long
from src.scenario import apply_scenario


def run_all_cases(project_root: Path, scenario: str = "BAU") -> None:
    input_csv = project_root / "data" / "Basic_data.csv"
    dirs = get_output_dirs(project_root)
    output_dir = dirs["tables_national"]
    output_dir.mkdir(parents=True, exist_ok=True)

    base_df = load_basic_data(input_csv)
    scenario_df = apply_scenario(base_df, scenario)

    grouped_results = {
        "high": {},
        "central": {},
        "low": {},
    }

    for (subclass, estimate), config in PFAS_CONFIGS.items():
        carpet_config = CARPET_CONFIGS[estimate]
        result_df = run_national_model(scenario_df, config, carpet_config)
        grouped_results[estimate][subclass] = result_df

        filename = f"{subclass}_{estimate.upper()}_{scenario}.csv"
        save_csv(result_df, output_dir / filename)
        print(f"Saved: {filename}")

    total_high = build_total_output(grouped_results["high"])
    total_central = build_total_output(grouped_results["central"])
    total_low = build_total_output(grouped_results["low"])

    save_csv(total_high, output_dir / f"TOTAL_HIGH_ALL_SUBCLASSES_{scenario}.csv")
    save_csv(total_central, output_dir / f"TOTAL_CENTRAL_ALL_SUBCLASSES_{scenario}.csv")
    save_csv(total_low, output_dir / f"TOTAL_LOW_ALL_SUBCLASSES_{scenario}.csv")

    contrib_high = build_contribution_long(grouped_results["high"], "high")
    contrib_central = build_contribution_long(grouped_results["central"], "central")
    contrib_low = build_contribution_long(grouped_results["low"], "low")
    contribution_long = pd.concat([contrib_high, contrib_low], ignore_index=True)

    contribution_long["scenario"] = scenario
    save_csv(contribution_long, output_dir / f"PFAS_subclass_contribution_long_{scenario}.csv")

    print(f"Saved: TOTAL_HIGH_ALL_SUBCLASSES_{scenario}.csv")
    print(f"Saved: TOTAL_LOW_ALL_SUBCLASSES_{scenario}.csv")
    print(f"Saved: PFAS_subclass_contribution_long_{scenario}.csv")