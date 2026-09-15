from __future__ import annotations

import pandas as pd


CORE_METRICS = [
    "F_stock",
    "F6",
    "F_accu_landfill",
    "PFAS_landfill_leachate",
]


def build_total_output(result_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    subclasses = list(result_dict.keys())
    years = result_dict[subclasses[0]]["year"].copy()

    total = pd.DataFrame({"year": years})

    for metric in CORE_METRICS:
        total[metric] = 0.0
        for subclass in subclasses:
            if metric in result_dict[subclass].columns:
                total[metric] += result_dict[subclass][metric].fillna(0.0)

    return total


def build_contribution_long(result_dict: dict[str, pd.DataFrame], estimate: str) -> pd.DataFrame:
    frames = []

    metric_map = {
        "F_stock": "in_use_stock",
        "F6": "in_use_emission",
        "F_accu_landfill": "landfill_accumulation",
        "PFAS_landfill_leachate": "landfill_emission",
    }

    for subclass, df in result_dict.items():
        for old_col, metric_name in metric_map.items():
            temp = pd.DataFrame({
                "year": df["year"],
                "subclass": subclass,
                "estimate": estimate,
                "metric": metric_name,
                "value": df[old_col] if old_col in df.columns else 0.0,
            })
            frames.append(temp)

    return pd.concat(frames, ignore_index=True)