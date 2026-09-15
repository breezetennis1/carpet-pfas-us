from pathlib import Path


def get_output_dirs(project_root: Path) -> dict:
    base = project_root / "outputs"

    dirs = {
        "tables_national": base / "tables" / "national",
        "tables_monte_carlo": base / "tables" / "monte_carlo",
        "tables_sobol": base / "tables" / "sobol",
        "tables_county": base / "tables" / "county",
        "tables_census": base / "tables" / "census_tracts",

        "figures_main": base / "figures" / "main_text",
        "figures_si": base / "figures" / "SI",
        "figures_sobol": base / "figures" / "sobol",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs