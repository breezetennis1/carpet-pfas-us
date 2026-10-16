# Per- and Polyfluoroalkyl Substances (PFAS) in Carpet: Disparities in In-Use Emissions and Landfill Burden Across the United States

## Authors

Mingming Chen¹†, Hossein Gazmeh²†, Yuhao Wang¹, Peter Berill³, Xinwu Qian², and Daqian Jiang¹*

¹ Department of Civil, Construction and Environmental Engineering, The University of Alabama, Tuscaloosa, AL, USA

² Department of Civil and Environmental Engineering, Rice University, Houston, TX, USA

³ Institute of Environmental Sciences (CML), Leiden University, Leiden, the Netherlands

† These authors contributed equally to this work.

\* Corresponding author: djiang6@ua.edu

---

# Overview

This repository contains the computational package for the manuscript:

**Per- and Polyfluoroalkyl Substances (PFAS) in Carpet: Disparities in In-Use Emissions and Landfill Burden Across the United States**

The package reproduces:

- National-scale PFAS material flow analysis (MFA)
- County-level PFAS allocation
- Census tract-level PFAS allocation
- Transportation assignment from census tracts to landfills
- Landfill PFAS accumulation and leachate releases
- Monte Carlo uncertainty analysis and Sobol global sensitivity analysis
- Landfill-level validation against EPA LMOP data
- Main-text and Supporting Information figures

---

# Repository Structure

```text
carpet-pfas-us/
│
├── data/
│   ├── Basic_data.csv                 national parameter table
│   ├── county/                        county population and floor area
│   ├── processed/                     tract and county commercial floor area
│   ├── raw/                           national inputs, HAZUS, CBECS, Census estimates
│   ├── landfill_central_BAU/          landfill attributes
│   ├── validation/                    LMOP data and previous allocation results
│   ├── census_tracts/                 (Zenodo data record)
│   ├── transport_assignment/          (Zenodo data record)
│   └── tl_2020_us_state/              (Zenodo data record)
│
├── src/                               source code modules
├── tools/                             data preparation and verification utilities
│
├── 01_run_national.py
├── 02_run_county_allocation.py
├── 03_run_census_tract_allocation.py
├── 04_transportation_CT_to_landfills.py
├── 05_run_landfill_PFAS_calculation.py
├── 06_run_plot_figure2.py
├── 07_run_plot_county_distribution.py
├── 08_run_plot_landfill_figure.py
├── 09_run_monte_carlo.py
├── 10_run_sobol.py
├── 11_plot_mc_sobol_2x4.py
├── 12_plot_SI_supplementary_figures.py
├── 13_validation.py
│
├── environment.yml
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

The numbered scripts give the execution order. All outputs are written to `outputs/`, which is not tracked and is recreated by running the scripts.

---

# Data

Inputs up to a few tens of MB are included in `data/`. The larger inputs are archived in a separate Zenodo data record:

**Zenodo data record: [DOI to be added]**

Download and extract it so that the following folders sit inside `data/`:

| Folder | Contents | Used by |
|---|---|---|
| `data/census_tracts/population/` | Census tract population, 1990-2060 (`CT_Population_<year>.csv`) | step 3 |
| `data/transport_assignment/annual_PFAS_accum_central_BAU/` | Tract-to-landfill assignment, PFAS (kg/yr) | step 5 |
| `data/transport_assignment/annual_carpt_volume_central_BAU/` | Tract-to-landfill assignment, carpet area (million m²/yr) | step 5 |
| `data/county/tl_2020_us_county/` | 2020 TIGER/Line county boundaries | step 7 |
| `data/tl_2020_us_state/` | 2020 TIGER/Line state boundaries | step 8 |

The TIGER/Line shapefiles can also be downloaded from the U.S. Census Bureau ([county](https://www2.census.gov/geo/tiger/TIGER2020/COUNTY/tl_2020_us_county.zip), [state](https://www2.census.gov/geo/tiger/TIGER2020/STATE/tl_2020_us_state.zip)).

---

# Software Requirements

The workflow was developed and tested using:

- Python 3.9.19
- Windows 11

Create the pinned conda environment (recommended; `mkl=2024.1.0` is required, see `environment.yml`):

```bash
conda env create -f environment.yml
conda activate carpet-pfas-rev
```

or install the required packages with pip:

```bash
pip install -r requirements.txt
```

---

# Workflow

Run the scripts from the repository root in the following order.

| Step | Script | Description | Outputs |
|---|---|---|---|
| 1 | `01_run_national.py` | National PFAS material flow analysis: stocks, flows, emissions and landfill inputs | `outputs/tables/national/` |
| 2 | `02_run_county_allocation.py` | Allocation of national quantities to counties | `outputs/tables/county/` |
| 3 | `03_run_census_tract_allocation.py` | Allocation of county quantities to census tracts | `outputs/tables/census_tracts/` |
| 4 | `04_transportation_CT_to_landfills.py` | Assignment of carpet waste and PFAS from census tracts to municipal solid waste landfills (code to be added; results are provided in `data/transport_assignment/`) | |
| 5 | `05_run_landfill_PFAS_calculation.py` | Landfill PFAS accumulation and leachate releases | `outputs/tables/landfill_central_BAU/` |
| 6 | `06_run_plot_figure2.py` | National stock and flow figures | `outputs/figures/` |
| 7 | `07_run_plot_county_distribution.py` | County distribution maps | `outputs/figures/county/` |
| 8 | `08_run_plot_landfill_figure.py` | Landfill accumulation and leachate maps | `outputs/figures/landfill/` |
| 9 | `09_run_monte_carlo.py` | Monte Carlo uncertainty analysis | `outputs/tables/monte_carlo/` |
| 10 | `10_run_sobol.py` | Sobol global sensitivity analysis | `outputs/tables/sobol/` |
| 11 | `11_plot_mc_sobol_2x4.py` | Monte Carlo and Sobol figure | `outputs/figures/` |
| 12 | `12_plot_SI_supplementary_figures.py` | Supporting Information figures | `outputs/figures/SI/` |
| 13 | `13_validation.py` | Landfill-level validation against EPA LMOP waste-in-place data | `outputs/figures/validation/` |

---

# Scenarios

The framework supports:

### Policy Scenarios

- Business-as-usual (BAU)
- 24% recycling scenario
- PFAS phase-out scenario

### Estimate Levels

- High estimate
- Central estimate (geometric mean of the low and high estimates)
- Low estimate

---

# Data Sources

Major input datasets include:

- U.S. Census Bureau (population estimates, TIGER/Line boundaries)
- American Community Survey (ACS)
- FEMA HAZUS general building stock
- EIA Commercial Buildings Energy Consumption Survey (CBECS)
- EPA EJScreen
- EPA Landfill Methane Outreach Program (LMOP)
- State landfill datasets
- Literature-derived PFAS concentrations and emission parameters

Detailed references are provided in the manuscript and Supporting Information.

---

# Data and Code Availability

GitHub: https://github.com/breezetennis1/carpet-pfas-us

Zenodo (code): [DOI to be added]

Zenodo (data): [DOI to be added]

---

# License

The code is released under the MIT License (see `LICENSE`). Third-party datasets in `data/` remain subject to the terms of their original providers.

---

# Contact

For questions regarding the code or data:

**Daqian Jiang**

Department of Civil, Construction and Environmental Engineering

The University of Alabama

Tuscaloosa, AL, USA

Email: djiang6@ua.edu
