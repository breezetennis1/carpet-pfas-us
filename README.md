# Per- and Polyfluoroalkyl Substances (PFAS) in Carpet: Disparities in In-Use Emissions and Landfill Burden Across the United States

## Authors

Mingming Chen¹†, Hossein Gazmeh²†, Yuhao Wang¹, Peter Berill³, Xinwu Qian², and Daqian Jiang¹*

¹ Department of Civil, Construction and Environmental Engineering, The University of Alabama, Tuscaloosa, AL, USA

² Department of Civil and Environmental Engineering, Rice University, Houston, TX, USA

³ Institute of Environmental Sciences (CML), Leiden University, Leiden, the Netherlands

† These authors contributed equally to this work.

* Corresponding author: djiang6@ua.edu

---

# Overview

This repository contains the complete reproducible computational package associated with the manuscript:

**Per- and Polyfluoroalkyl Substances (PFAS) in Carpet: Disparities in In-Use Emissions and Landfill Burden Across the United States**

The package reproduces:

- National-scale PFAS material flow analysis (MFA)
- County-level PFAS allocation
- Census tract-level PFAS allocation
- Transportation assignment from census tracts to landfills
- Landfill PFAS accumulation
- Landfill PFAS leachate releases
- Spatial distribution analyses
- Environmental justice analyses
- Monte Carlo uncertainty analysis
- Sobol global sensitivity analysis
- Figure generation

---

# Repository Structure

```text
carpet_PFAS_data_processing_v1/
│
├── data/
│   ├── Basic_data.csv
│   ├── county/
│   ├── census_tracts/
│   ├── landfill_central_BAU/
│   └── tl_2020_us_state/
│
├── outputs/
│   ├── figures/
│   └── tables/
│
├── src/
│   ├── __init__.py
│   ├── allocation_census_tract.py
│   ├── allocation_county.py
│   ├── config.py
│   ├── io_utils.py
│   ├── landfill_PFAS_calculation.py
│   ├── model.py
│   ├── monte_carlo.py
│   ├── paths.py
│   ├── plot_county_distribution.py
│   ├── plot_figure2.py
│   ├── plot_landfill_figure.py
│   ├── postprocess.py
│   ├── runner.py
│   ├── scenario.py
│   ├── sobol_analysis.py
│   └── transportation_CT_to_landfills.py
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
│
├── main.py
├── README.md
└── requirements.txt
```

The `src/` directory contains the source code modules.

The numbered scripts (`01_run_national.py` through `10_run_sobol.py`) provide the recommended execution order for reproducing the complete workflow.

---

# Software Requirements

The workflow was developed and tested using:

- Python 3.11
- Windows 11

Install required packages:

```bash
pip install -r requirements.txt
```

---

# Workflow

The scripts should be executed in the following order.

---

## Step 1: National PFAS Material Flow Analysis

```bash
python 01_run_national.py
```

Outputs:

```text
outputs/tables/national/
```

Generates national-scale annual PFAS stocks, flows, emissions, and landfill inputs.

---

## Step 2: County-Level Allocation

```bash
python 02_run_county_allocation.py
```

Outputs:

```text
outputs/tables/county/
```

Allocates national PFAS quantities to U.S. counties.

---

## Step 3: Census Tract Allocation

```bash
python 03_run_census_tract_allocation.py
```

Outputs:

```text
outputs/tables/census_tracts/
```

Allocates county-level PFAS quantities to census tracts.

---

## Step 4: Transportation Assignment

```bash
python 04_transportation_CT_to_landfills.py
```

Outputs:

```text
outputs/tables/transportation_assignment/
```

Assigns carpet waste and associated PFAS from census tracts to municipal solid waste landfills.

---

## Step 5: Landfill PFAS Calculation

```bash
python 05_run_landfill_PFAS_calculation.py
```

Outputs:

```text
outputs/tables/national/landfill_central_BAU/
```

Calculates:

- Landfill PFAS accumulation
- Landfill PFAS leachate releases
- Landfill accumulation after leachate removal

---

## Step 6: National Figures

```bash
python 06_run_plot_figure2.py
```

Outputs:

```text
outputs/figures/
```

Generates national-scale PFAS stock and flow figures.

---

## Step 7: County Distribution Figures

```bash
python 07_run_plot_county_distribution.py
```

Outputs:

```text
outputs/figures/
```

Generates county-level PFAS distribution maps.

---

## Step 8: Landfill Figures

```bash
python 08_run_plot_landfill_figure.py
```

Outputs:

```text
outputs/figures/national/
```

Generates landfill accumulation and leachate figures.

---

## Step 9: Monte Carlo Uncertainty Analysis

```bash
python 09_run_monte_carlo.py
```

Outputs:

```text
outputs/tables/monte_carlo/
```

Performs uncertainty analysis for key model outputs.

---

## Step 10: Sobol Global Sensitivity Analysis

```bash
python 10_run_sobol.py
```

Outputs:

```text
outputs/tables/sobol/
```

Performs variance-based global sensitivity analysis.

---

# Scenarios

The framework supports:

### Policy Scenarios

- Business-as-usual (BAU)
- 24% recycling scenario
- PFAS phase-out scenario

### Estimate Levels

- High estimate
- Central estimate
- Low estimate

depending on the parameter configuration used.

---

# Major Outputs

The workflow generates:

### National-scale outputs

- PFAS in-use stock
- PFAS in-use emissions
- PFAS landfill accumulation
- PFAS landfill leachate releases

### Spatial outputs

- County-level PFAS distributions
- Census tract-level PFAS distributions
- Landfill-specific PFAS accumulation
- Landfill-specific PFAS leachate releases

### Uncertainty outputs

- Monte Carlo simulations
- Sobol sensitivity indices

### Figures

- National temporal trends
- County distribution maps
- Landfill accumulation maps
- Landfill leachate maps

---

# Data Sources

Major input datasets include:

- U.S. Census Bureau
- American Community Survey (ACS)
- EPA EJScreen
- EPA Landfill Methane Outreach Program (LMOP)
- State landfill datasets
- Literature-derived PFAS concentrations
- Literature-derived PFAS emission parameters

Detailed references are provided in the manuscript and Supporting Information.

---

# Data and Code Availability

All data, source code, workflow scripts, and documentation required to reproduce the results presented in this study are publicly available through the project repository:

GitHub:

https://github.com/hogazme/carpet_pfas_us

Zenodo:

https://zenodo.org/records/14006378

---

# Contact

For questions regarding the code or data:

**Daqian Jiang**

Department of Civil, Construction and Environmental Engineering

The University of Alabama

Tuscaloosa, AL, USA

Email: djiang6@ua.edu