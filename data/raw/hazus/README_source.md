# Hazus tract-level square footage by occupancy (source record)

- Source: FEMA Hazus Inventory National Database, data dictionary dated May 2023 (https://www.loc.gov/item/2024595033)
- Basis: NSI 2022 building inventory; 2020 census tract boundaries
- Local source path: `C:\Users\mchen48\Downloads\master-gdc-gdcdatasets-2024595033-2024595033\2024595033\National.gdb`
- Table (layer) name in National.gdb: `SquareFootageByOccupancyCensusTractLevel`
- Extracted: 2026-09-23 by extract_hazus_sqft_tract.py
- Output: `hazus_floor_area_m2_by_occupancy_tract.csv` (85,143 rows), SHA-256 `b543bfac740c78af59c40def09a24c50debf2e64f690bc356699cc64f5264dbb`
- Units: source **thousand sq ft** (fields `<OCCUPANCY>F`), converted to **m2** (fields `<OCCUPANCY>_m2`) with factor 92.90304 m2 per thousand sq ft; added `state_fips`, `county_fips` = Tract[:2], Tract[:5]
- Non-state FIPS kept (territories): ['00', '60', '66', '69', '72', '78']

## National totals, 50 states + DC (billion m2; CBECS 2018 commercial ~8.96 billion m2 for comparison)
- RES1-3 (Berrill-comparable): 21.08
- RES4-6: 0.65
- COM: 6.48
- EDU+GOV+REL: 2.44
- IND+AGR: 3.10

## Checks
- WARNING: non-11-digit / non-numeric Tract IDs present
- WARNING: 1 duplicated Tract IDs