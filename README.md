# Concise‑Representations‑of‑Complex‑Data

Surrogate and baseline pipelines for DDQ longwave radiative transfer on CKDMIP evaluation profiles. The production surrogate is the 6‑term log‑polynomial κₘ(ν; ln p, ln T) for CO₂ and O₃; PLS experiments are kept only for reference.

## Contents (this folder)
- `environment.yml` — conda env (`ddq_fluxsim_tutorial`) with pyarts, numpy/xarray, cvxpy, sklearn, matplotlib.
- `util.py` — shared library: constants, ARTS/LBL wrappers, log‑poly fitting, cost functions, caches, diagnostics.
- `fitting.ipynb` — fits log‑poly κₘ per species; writes coeffs and grids to `fit_cache/`.
- `inference.ipynb` — evaluates surrogate vs DDQ and dense baselines; uses caches in `lbl_ddq/`, `lbl_dense/`.
- Data/caches:  
  - `DDQ_LW_present.h5`, `ckdmip_evaluation1_concentrations_present.nc` (inputs).  
  - `fit_cache/` (co2/o3 fit_data, κ grids, PLS preds).  
  - `lbl_ddq/` (per‑column DDQ LBL caches), `lbl_dense/dense_lbl_cached_in_drive` (per‑column dense LBL caches stored in google drive).  
  - `pyarts-fluxes/`, `ddq-data-paper/` (supporting artifacts).

## Quick start
```bash
conda env create -f environment.yml
conda activate ddq_fluxsim_tutorial
```

## Typical use
1) Run `fitting.ipynb` to regenerate log‑poly coefficients (optional if `fit_cache/` already present).  
2) Run `inference.ipynb` to build/load DDQ & dense baselines and compute global costs/diagnostics.

## Notes
- Surrogate of record: log‑poly; PLS results in `fit_cache/` are exploratory and not used in production.  
- Caches speed reruns; delete specific files if you need fresh LBL or fits.  
- Extend constants in util.py to scope up. For example extend `MOLAR_MASS = {"CO2": 44.0095e-3, "O3": 47.9982e-3}` and `SPECIES = ["CO2", "O3"]` to fit more species
