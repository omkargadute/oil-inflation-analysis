# Oil prices and US inflation

A three-variable Bayesian analysis of monthly Brent price changes, subsequent US consumer price inflation, and prior global economic activity, January 2000–December 2024.

## Start here

- [Executed notebook](notebooks/oil_inflation.ipynb): data checks, simulation recovery, Bayesian estimation, diagnostics, prediction, and robustness.
- [Project plan](docs/PROJECT_PLAN.md): research question, assumptions, completed work, and next steps.
- [Country availability audit](docs/COUNTRY_AVAILABILITY.md): verified China/Japan coverage and other candidates; requirements for comparable international analysis.
- [Analysis source](analysis/oil_inflation.py): the notebook's executable cells.
- [Results](results/): diagnostic tables and figures.

## Current findings

All three aligned series contain 300 complete US months. The Gaussian AR(1) model estimates a positive association of about **0.0945 monthly inflation percentage points per 10-percentage-point higher oil change**, with a model-conditional 95% equal-tailed credible interval of **[0.0602, 0.1284]**. Sampling diagnostics and known-parameter recovery pass. Residual dependence and negative-tail prediction remain problematic; adjustment for the demand proxy barely changes the estimate.

These are exploratory associations under strong causal assumptions. They do not identify an oil supply shock or establish a causal effect. A heavy-tailed error model is planned, not yet implemented. Interest rates and international comparisons are outside the current analysis.

## Reproduce

Python 3.11 was used. From the repository root:

```sh
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements.txt
python scripts/download_data.py --verify
python scripts/build_notebook.py
```

The notebook builder executes the analysis and updates the notebook, processed data, and results. It runs four Metropolis chains per fitted Bayesian model and can take several minutes. All input snapshots are included; rebuilding needs no data downloads.

To explicitly refresh source snapshots, run `python scripts/download_data.py --refresh`. This updates provenance and can change results because official series are revised. Rerun the notebook afterward.

## Structure

```text
analysis/oil_inflation.py       Analysis cells
notebooks/oil_inflation.ipynb   Executed notebook
scripts/                      Download verification and notebook builder
data/raw/                     Three official data snapshots and manifest
data/processed/us_monthly.csv  Aligned analysis data
results/                      Tables and figures
docs/PROJECT_PLAN.md           Scope and limitations
requirements.txt              Reproduction dependencies
```

## Data and attribution

Brent: [EIA via FRED, MCOILBRENTEU](https://fred.stlouisfed.org/series/MCOILBRENTEU).
US seasonally adjusted CPI: [BLS via FRED, CPIAUCSL](https://fred.stlouisfed.org/series/CPIAUCSL).
Global activity: [Dallas Fed via FRED, IGREA](https://fred.stlouisfed.org/series/IGREA), a shipping-based commodity-market activity proxy, not GDP growth.
Exact download URLs, recorded retrieval timestamps, and SHA-256 hashes are in [the manifest](data/raw/manifest.json). Source data retain their providers' applicable terms; no blanket license is asserted over third-party data.
