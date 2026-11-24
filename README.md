# Oil prices and inflation across four countries

The final analysis uses **January 2000–October 2026** for the United States, China, Japan and Canada: 322 months per country, 1,288 calendar rows and 1,286 observed monthly inflation outcomes. All main descriptive charts, Bayesian estimates and robustness checks use this combined period. Forecasts use chronological training samples and evaluate later observations through October 2026.

## Start here

- [Final executed notebook](notebooks/final_project.ipynb): nine sections covering data, EDA, causal assumptions, priors, synthetic validation, main training, posterior checks, chronological prediction and findings.
- [Combined-notebook guide](docs/COMBINED_NOTEBOOK.md): dataset, missing-state likelihood, model comparisons, outputs and reproduction.
- [Current findings](results/combined_project/FINDINGS.md) and [country estimates](results/combined_project/country_estimates.csv).
- [Jupyter setup](docs/NOTEBOOK_SETUP.md) and [project plan](docs/PROJECT_PLAN.md).

Separate country models use three substantive constructs: previous-month Brent price changes, monthly CPI inflation, and earlier global activity. Every country uses **unadjusted national all-items CPI**, with consistent calendar-month effects. Shared oil histories are not independent country exposures. Estimates remain conditional associations without sufficient causal-identification assumptions.

Missing US October 2025 CPI makes October and November monthly growth undefined. Raw gaps remain missing. Gaussian and Student-t AR models integrate these unknown outcomes as latent states during fitting, preserving later observations and the calendar. Forecast scoring never uses latent test outcomes as observed histories. The 2000–2024 estimates are a labeled historical comparison, not the main dataset.

## Reproduce

Python 3.11 was used. From the repository root:

```powershell
.\scripts\setup_notebook.ps1
.\scripts\start_jupyter.ps1
```

Choose **Python (Oil Inflation Project)** and open `notebooks/final_project.ipynb`.

```powershell
.\.venv\Scripts\python.exe -m analysis.combined_analysis
.\.venv\Scripts\python.exe scripts/build_notebook.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Precomputation is optional; the notebook fits models itself. The first run compiles and samples; unchanged runs reuse a content-addressed cache in `.local-archive/combined-model-cache/`. Every retained parameter and latent outcome must pass R-hat and effective-sample-size checks. Saved snapshots make notebook execution independent of network access.

To deliberately fetch newer releases, run `.\.venv\Scripts\python.exe scripts/download_latest.py`, inspect the availability/missingness audit, and review the notebook's frozen October 2026 endpoint before rebuilding. The older `download_data.py --refresh` retains the historical OECD query cutoff and is not the current-data workflow.

## Structure

```text
analysis/final_project.py                   Final notebook source in percent-cell format
analysis/combined_sampler.py                Gaussian/Student-t sampler with latent missing outcomes
analysis/combined_model.py                  Fitting, posterior checks and calendar-safe forecasts
analysis/combined_analysis.py               Combined-period comparison suite
analysis/latest_data.py                     Current snapshots and gap-preserving transformations
notebooks/final_project.ipynb                Final executed notebook
data/raw/latest/                            Current official snapshots and byte hashes
data/processed/multicountry_monthly.csv      Main combined dataset
data/processed/multicountry_monthly_latest.csv  Same calendar, compatibility export
results/combined_project/                   Current results, figures and fit diagnostics
results/latest/                            Source audits and earlier extension results
results/strengthening/                     Earlier robustness and 40-dataset simulation study
results/final_project/                     Earlier computational report, historical reference
```

Earlier exploratory notebooks/scripts are historical references; their dates and estimates are not the combined final results. The code-inclusive submission PDF and group screencast remain separate deliverables.

## Sources

- CPI: [OECD Data Explorer](https://data-explorer.oecd.org/), monthly national all-items unadjusted indices. US/China use COICOP 1999; Japan/Canada use COICOP 2018, without within-country splicing.
- Monthly Brent: [EIA via FRED, MCOILBRENTEU](https://fred.stlouisfed.org/series/MCOILBRENTEU).
- Daily Brent, context only: [DCOILBRENTEU](https://fred.stlouisfed.org/series/DCOILBRENTEU).
- Activity: [Dallas Fed via FRED, IGREA](https://fred.stlouisfed.org/series/IGREA), not GDP growth.

URLs, retrieval timestamps and SHA-256 hashes are in `data/raw/latest/manifest.json`. Official CSV bytes are preserved by Git attributes. Source data retain provider terms; no blanket license is asserted over third-party data. Revised snapshots do not reconstruct real-time vintages.
