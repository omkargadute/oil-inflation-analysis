# Oil prices and inflation across four countries

A Bayesian comparison of **the United States, China, Japan and Canada**, January 2000–December 2024. Each country has 300 complete monthly observations. Separate country models use the same three substantive variables: lagged Brent price changes, subsequent monthly CPI inflation, and prior global economic activity.

## Start here

- [Current-release extension](docs/LATEST_DATA.md): verified monthly data through October 2026, daily Brent through November 17, calendar-safe handling of missing US October 2025 CPI, updated coefficients and fixed-training recent forecasts. The original 2000–2024 study remains a reproducible reference.
- [Model-strengthening findings](results/strengthening/FINDINGS.md): dependence/tail comparisons, three rolling windows, 40 independent simulation datasets, lag/period sensitivity and retained-outlier audit.

- [Completed computational notebook](notebooks/final_project.ipynb): all nine report sections, Gaussian/Student-t estimation, synthetic validation, posterior checks, sensitivity, matched no-oil forecasts, and conclusions. The appendix includes the shared computational source.
- [Jupyter setup and workflow](docs/NOTEBOOK_SETUP.md): isolated `.venv`, dedicated kernel, launch and rebuild commands.
- [Exploratory notebook](notebooks/oil_inflation.ipynb): earlier four-country fits, simulation recovery, diagnostics and holdout comparisons; retained as a reference, not the final submission.
- [Project plan](docs/PROJECT_PLAN.md): estimand, common specification, assumptions and course limitations.
- [Final report source](analysis/final_project.py), [Gaussian model](analysis/model.py), and [Student-t model and evaluation](analysis/robust_model.py).
- [Current findings](results/final_project/FINDINGS.md), [country estimates](results/final_project/country_estimates.csv), and [matched oil/no-oil forecast comparison](results/final_project/oil_forecast_comparison.csv).
- [Country availability audit](docs/COUNTRY_AVAILABILITY.md): broader 17-country coverage check.

## Consistent seasonality treatment

All four countries use **unadjusted national all-items CPI**, with calendar-month effects estimated separately within each country. We do not mix adjusted US data with unadjusted Chinese/Japanese data. Both confounder comparisons retain the same seasonal terms; only global-activity adjustment differs. This model estimates recurring monthly structure, not an officially adjusted CPI series. Moving holidays, changing seasonality and national index definitions remain limitations.

The completed analysis compares Gaussian AR(1) errors with Student-t innovations at fixed five degrees of freedom. All 32 simulation/full-data/training fits pass sampling checks; model adequacy remains imperfect. Heavy tails help some extreme-value checks but leave residual dependence. Matched Student-t forecasts improve with oil in China, show negligible/mixed change in Japan, and worsen in the US and Canada over 2020–2024. These exploratory predictive results do not identify causal effects.

Shared oil and demand histories mean the 1,200 rows are dependent. Country intervals are not a test of country differences. Student-t initial residuals have matched AR variance, not an asserted exact stationary t-AR distribution. Degrees of freedom, seasonality, omitted confounding and regime changes remain limitations. Submission-PDF pagination and the group screencast are separate deliverables.

## Set up Jupyter and reproduce

Python 3.11 was used. From the repository root:

```powershell
.\scripts\setup_notebook.ps1
.\scripts\start_jupyter.ps1
```

Open `notebooks/final_project.ipynb` with **Python (Oil Inflation Project)**. The setup script uses `requirements-lock.txt` when available, checks dependencies and registers the local kernel; activation is optional.

Rebuild the active report with `.\.venv\Scripts\python.exe scripts/build_notebook.py`. To reproduce the earlier full exploratory run instead, add `--target exploratory`; that slower run fits four chains per country/model and separate simulations. Development-report outputs are stored separately in `results/final_project/`.

Final fits use a local content-addressed cache in `.local-archive/final-model-cache/`, keyed to model code, data, priors and sampler settings. A clean clone fits all models on its first run; unchanged subsequent runs reuse them transparently. To precompute fits, run `.\.venv\Scripts\python.exe scripts/fit_final_models.py`. Run independent density tests with `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.

Inputs are included; reproduction needs no external data requests. To explicitly refresh snapshots, run `.\.venv\Scripts\python.exe scripts/download_data.py --refresh`, then rerun the coverage audit and the desired notebook. Revisions can change results.

## Structure

```text
analysis/oil_inflation.py                 Four-country analysis cells
analysis/final_project.py                 Completed computational report source
analysis/data.py                          Verified calendar-aligned data preparation
analysis/model.py                         Seasonal AR(1) likelihood and sampler
analysis/robust_model.py                  Student-t sampler, cache and predictive evaluation
analysis/strengthening.py                 Five model-strengthening experiment families
analysis/latest_data.py                   Current snapshot verification and gap-preserving preparation
analysis/latest_model.py                  Segmented AR1 likelihood and recent forecasts
analysis/latest_analysis.py               Current-release analysis and generated findings
notebooks/final_project.ipynb              Executed nine-section computational report
notebooks/oil_inflation.ipynb              Earlier exploratory analysis
scripts/                                 Source verification, audit, notebook builder
data/raw/                                Global predictors and OECD CPI snapshots/provenance
data/processed/multicountry_monthly.csv    Main four-country analysis panel
data/processed/country_cpi_candidate_panel.csv  Broader availability-audit panel
results/final_project/                   Current final computational report outputs
results/strengthening/                   Matched conditional robustness experiments
results/latest/                          Current-release results and source/gap/revision audits
results/                                 Earlier exploratory outputs and coverage audit
docs/PROJECT_PLAN.md                      Scope, statistical model and limitations
requirements.txt                         Reproduction dependencies
requirements-dev.txt                     Jupyter development dependencies
requirements-lock.txt                    Resolved isolated environment
```

The prior US-only analysis and outputs are retired; a local ignored archive preserves them. Interest rates are reserved for future work.

## Sources and attribution

The expanded [strengthening experiments](docs/STRENGTHENING.md) test dependence, tail shape, three rolling prediction windows, repeated simulation recovery, and lag/period sensitivity. Run `.\.venv\Scripts\python.exe scripts/run_strengthening.py` to precompute them. Tables, diagnostic plots and generated findings are in `results/strengthening/` and are displayed in the final notebook. These matched conditional fits use a common 13-month history and retain the original causal limitations.

- CPI: [OECD Data Explorer](https://data-explorer.oecd.org/), national all-items, monthly unadjusted indices. US/China use the COICOP 1999 flow; Japan/Canada use the COICOP 2018 flow. Each country uses a single complete series without cross-flow splicing.
- Brent: [EIA via FRED, MCOILBRENTEU](https://fred.stlouisfed.org/series/MCOILBRENTEU).
- Global activity: [Dallas Fed via FRED, IGREA](https://fred.stlouisfed.org/series/IGREA), a shipping-based industrial-commodity activity proxy, not GDP growth.

Exact source URLs, recorded retrieval timestamps and SHA-256 hashes are in `data/raw/manifest.json` and `data/raw/country_manifest.json`. Source data retain their providers' applicable terms; no blanket license is asserted over third-party data. This study uses revised observations and does not reconstruct real-time publication vintages.
