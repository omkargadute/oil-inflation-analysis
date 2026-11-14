# Oil prices and inflation across four countries

A Bayesian comparison of **the United States, China, Japan and Canada**, January 2000–December 2024. Each country has 300 complete monthly observations. Separate country models use the same three substantive variables: lagged Brent price changes, subsequent monthly CPI inflation, and prior global economic activity.

## Start here

- [Executed notebook](notebooks/oil_inflation.ipynb): four-country data validation, prior prediction, adjusted/unadjusted fits, simulation recovery, sampling diagnostics, predictive checks and holdout comparisons.
- [Project plan](docs/PROJECT_PLAN.md): estimand, common specification, assumptions and course limitations.
- [Analysis source](analysis/oil_inflation.py) and [numerical model](analysis/model.py). The notebook builder includes the model implementation so notebook exports contain the computational code.
- [Current results](results/FINDINGS.md) and [coefficient table](results/country_coefficients.csv).
- [Country availability audit](docs/COUNTRY_AVAILABILITY.md): broader 17-country coverage check.

## Consistent seasonality treatment

All four countries use **unadjusted national all-items CPI**, with calendar-month effects estimated separately within each country. We do not mix adjusted US data with unadjusted Chinese/Japanese data. Both confounder comparisons retain the same seasonal terms; only global-activity adjustment differs. This model estimates recurring monthly structure, not an officially adjusted CPI series. Moving holidays, changing seasonality and national index definitions remain limitations.

The Gaussian AR(1) baseline is exploratory. Parameter recovery and sampling quality do not establish causal identification. Shared oil and demand histories mean the 1,200 country-month observations are dependent. Country credible intervals are not a formal test of differences between countries. Heavy-tailed innovations remain future work.

## Reproduce

Python 3.11 was used. From the repository root:

```sh
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements.txt
python scripts/download_data.py --verify
python scripts/check_country_coverage.py
python scripts/build_notebook.py
```

The builder runs four chains for each adjusted/unadjusted and training-period country fit, plus separate simulation fits. It updates the notebook, analysis panel, tables and figures and can take several minutes. Inputs are included; reproduction needs no external data requests. To explicitly refresh official snapshots, run `python scripts/download_data.py --refresh`, then rerun the audit and notebook. Revisions can change results.

## Structure

```text
analysis/oil_inflation.py                 Four-country analysis cells
analysis/model.py                         Seasonal AR(1) likelihood and sampler
notebooks/oil_inflation.ipynb              Executed notebook, including model implementation
scripts/                                 Source verification, audit, notebook builder
data/raw/                                Global predictors and OECD CPI snapshots/provenance
data/processed/multicountry_monthly.csv    Main four-country analysis panel
data/processed/country_cpi_candidate_panel.csv  Broader availability-audit panel
results/                                 Current four-country tables and figures
docs/PROJECT_PLAN.md                      Scope, statistical model and limitations
requirements.txt                         Reproduction dependencies
```

The prior US-only analysis and outputs are retired; a local ignored archive preserves them. Interest rates are reserved for future work.

## Sources and attribution

- CPI: [OECD Data Explorer](https://data-explorer.oecd.org/), national all-items, monthly unadjusted indices. US/China use the COICOP 1999 flow; Japan/Canada use the COICOP 2018 flow. Each country uses a single complete series without cross-flow splicing.
- Brent: [EIA via FRED, MCOILBRENTEU](https://fred.stlouisfed.org/series/MCOILBRENTEU).
- Global activity: [Dallas Fed via FRED, IGREA](https://fred.stlouisfed.org/series/IGREA), a shipping-based industrial-commodity activity proxy, not GDP growth.

Exact source URLs, recorded retrieval timestamps and SHA-256 hashes are in `data/raw/manifest.json` and `data/raw/country_manifest.json`. Source data retain their providers' applicable terms; no blanket license is asserted over third-party data. This study uses revised observations and does not reconstruct real-time publication vintages.
