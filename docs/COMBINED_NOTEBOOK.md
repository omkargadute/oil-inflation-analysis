# Final notebook: combined January 2000–October 2026 analysis

[notebooks/final_project.ipynb](../notebooks/final_project.ipynb) now uses one combined dataset throughout its main analysis. Current observations are no longer presented only as an extension after a 2024 report.

## Scope and training

- Four countries, 322 months each: 1,288 rows and 1,286 observed monthly outcomes.
- All EDA, main Gaussian/Student-t fits, tail/dependence checks, alternative lags and period fits use the combined calendar.
- Section 7 explicitly calls `fit_combined` to train the main models; Section 8 runs the comparison suite.
- Every model conditions on the first 13 observed months. Subsequent positions number 309 per country; two US outcomes are latent, leaving 307 observed US outcomes after that prefix.
- Disjoint forecast windows cover 2010–2014, 2015–2019, and 2020–October 2026. Training ends before each window. A separate through-2024 fit evaluates 2025 onward and provides the labeled historical comparison.
- Synthetic data stay separate. The existing 40-dataset stress study uses the same complete-calendar conditional AR1 likelihood, with independently tested kernel equivalence. New Gaussian/Student-t recovery fits also run.

## Missing-data likelihood

October 2025 US CPI is absent, so October and November monthly inflation remain missing. No interpolation or forward fill is performed. Earlier extension code restarted conditional segments; that implementation is now a historical reference.

The final model integrates unknown monthly outcomes under the AR likelihood. Given parameters and Normal/Gamma mixture weights, the missing outcomes have an exact multivariate Gaussian conditional. Gibbs updates sample them internally alongside parameters. No latent draw is exported as observed inflation. Later observed US outcomes inform the posterior, including transitions that depend on missing history.

Tests compare the latent conditional against analytic Gaussian AR covariance and verify complete-calendar Student-t transitions match the established sampler. R-hat and bulk/tail ESS thresholds apply to latent outcomes and parameters.

The likelihood targets observed monthly growth. It does not impose the known September–November CPI-level ratio as an additional nonlinear constraint, leaving some index-level information unused. The first 13 months are conditioning history, not likelihood targets. No exact stationary Student-t initial distribution is asserted.

## Forecast and diagnostic integrity

Forecast coefficients use only declared earlier samples. One-step predictions condition on observed lagged residuals. A score is unavailable if the target or a required lagged outcome is unavailable; latent test outcomes are never supplied as actual observations. Individual model tables report their counts; structural comparisons use identical eligible positions across all six dependence/oil variants within each country/window.

Paired circular 12-month block intervals resample within disjoint windows, preserving missing calendar positions. The final window is longer than five years; aggregate scores/RMSE use matched month counts, not equal weighting of unequal windows. Intervals condition on fitted forecasts. Revised inputs and inspected validation periods prevent real-time or confirmatory-test claims.

Innovation correlations use calendar-aligned available pairs. Ljung–Box checks use the longest contiguous observed innovation run, with dates/size reported. The US gap is not compressed into adjacent months. Posterior replications propagate innovations through the entire calendar and compare summaries at observed outcome positions.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m analysis.combined_analysis
.\.venv\Scripts\python.exe scripts/build_notebook.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The notebook precomputes fits if necessary. Inputs are local hash-verified snapshots. Caches in `.local-archive/combined-model-cache/` include source, data, priors/settings and Numba version. Changes invalidate relevant fits.

The canonical dataset is `data/processed/multicountry_monthly.csv`. Current results are in `results/combined_project/`, including findings, country estimates, fit/latent-state diagnostics, posterior checks and forecast comparisons. Earlier directories are historical references. Interest rates, identified supply shocks and causal country rankings remain outside this design.
