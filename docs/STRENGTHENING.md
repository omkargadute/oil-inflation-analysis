# Model-strengthening experiments

Run `.\.venv\Scripts\python.exe scripts/run_strengthening.py`, then rebuild the final notebook with `scripts/build_notebook.py`. Results are saved in `results/strengthening/`; the notebook displays the comparisons alongside the original analysis. The first run compiles and fits models; subsequent runs reuse content-addressed local caches. No source data are downloaded or altered by the experiment suite.

## Comparable likelihoods

All extension models condition on the first 13 months. A full-data fit therefore targets 287 outcomes, beginning February 2001. This is deliberately different from the original 300-outcome baseline with its initial-residual density. Compare extensions to the matched conditional AR1 model, rather than attributing every difference from the original fit to a dependence change.

Writing the regression residual as $r_t=Y_t-X_t b$, we fit:

- AR1: $(1-\phi B)r_t=\epsilon_t$.
- AR2: $(1-a_1B-a_2B^2)r_t=\epsilon_t$, with $a_1=\kappa_1(1-\kappa_2)$ and $a_2=\kappa_2$.
- Seasonal AR: $(1-\phi B)(1-\Phi B^{12})r_t=\epsilon_t$.

Innovations follow $t_\nu(0,\sigma)$, where $\sigma$ is a scale. Independent uniform priors on the partial autocorrelations, or on $\phi,\Phi$, ensure stability. Regression and scale priors match the original Student-t specification. Gaussian scale-mixture Gibbs updates and Metropolis updates sample four chains; every retained fit must have rank-normalized R-hat below 1.01 and bulk/tail ESS above 400 for every parameter. Failed diagnostics trigger longer runs. Numerical tests independently verify the density, stability transformation and compiled conditional target. Numba compilation changes the random-number implementation, with its version and engine source included in cache keys.

## Five experiment families

1. **Dependence:** AR1, AR2 and multiplicative seasonal AR, with ACF/PACF, Ljung-Box checks and conditional posterior replications. Diagnostic p-values are approximate and exploratory after estimation; they do not prove adequacy.
2. **Tails:** fixed degrees of freedom 3, 5 and 10 under AR1. The scale prior is held fixed, so innovation variance changes with nu: this is not a variance-matched pure tail comparison. No tail choice is selected for a stronger oil coefficient.
3. **Prediction:** expanding training samples end in 2009, 2014 and 2019; the following five years are evaluated without coefficient refitting. Each structure has a matched no-oil model. Paired circular 12-month blocks, 2,000 replicates, estimate uncertainty in oil's RMSE gain. Aggregate resampling stays within windows and uses common month indices across countries. These revised-data conditional forecasts use observed lagged residuals; they are not real-time forecasts. Inspected validation periods are not an untouched test set.
4. **Repeated recovery:** ten independent datasets in each of four combinations of weak/strong confounding and low/high persistence. Adjusted and omitted-confounder fits give bias, estimation RMSE and interval coverage. Wilson intervals quantify Monte Carlo uncertainty. Forty datasets demonstrate behavior under a known data-generating process, not calibration under every misspecification; ten per cell cannot establish precise 95% coverage.
5. **Sensitivity and event audit:** oil lags two and three shift the pre-treatment demand proxy correspondingly. Separate pre/post-2020 Bayesian fits use consecutive calendars; the latter conditions on 13 pre-2020 months and targets all 60 post-2020 outcomes. Every country's minimum/maximum observation is retained and flagged if it lies in the conditioning prefix. Changing the lag changes the estimand; period differences are not a formal interaction test.

See [generated findings](../results/strengthening/FINDINGS.md), `fit_registry.csv` for fit-level sampling checks, and `sampling_diagnostics.csv` for parameter-level evidence. Models are judged jointly by prediction, residual behavior and coefficient stability. Selection using these comparisons remains exploratory. All models retain the original confounding and causal-identification limitations.

## Extreme-observation source audit

The OECD snapshot and transformation checks cover every observation. Independent release checks below are a separate corroboration, sometimes only to published rounding. Differences in index base, revisions and rounding prevent demanding identical final decimals. Unadjusted monthly changes must not be compared to seasonally adjusted headline figures.

| Country/date | Snapshot monthly change | Independent primary-source check |
|---|---:|---|
| US, November 2008 | -1.9153% | [BLS release](https://www.bls.gov/news.release/archives/cpi_12162008.htm) reports an unadjusted 1.9% decrease. |
| US, June 2022 | 1.3736% | [BLS release tables](https://www.bls.gov/news.release/archives/cpi_07132022.htm) report 1.4% unadjusted; the 1.3% headline is seasonally adjusted. |
| China, February 2008 | 2.6000% | [NBS release](https://www.stats.gov.cn/english/NewsEvents/200803/t20080311_26001.html) explicitly reports 2.6% growth from January. |
| China, March 2000 | -1.3889% | Snapshot provenance verified; an independent contemporary monthly release was not recovered. No event mechanism asserted. This month is in the extensions' conditioning prefix. |
| Japan, April 2014 | 2.1053% | [Statistics Bureau handbook](https://www.stat.go.jp/english/data/handbook/pdf/2014all.pdf) confirms the April consumption-tax increase from 5% to 8%. This corroborates context, not the exact all-items monthly figure. |
| Japan, November 2008 | -0.8028% | Snapshot provenance verified; exact independent contemporary monthly release not recovered. |
| Canada, October 2008 | -1.0372% | [Statistics Canada analysis](https://www150.statcan.gc.ca/n1/pub/62-001-x/2008010/6500338-eng.htm) reports an unadjusted 1.0% monthly decline. |
| Canada, March 2022 | 1.4305% | [Statistics Canada release](https://www150.statcan.gc.ca/n1/daily-quotidien/220420/dq220420a-eng.htm) reports an unadjusted 1.4% monthly increase. |

Source context does not establish that oil caused an individual extreme. Genuine events and unexplained observations stay in the likelihood where the conditioning design permits; robust weights are diagnostics, not evidence of data corruption.

Alternative-lag fits each contain a single oil regressor. They do not estimate a distributed-lag response, cumulative pass-through or an impulse response. A negative coefficient at one lag must not be added to another model's coefficient to construct a cumulative causal effect.
