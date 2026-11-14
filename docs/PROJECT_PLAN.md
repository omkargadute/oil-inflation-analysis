# Four-country oil–inflation project plan

## Scope and question

**United States, China, Japan and Canada, January 2000–December 2024.** Fit the same model separately in each country, using 300 complete months per country. International comparison is part of the analysis; interest rates are future work.

Question: under the proposed causal assumptions, how does a 10-percentage-point higher monthly oil-price change affect next-month monthly CPI inflation, accounting for prior global economic activity? How do the model-conditional estimates vary across these countries?

| Role | Common definition |
|---|---|
| Treatment T | Brent monthly percentage change in t−1, divided by 10 |
| Outcome Y | 100 × (unadjusted CPI_t / CPI_(t−1) − 1), not annualized |
| Potential confounder C | IGREA in t−2, divided by 100 |

Candidate DAG: C→T, C→Y, T→Y. Timing and one imperfect demand proxy do not establish sufficient adjustment. Observed oil movements combine demand and supply events. The USD oil treatment is not a local-currency energy-cost measure.

## Consistent four-country measurement

Use national monthly all-items CPI indices from OECD, unadjusted for **every country**, including the US. US/China: `DSD_PRICES@DF_PRICES_ALL`; Japan/Canada: `DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL`. No within-country old/new series splicing, interpolation or missing-month filling. Data begin in 1999 for transformations; the analysis uses 2000–2024.

The previous officially adjusted US CPI model is a retired baseline. Its numerical results must not be quoted as results of the new model. CPI baskets, coverage and classification differ across countries even with common index reference years.

## Statistical model

Estimate independently for each country i:

    Y_it = alpha_i + beta_i T_t + gamma_i C_t + s_i,month(t) + u_it
    u_it = phi_i u_i,t−1 + epsilon_it
    epsilon_it ~ Normal(0, sigma_i²)
    u_i0 ~ Normal(0, sigma_i²/(1−phi_i²)), |phi_i| < 1

Twelve calendar-month effects sum to zero and are represented by 11 orthonormal Helmert contrasts. They model recurring seasonality within the likelihood, not an official seasonal adjustment. Their estimates are country-specific. The unadjusted model removes C only, retaining the same calendar structure and priors.

Independent working priors: alpha~Normal(0,0.5²), beta~Normal(0,0.3²), gamma~Normal(0,0.5²), each seasonal contrast~Normal(0,0.5²), sigma~HalfNormal(0.5), phi~Uniform(−1,1). These are fixed unit scales, shared across countries; external justification is provisional. Prior prediction and beta-prior sensitivity are included.

The sampler uses exact Gaussian coefficient Gibbs draws and Metropolis updates for log(sigma)/atanh(phi), including Jacobians. Four chains per fit, 2,000 warmup + 6,000 retained iterations each; adaptation stops after warmup. ArviZ rank R-hat, bulk/tail ESS, MCSE and traces are saved. The likelihood is independently checked against a multivariate Normal covariance calculation.

## Current full-data associations

| Country | Adjusted beta | 95% equal-tailed credible interval |
|---|---:|---|
| United States | 0.09475 | [0.05955, 0.13088] |
| China | 0.08371 | [0.03582, 0.13208] |
| Japan | 0.03475 | [0.00413, 0.06550] |
| Canada | 0.11798 | [0.08044, 0.15514] |

Units are monthly inflation percentage points per 10-percentage-point higher monthly oil change. These are model-conditional associations. The intervals do not account for omitted confounding or every model failure and are not a test of country differences. Sampling passes for all full-data adjusted and unadjusted parameters; model adequacy is assessed separately. Full current tables are in `results/`.

The 12-lag innovation diagnostic gives p≈0.0529 (US), 0.0167 (China), 0.0072 (Japan), and 0.1660 (Canada), so dependence remains particularly concerning for China and Japan. Predictive replications miss the extreme US negative value and Chinese/Japanese positive peaks. Do not interpret good chain convergence as a clean bill of health for the likelihood. The revised synthetic validation recovers beta≈0.15034 for a true 0.15000.

## Analysis and validation

1. Verify provenance, unique keys, complete calendars, national CPI units, lags and transformations for all four countries.
2. Show distributions and country inflation histories; inspect prior predictions.
3. Estimate seasonal adjusted/unadjusted Bayesian AR models separately for each country.
4. Recover known parameters in a separate simulated system with seasonality, demand confounding and AR errors. Keep simulated measurements out of real-data analysis.
5. Check sampling, posterior predictive intervals, extremes, innovation dependence, subperiods and large oil movements. Report failures rather than suppressing them.
6. Train on 2000–2019 and assess retrospective one-step predictions over 2020–2024. Seasonal effects use training data only. Compare persistence, training calendar-month means and the same month last year. Prior outcome history enters the AR stochastic specification. These benchmarks do not isolate oil's incremental forecast contribution; a matched no-oil AR forecast model is not fitted.
7. Present country estimates with uncertainty without claiming a causal ranking or formal between-country difference. No joint country contrast or correlated-country error model is implemented.

## Limits and next steps

Calendar-month effects do not fully address Lunar New Year, changing seasonality, consumption taxes or administered prices. Gaussian innovations can miss extremes; Student-t innovations remain a planned extension. Good sampling is not evidence that the likelihood or DAG is correct. The 1,200 rows share 300 global oil/demand histories, so they are not independent exposures.

One global-activity proxy may fail to block relevant demand, expectations, supply-chain, exchange-rate and policy paths. Japan and China are large economies; treating global oil/demand as external to domestic conditions is substantive. The empirical coefficients are conditional associations; causal language requires the stated strong assumptions.

For the course, the three substantive constructs are preserved in separate country fits. Country-specific parameters, calendar effects and AR dependence add statistical complexity, and their temporal interpretation must be justified under the instructor's three-variable scope. Do not call the seasonal terms extra proven confounder adjustment. The analysis is exploratory; the holdout period has already informed earlier model development.

## Submission workflow

Finalize a one-page proposal with the four-country scope, DAG, estimand, likelihood, priors and independent mechanism reference. Check class dataset uniqueness. Prepare a notebook plus code-inclusive PDF of at most 20 pages and a group screencast; these submission tasks are not completed by this repository. The notebook builder includes `analysis/model.py` in the executable notebook so exported code is visible.

Mechanism reference: [Federal Reserve, Oil Price Shocks and Inflation in a DSGE Model of the Global Economy (2024)](https://www.federalreserve.gov/econres/notes/feds-notes/oil-price-shocks-and-inflation-in-a-dsge-model-of-the-global-economy-20240802.html). It explains consumer-energy and production-cost transmission; its model-based supply-shock effect is not a direct calibration of our one-month observational nominal-price estimand.
