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

The completed report compares Gaussian errors with Student-t innovations at fixed nu=5. Student-t initial residuals are independently t_5 with scale sigma/sqrt(1−phi²): this matches stationary variance, not the exact t-AR stationary distribution. Sigma is a scale; its Student-t innovation SD is sigma*sqrt(5/3), so equal scale priors imply larger innovation-variance priors in that family. The Student-t sampler uses Normal/Gamma augmentation (shape/rate), exact Gaussian coefficient Gibbs draws, and transformed sigma/phi Metropolis updates including Jacobians. Four chains have 1,500 warmup + 4,000 retained iterations; adaptation stops after warmup. ArviZ R-hat, bulk/tail ESS, MCSE and traces are saved. Independent tests check the integrated mixture density and Gaussian joint AR covariance.

## Current full-data associations: Student-t, demand-adjusted

| Country | Adjusted beta | 95% equal-tailed credible interval |
|---|---:|---|
| United States | 0.09105 | [0.05837, 0.12670] |
| China | 0.09157 | [0.04639, 0.13633] |
| Japan | 0.03093 | [0.00298, 0.05895] |
| Canada | 0.12277 | [0.08442, 0.16079] |

Units are monthly inflation percentage points per 10-percentage-point higher monthly oil change. These are model-conditional associations. The intervals do not account for omitted confounding or every model failure and are not a test of country differences. Sampling passes for all 32 simulation, full-data and training fits; model adequacy is assessed separately. Current tables and generated findings are in `results/final_project/`; older `results/` tables remain exploratory references.

Student-t 12-lag innovation diagnostics give p≈0.0256 (US), 0.0140 (China), 0.0027 (Japan), and 0.1650 (Canada). Heavy tails improve US/Chinese extreme-value checks but Japan's maximum remains outside the predictive envelope, and dependence is not resolved. Synthetic adjusted beta is about 0.1502 under Gaussian and 0.1456 under Student-t data for a true 0.1500; both intervals include the truth. Omitting the known simulated confounder shifts estimates to about 0.20. One successful recovery per family is not a coverage study.

## Analysis and validation

1. Verify provenance, unique keys, complete calendars, national CPI units, lags and transformations for all four countries.
2. Show distributions and country inflation histories; inspect prior predictions.
3. Estimate seasonal adjusted/unadjusted Bayesian AR models separately for each country.
4. Recover known parameters in a separate simulated system with seasonality, demand confounding and AR errors. Keep simulated measurements out of real-data analysis.
5. Check sampling, posterior predictive intervals, extremes, innovation dependence, subperiods and large oil movements. Report failures rather than suppressing them.
6. Train on 2000–2019 and assess retrospective one-step predictions over 2020–2024. Seasonal effects use training data only. Compare simple benchmarks and a matched Student-t AR model without oil, retaining C and the same seasonal structure and priors on retained parameters. Adding oil reduces RMSE about 4.79% in China, barely changes Japan (0.25% lower RMSE but worse log score), and raises RMSE about 4.17% in the US and 2.53% in Canada. These are exploratory single-period comparisons, not significance tests or real-time forecasts.
7. Present country estimates with uncertainty without claiming a causal ranking or formal between-country difference. No joint country contrast or correlated-country error model is implemented.

## Limits and next steps

Calendar-month effects do not fully address Lunar New Year, changing seasonality, consumption taxes or administered prices. Student-t innovations are implemented, but fixed nu=5, residual dependence and the initial-distribution assumption remain limitations. Good sampling is not evidence that the likelihood or DAG is correct. The 1,200 rows share 300 global oil/demand histories, so they are not independent exposures.

One global-activity proxy may fail to block relevant demand, expectations, supply-chain, exchange-rate and policy paths. Japan and China are large economies; treating global oil/demand as external to domestic conditions is substantive. The empirical coefficients are conditional associations; causal language requires the stated strong assumptions.

For the course, the three substantive constructs are preserved in separate country fits. Country-specific parameters, calendar effects and AR dependence add statistical complexity, and their temporal interpretation must be justified under the instructor's three-variable scope. Do not call the seasonal terms extra proven confounder adjustment. The analysis is exploratory; the holdout period has already informed earlier model development.

## Submission workflow

The computational notebook is complete, and its appendix includes data preparation, Gaussian sampling and Student-t/cache/evaluation source. Submission PDF pagination (at most 20 pages in 12-point font), class dataset uniqueness registration, and the group screencast remain separate deliverables. The proposal files remain available locally. The report documents external prior-scale context and uncertainty rather than claiming that prior simulation alone establishes an objectively correct prior.

Mechanism reference: [Federal Reserve, Oil Price Shocks and Inflation in a DSGE Model of the Global Economy (2024)](https://www.federalreserve.gov/econres/notes/feds-notes/oil-price-shocks-and-inflation-in-a-dsge-model-of-the-global-economy-20240802.html). It explains consumer-energy and production-cost transmission; its model-based supply-shock effect is not a direct calibration of our one-month observational nominal-price estimand.
