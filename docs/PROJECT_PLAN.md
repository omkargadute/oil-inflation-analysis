# Working project plan

## Scope

Proceed with a focused US oil–inflation modeling study, 2000–2024. The earlier four-country inflation/interest-rate plan and superseded probes are retired. The project is an initial, assumption-dependent causal modeling exercise, not a claim of established macroeconomic causation.

**Question:** Under the proposed causal assumptions, how does a 10-percentage-point higher monthly oil-price change affect next-month US monthly CPI inflation, accounting for prior global economic activity?

| Role | Definition |
|---|---|
| Treatment T | Brent monthly percentage change in month t−1, divided by 10 |
| Outcome Y | Seasonally adjusted monthly CPI percentage change in month t |
| Potential confounder C | IGREA in month t−2, divided by 100 |

Candidate DAG: C→T, C→Y, T→Y. These are hypothesized causal relationships. IGREA is an imperfect demand proxy; timing does not establish sufficient adjustment or exogeneity. AR residual structure represents time dependence, not an extra measured confounder. Its temporal causal interpretation needs explanation.

## Completed baseline

- Verified 300 complete aligned months without interpolation.
- Compared monthly and year-over-year inflation; monthly avoids overlapping 12-month growth windows and reduces residual persistence.
- Estimated adjusted and unadjusted Bayesian Gaussian AR(1) models with explicit independent priors.
- Performed prior prediction, synthetic parameter recovery, four-chain diagnostics, and posterior predictive checks.
- Checked prior sensitivity, subperiods, extreme oil observations, higher AR orders, and retrospective one-step predictions.
- Cross-checked the AR likelihood against a multivariate Normal calculation and diagnostics against ArviZ.

Adjusted posterior oil coefficient: 0.0945, with 95% equal-tailed credible interval [0.0602, 0.1284], in monthly inflation percentage points per treatment unit. Unadjusted estimate: 0.0936. In simulation the estimated coefficient is 0.1498 for a true 0.1500. These do not certify real-world causal adjustment.

## Remaining model work

The Gaussian AR(1) model leaves dependence (12-lag innovation diagnostic p≈0.0025) and misses extreme negative inflation. Test a Student-t innovation specification with a modest dependence structure. Repeat simulation recovery, sampler diagnostics, and posterior prediction for any replacement model. Do not describe this extension as completed or discard observations to manufacture a favorable result.

The retrospective test RMSE is 0.3221 for the Bayesian AR(1) model versus 0.3158 for last-month inflation. Forecasting is secondary to the causal modeling exercise; revised data and assumed release availability limit the interpretation of this test.

Demand adjustment making little difference is a valid result. Simulation can show bias when a genuine confounder is omitted; the actual data cannot establish that this proxy removes all confounding. Explain omitted global-demand, expectations, supply-chain, exchange-rate, and policy paths. Observed oil-price changes mix supply and demand influences.

## Deliverable workflow

1. Specify the estimand and candidate three-variable DAG, including limitations.
2. Justify likelihood and priors using external information and prior predictive simulation.
3. Validate known-parameter recovery on separate synthetic data.
4. Estimate adjusted/unadjusted models on real observations.
5. Present posterior uncertainty, sampling diagnostics, predictive checks, and sensitivity findings.
6. Discuss extensions, including interest rates and international comparisons, without expanding the current causal analysis.

The course also requires a one-page proposal, dataset uniqueness registration, a code-inclusive notebook PDF of at most 20 pages, and a group screencast. These submission tasks are not completed by this research repository. The course handout stays local and is excluded from Git.

## Mechanism reference

[Federal Reserve, Oil Price Shocks and Inflation in a DSGE Model of the Global Economy (2024)](https://www.federalreserve.gov/econres/notes/feds-notes/oil-price-shocks-and-inflation-in-a-dsge-model-of-the-global-economy-20240802.html) explains consumer-energy and production-cost transmission. Its identified/model-based supply-shock estimand differs from this observational one-month price-change estimand; its effect size is not a direct calibration for our slope prior.
