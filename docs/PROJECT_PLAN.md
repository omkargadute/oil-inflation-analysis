# Four-country oil–inflation project plan

## Final scope

**United States, China, Japan and Canada; January 2000–October 2026.** One combined calendar supports EDA, main estimation and robustness: 1,288 country-month rows and 1,286 observed monthly inflation outcomes. See [COMBINED_NOTEBOOK.md](COMBINED_NOTEBOOK.md) and [current findings](../results/combined_project/FINDINGS.md).

Question: how does a 10-percentage-point higher monthly Brent change relate to next-month monthly CPI inflation, accounting for prior global activity? Country models are separate; shared global histories are not independent country shocks. Interest rates remain future work.

| Construct | Definition |
|---|---|
| T | Brent monthly percentage change in t−1, divided by 10 |
| Y | 100 × (unadjusted CPI_t / CPI_(t−1) − 1), monthly, not annualized |
| C | IGREA in t−2, divided by 100 |

The DAG is C→T, C→Y, T→Y. Consistency, overlap, adequate temporal structure and sufficient confounding control are substantive assumptions. Exchange rates, taxes, subsidies, expectations, supply constraints and policy feedback can violate identification. Results are conditional associations unless those assumptions hold.

## Measurement and model

All countries use national all-items unadjusted CPI and 11 orthonormal month contrasts. US/China use OECD COICOP 1999; Japan/Canada use COICOP 2018, with no within-country splicing. Brent is nominal USD exposure; IGREA is a shipping-based proxy, not GDP growth. Revised reference-month snapshots are not real-time vintages.

Gaussian and Student-t AR1 regressions retain/omit only C for the adjustment comparison. Stable AR2, multiplicative seasonal AR, and nu=3/5/10 provide sensitivities. Established monthly-unit priors are retained. Every family conditions on the same first 13 observed months, avoiding an unsupported stationary Student-t initial density.

Missing October 2025 US CPI implies two unknown monthly outcomes. They remain missing in the dataset and are integrated during fitting with exact Gaussian conditional updates under the scale mixture. Later observations are retained. Forecasts require observed targets and required observed lag histories. The known two-month index-level constraint is not additionally modeled.

## Validation

1. Verify hashes, definitions, keys, calendars, missingness and pre-treatment timing.
2. Describe the combined period and conditional prior predictions.
3. Train full-period Gaussian/Student-t models with/without C; assess every parameter and latent-state chain.
4. Recover known parameters in independent Gaussian/Student-t data; inspect 40 repeated synthetic datasets across confounding/persistence scenarios.
5. Evaluate posterior replications, extremes and calendar-aware residual dependence.
6. Compare chronological forecasts in 2010–2014, 2015–2019 and 2020–October 2026. Compare oil/no-oil and dependence variants on identical eligible months, using paired calendar-block uncertainty.
7. Evaluate 2025 onward using pre-2025 training. Show through-2024 coefficients only as historical references.
8. Report full-period tail, lag, period and prior sensitivity without selecting by oil significance. Single-lag estimates are not cumulative effects.

Good sampling does not establish model adequacy or causality. Missing-state integration is model dependent. Short recent windows, imperfect demand measurement, revised inputs and inspected comparisons limit predictive conclusions. October oil exposures require November inflation to observe the next-month response.

The three substantive constructs are preserved; seasonality/dynamics add statistical complexity that must be justified under the course scope. No formal causal country ranking is implemented.

## Deliverables

`notebooks/final_project.ipynb`, sourced from `analysis/final_project.py`, includes the current computational implementation. Current tables/figures live in `results/combined_project/`. Earlier outputs remain historical references. The course also requires a code-inclusive, 12-point PDF of at most 20 pages, dataset uniqueness registration and a group screencast; these remain separate submission steps.
