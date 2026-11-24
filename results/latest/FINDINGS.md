# Current-release extension

Common monthly endpoint: **October 2026**. Four country calendars contain 1,288 rows and 1,286 observed monthly inflation outcomes. Daily Brent ends November 17, 2026; the last observation is $92.94/barrel. The complete October 2026 monthly average is $123.93/barrel. Daily prices are context only and are never substituted for monthly averages.

## Missing US month and likelihood

US October 2025 CPI is absent. Calendar reindexing leaves October and November monthly inflation undefined. No interpolation, forward fill or two-month-as-one-month calculation is used. The conditional AR1 model includes only observed consecutive residual pairs after the original 13-month prefix. The first observed residual after each gap is conditioned on, restarting the likelihood; all ten 2026 US outcomes can still contribute. This is a segmented conditional likelihood, not an exact stationary initial-state or missing-state marginal likelihood.

BLS confirms the missing October release followed the shutdown: [official explanation](https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm).

## Matched current-vintage coefficient comparison

- **United States:** through-2024 beta 0.0913 [0.0571, 0.1281]; extended beta 0.0895 [0.0582, 0.1229]; 306 observed AR1 transitions in the extended likelihood.
- **China:** through-2024 beta 0.1030 [0.0573, 0.1489]; extended beta 0.0914 [0.0503, 0.1329]; 309 observed AR1 transitions in the extended likelihood.
- **Japan:** through-2024 beta 0.0371 [0.0073, 0.0670]; extended beta 0.0344 [0.0056, 0.0641]; 309 observed AR1 transitions in the extended likelihood.
- **Canada:** through-2024 beta 0.1160 [0.0779, 0.1538]; extended beta 0.1108 [0.0735, 0.1475]; 309 observed AR1 transitions in the extended likelihood.

## Fixed-training recent predictions

Coefficients use only outcomes through December 2024. Predictions condition on the previous observed monthly residual. They use revised data, not real-time vintages; short recent windows cannot establish a stable crisis regime or causal supply-shock effect.

- **United States:** 19 scored months; oil RMSE gain 12.24%; 3-month block interval [-6.66%, 21.21%], 6-month block interval [-7.86%, 18.43%].
- **China:** 22 scored months; oil RMSE gain -5.40%; 3-month block interval [-15.33%, 4.57%], 6-month block interval [-12.09%, 4.67%].
- **Japan:** 22 scored months; oil RMSE gain 0.54%; 3-month block interval [-6.86%, 5.92%], 6-month block interval [-4.01%, 3.99%].
- **Canada:** 22 scored months; oil RMSE gain 0.60%; 3-month block interval [-19.47%, 13.98%], 6-month block interval [-13.78%, 14.57%].

## Interpretation

The nominal-price model combines demand, supply, exchange-rate and policy influences. Observed price surges or reversals cannot independently identify a geopolitical causal effect. The newest monthly oil movement can affect next-month inflation under our lag-one estimand, so its full response may not yet be observed. The sample contains only ten 2026 months. Keep the original study and strengthening results as reference; use this extension to assess temporal stability and recent forecast behavior. Historical revisions are audited separately.
