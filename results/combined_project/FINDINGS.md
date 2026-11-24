# Combined-period findings: January 2000–October 2026

Verified 1,288 calendar rows and 1,286 observed monthly inflation outcomes. Two US outcomes are integrated as latent states; raw gaps remain unchanged. All retained parameter and latent-state sampling checks pass.

## Main Student-t AR1 associations

- **United States:** beta 0.0902 [0.0596, 0.1251]; omitting demand gives 0.0873.
- **China:** beta 0.0915 [0.0511, 0.1337]; omitting demand gives 0.0937.
- **Japan:** beta 0.0346 [0.0056, 0.0637]; omitting demand gives 0.0351.
- **Canada:** beta 0.1101 [0.0737, 0.1463]; omitting demand gives 0.1114.

Units: monthly inflation percentage points per 10-percentage-point higher previous-month oil change. These are model-conditional associations, not identified geopolitical supply-shock effects.

## Matched chronological evaluation, 2010–October 2026

- **United States:** best inspected predictive score: ar1; RMSE 0.2402, AR1 0.2402. Structure gain 0.00% [0.00%, 0.00%]. AR1 oil gain 3.81% [-4.74%, 15.45%], n=197 matched months.
- **China:** best inspected predictive score: ar2; RMSE 0.3922, AR1 0.3937. Structure gain 0.37% [-0.14%, 0.88%]. AR1 oil gain -0.02% [-3.31%, 3.43%], n=202 matched months.
- **Japan:** best inspected predictive score: sar12; RMSE 0.3114, AR1 0.3245. Structure gain 4.04% [2.14%, 6.77%]. AR1 oil gain -1.43% [-3.16%, 0.82%], n=202 matched months.
- **Canada:** best inspected predictive score: ar2; RMSE 0.3187, AR1 0.3181. Structure gain -0.18% [-3.02%, 2.87%]. AR1 oil gain 3.19% [-2.57%, 9.04%], n=202 matched months.

## Recent-event forecasts: pre-2025 training only

- **United States:** oil RMSE gain 12.28%, 19 scored months; 3-month block interval [-6.03%, 21.01%], 6-month block interval [-6.89%, 18.26%]. The short window gives limited precision.
- **China:** oil RMSE gain -5.09%, 22 scored months; 3-month block interval [-15.19%, 4.59%], 6-month block interval [-11.65%, 4.74%]. The short window gives limited precision.
- **Japan:** oil RMSE gain 0.50%, 22 scored months; 3-month block interval [-7.12%, 5.94%], 6-month block interval [-4.30%, 4.01%]. The short window gives limited precision.
- **Canada:** oil RMSE gain 0.45%, 22 scored months; 3-month block interval [-19.70%, 13.79%], 6-month block interval [-14.00%, 14.48%]. The short window gives limited precision.

## What the checks establish

Repeated simulation shows demand adjustment reduces bias under the known simulated DAG, particularly with strong confounding. That does not establish sufficient real-world adjustment. Tail, lag, period and prior comparisons are reported without selecting coefficients by significance.

The common Student-t AR1 remains the reference; inspect seasonal/AR2 alternatives jointly by prediction and residual behavior. Shared country shocks, imperfect global-demand measurement, exchange rates, taxes, moving holidays and policy changes remain limitations. No formal causal country ranking follows.

All model families now condition on the same initial 13 months. Missing monthly outcomes are integrated, rather than assigned observed values; their known two-month index-level relation is not added to the likelihood. Residual diagnostics use available calendar pairs and a reported contiguous run, not a compressed gap.

Forecast coefficients use only their declared training outcomes. They condition on observed lagged residuals and revised source snapshots; this is not a real-time forecasting exercise. Model comparisons use inspected evaluation periods. October oil exposures require November inflation to assess their next-month response, so the newest event is not fully observed.

The three substantive variables are oil, inflation and prior activity. Seasonality/dynamics add statistical structure that must be justified under the course scope. Interest rates and identified supply shocks require an expanded design. Submission PDF pagination and the group screencast are separate deliverables.
