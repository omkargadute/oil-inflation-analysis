# Strengthening experiments: results and interpretation

All three dependence models use the same 13-month conditioning prefix and target observations. These conditional fits are sensitivity analyses, not interchangeable with the earlier 300-observation stationary-initial AR1 posterior.

Completed 188 registered cases (184 unique retained fits); all retained sampling checks passed. Forty independent simulated datasets each have adjusted and omitted-confounder fits.

## Dependence and rolling prediction

- **United States:** highest average rolling log score: ar1; RMSE 0.2416 (matched AR1 0.2416); 24-lag innovation p=0.0385. Oil RMSE gain 3.20%, block interval [-4.86%, 14.95%].
  Structure RMSE gain versus oil-containing AR1: 0.00% [0.00%, 0.00%].
- **China:** highest average rolling log score: ar1; RMSE 0.3936 (matched AR1 0.3936); 24-lag innovation p=0.0117. Oil RMSE gain 0.51%, block interval [-3.00%, 4.34%].
  Structure RMSE gain versus oil-containing AR1: 0.00% [0.00%, 0.00%].
- **Japan:** highest average rolling log score: sar12; RMSE 0.3116 (matched AR1 0.3211); 24-lag innovation p=0.1223. Oil RMSE gain -0.85%, block interval [-2.83%, 1.85%].
  Structure RMSE gain versus oil-containing AR1: 2.95% [1.46%, 5.34%].
- **Canada:** highest average rolling log score: ar2; RMSE 0.3222 (matched AR1 0.3222); 24-lag innovation p=0.1502. Oil RMSE gain 4.09%, block interval [-2.44%, 11.29%].
  Structure RMSE gain versus oil-containing AR1: 0.02% [-2.91%, 3.34%].

Model rankings use the inspected validation periods, not an untouched confirmatory test. A higher score or a larger diagnostic p-value is not proof of causality or model adequacy. Block intervals use circular 12-month blocks, 2,000 replicates, and separate resampling within each evaluation window. Matching time indices are used across countries, preserving common-month dependence for comparisons.

## Tail, lag and period sensitivity

The suite fits nu=3,5,10 at lag one, oil lags two/three with correspondingly earlier demand proxies, and separate pre/post-2020 Bayesian fits. Lag changes alter the timing estimand. Post-2020 fits retain 13 pre-period months as conditioning history, so all 60 evaluation-period outcomes enter their likelihood. Real outliers remain in the data; the event audit distinguishes fitted observations from the conditioning prefix.

Single-lag models are timing sensitivities, not a distributed-lag or cumulative causal response.

- **United States:** nu=3/10 beta range 0.0888 to 0.0933; lag-two beta -0.0117, lag-three beta -0.0156; pre-2020 beta 0.1314, post-2020 beta 0.0546. See coefficient_sensitivity.csv for every credible interval.
- **China:** nu=3/10 beta range 0.0996 to 0.1062; lag-two beta -0.0040, lag-three beta 0.0119; pre-2020 beta 0.1008, post-2020 beta 0.1144. See coefficient_sensitivity.csv for every credible interval.
- **Japan:** nu=3/10 beta range 0.0365 to 0.0380; lag-two beta 0.0185, lag-three beta -0.0086; pre-2020 beta 0.0514, post-2020 beta 0.0080. See coefficient_sensitivity.csv for every credible interval.
- **Canada:** nu=3/10 beta range 0.1135 to 0.1194; lag-two beta -0.0461, lag-three beta 0.0134; pre-2020 beta 0.1493, post-2020 beta 0.0637. See coefficient_sensitivity.csv for every credible interval.

## Repeated recovery

Forty independent simulated datasets cross weak/strong confounding and low/high persistence. Each is fitted with and without the confounder. Coverage is reported with Wilson Monte Carlo intervals; ten replications per scenario are too few to establish exact 95% calibration.

- Confounding 0.1, persistence 0.2: adjusted bias -0.0010, omitted-C bias 0.0063; adjusted estimation RMSE 0.0137 versus 0.0193; coverage 100% versus 100%.
- Confounding 0.1, persistence 0.7: adjusted bias -0.0066, omitted-C bias 0.0045; adjusted estimation RMSE 0.0143 versus 0.0158; coverage 90% versus 100%.
- Confounding 0.8, persistence 0.2: adjusted bias 0.0010, omitted-C bias 0.0825; adjusted estimation RMSE 0.0123 versus 0.0837; coverage 90% versus 0%.
- Confounding 0.8, persistence 0.7: adjusted bias -0.0030, omitted-C bias 0.0592; adjusted estimation RMSE 0.0155 versus 0.0612; coverage 90% versus 10%.

## Scope

All five requested experiment families are run. Compare residual checks, prediction and coefficient stability together. No primary lag, country ranking or tail shape is selected by statistical significance. Substantive causal identification and the three-variable course-scope interpretation remain assumptions.
