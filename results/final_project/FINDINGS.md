# Completed four-country computational analysis

All source hashes and 1,200 country-month rows verified. All simulation, full-data and training sampling checks passed.

## Oil–inflation associations (Student-t, demand-adjusted)

- **United States:** beta 0.0910, 95% credible interval [0.0584, 0.1267]; omitting demand gives 0.0873.
- **China:** beta 0.0916, 95% credible interval [0.0464, 0.1363]; omitting demand gives 0.0931.
- **Japan:** beta 0.0309, 95% credible interval [0.0030, 0.0590]; omitting demand gives 0.0315.
- **Canada:** beta 0.1228, 95% credible interval [0.0844, 0.1608]; omitting demand gives 0.1243.

Units are monthly inflation percentage points per 10-percentage-point higher monthly oil change. These are model-conditional associations.

## Predictive checks and residual dependence

- **United States:** Student-t 12-lag innovation p=0.0256; predictive-summary misses: lag1.
- **China:** Student-t 12-lag innovation p=0.0140; predictive-summary misses: none among the four checked summaries.
- **Japan:** Student-t 12-lag innovation p=0.0027; predictive-summary misses: maximum.
- **Canada:** Student-t 12-lag innovation p=0.1650; predictive-summary misses: none among the four checked summaries.

## Does oil improve the matched retrospective forecast?

- **United States:** adding oil increases RMSE by 4.17% (0.3175 without oil versus 0.3308 with oil); mean log-score gain -0.0677.
- **China:** adding oil reduces RMSE by 4.79% (0.4461 without oil versus 0.4247 with oil); mean log-score gain 0.0662.
- **Japan:** adding oil reduces RMSE by 0.25% (0.3533 without oil versus 0.3524 with oil); mean log-score gain -0.0044.
- **Canada:** adding oil increases RMSE by 2.53% (0.3886 without oil versus 0.3984 with oil); mean log-score gain -0.0636.

## Conclusions and future work

Oil coefficients, confounder adjustment and forecast gains answer different questions. The simulation shows what omitted-confounder bias can look like under a known DAG; real-data adjustment cannot certify that IGREA blocks every relevant path. Heavy-tailed innovations may improve extremes while leaving serial or calendar dependence. We report both families rather than selecting by significance.

Interpretation depends on consistency, overlap, adequate temporal structure and sufficient confounding control. Global-demand measurement error, exchange rates, subsidies, taxes, moving holidays and shared shocks remain limitations. Student-t nu=5 and the variance-matched initial distribution are additional assumptions; sigma is a scale, not the Student-t innovation SD. Prior reweighting and HAC checks do not cover every form of model uncertainty.

The strengthening experiments above test alternative tails, dependence, lags and periods; their generated findings qualify the baseline results. Moving holidays, local-currency oil exposure and better-identified supply shocks remain future work. Interest rates and pooled country-effect comparisons require a separate expanded design. No causal ranking across countries or real-time forecast claim follows from this analysis.

## Completed strengthening experiments

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


## Current-release extension

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
