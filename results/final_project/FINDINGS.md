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

Future work should examine moving holidays and regime changes, alternative tail/dependence assumptions, local-currency oil exposure and better-identified supply shocks. Interest rates and pooled country-effect comparisons require a separate expanded design. No causal ranking across countries or real-time forecast claim follows from this analysis.
