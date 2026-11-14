# Four-country model results

```text
      country iso3      model    beta  lower95  upper95  sampling_pass
United States  USA unadjusted 0.09387  0.05911  0.13039           True
United States  USA   adjusted 0.09475  0.05955  0.13088           True
        China  CHN unadjusted 0.08719  0.03815  0.13600           True
        China  CHN   adjusted 0.08371  0.03582  0.13208           True
        Japan  JPN unadjusted 0.03533  0.00415  0.06673           True
        Japan  JPN   adjusted 0.03475  0.00413  0.06550           True
       Canada  CAN unadjusted 0.11917  0.08164  0.15637           True
       Canada  CAN   adjusted 0.11798  0.08044  0.15514           True
```

Unadjusted monthly CPI with country-specific calendar-month effects, separate Gaussian AR(1) models. Intervals are conditional associations, not identified causal effects or country-difference tests. Inspect residual, predictive, robustness and holdout tables for limitations.

## Model checks

All full-data, training and simulation sampling checks pass. This does not imply adequate residual or tail behavior.

- United States: 12-lag innovation p=0.0529; extreme-statistic misses: minimum.
- China: 12-lag innovation p=0.0167; extreme-statistic misses: maximum.
- Japan: 12-lag innovation p=0.0072; extreme-statistic misses: maximum.
- Canada: 12-lag innovation p=0.1660; extreme-statistic misses: none in these two checks.

## Retrospective holdout RMSE, 2020–2024

- United States: seasonal demand-adjusted AR1 0.3369; best simple benchmark 0.3765 (last-month inflation).
- China: seasonal demand-adjusted AR1 0.4216; best simple benchmark 0.4572 (training calendar-month mean).
- Japan: seasonal demand-adjusted AR1 0.3508; best simple benchmark 0.3418 (training mean).
- Canada: seasonal demand-adjusted AR1 0.3977; best simple benchmark 0.4011 (training calendar-month mean).

These are exploratory one-step predictions from revised data with training-only seasonal coefficients. They are not real-time vintage forecasts or a controlled test of oil information alone: the benchmarks differ in more than the oil predictor. No no-oil AR counterfactual forecast model or joint country-difference test is fitted. Gaussian model failures and imperfect demand adjustment remain. Student-t errors and moving-holiday treatment are future work.
