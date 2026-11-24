# Extension to current releases

Downloaded and verified November 24, 2026. The final notebook retains the original 2000–2024 analysis and adds a current-release section through **October 2026** for all four countries. The expanded calendar has 1,288 country-month rows and 1,286 observed monthly inflation outcomes. No historical values changed in the overlapping 2000–2024 core series in this retrieval.

| Input | Latest observation in downloaded snapshot | Model use |
|---|---|---|
| Monthly Brent | October 2026 | Monthly average; treatment is its previous-month percentage change |
| IGREA | September 2026 | Two-month lag, so it supports the October outcome |
| National CPI, all four countries | October 2026 | Unadjusted national all-items index; same country flows as the original study |
| Daily Brent | November 17, 2026 | Descriptive context only; not substituted for a complete monthly average |

Source URLs, retrieval timestamps and SHA-256 hashes are in [the current manifest](../data/raw/latest/manifest.json). The five current snapshots are separate from the original snapshots. Downloading current releases removes the old OECD query's December 2024 endpoint; merely refreshing that old URL would retain its cutoff.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts/download_latest.py
.\.venv\Scripts\python.exe -m analysis.latest_analysis
.\.venv\Scripts\python.exe scripts/build_notebook.py
```

The download command deliberately refreshes current snapshots. Subsequent analysis and notebook execution use the saved inputs without network requests. The model cache is `.local-archive/latest-model-cache/`; its keys include model code, data, retained transitions, priors, sampler settings and Numba version. Results are in [results/latest](../results/latest/FINDINGS.md). After a future refresh, inspect the availability and missing-observation audits before interpreting new results.

## US CPI gap

The OECD source omits October 2025 US CPI. [BLS confirms the shutdown prevented collection and publication](https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm). Calendar reindexing consequently leaves **both October and November 2025 monthly growth undefined**. December growth is available, but its previous monthly residual is not; forecasts and AR1 likelihood factors therefore omit October, November and December. January–October 2026 can all be evaluated. The recent US forecast has 19 scored months rather than 22.

No values are interpolated or carried forward. For the extended AR1 likelihood, define $\mathcal I=\{t:t\geq13,\ Y_t\text{ and }Y_{t-1}\text{ observed}\}$. We use

$$L(b,\phi,\sigma)=\prod_{t\in\mathcal I}\frac{1}{\sigma}t_5\!\left(\frac{(Y_t-X_tb)-\phi(Y_{t-1}-X_{t-1}b)}{\sigma}\right).$$

This defines a **segmented conditional model**: each observed run conditions on its first residual, and the original first 13 months are also conditioned out. Runs restart independently conditional on those residuals and shared parameters. Dependence across the unobserved gap is discarded. This is an additional modeling assumption, not an exact marginal likelihood integrating missing states of a continuous uninterrupted AR process. Other countries have complete calendars and the usual matched conditional AR1 likelihood. A numerical test verifies the specialized sampler equals the original conditional AR1 sampler on complete data.

## Findings and recommendation

The extended oil coefficients remain positive: US 0.0895, China 0.0914, Japan 0.0344, Canada 0.1108. Units remain monthly inflation percentage points per 10-percentage-point higher previous-month oil change. Full credible intervals and matched through-2024 estimates are in [the generated findings](../results/latest/FINDINGS.md).

Keeping coefficients trained through December 2024 fixed, oil changes RMSE over January 2025–October 2026 by +12.24% gain in the US, -5.40% in China, +0.54% in Japan and +0.60% in Canada. Every paired block interval includes zero. Three- and six-month block choices show uncertainty in the short evaluation period; calendar blocks preserve US missing-month positions. These intervals condition on fitted forecasts and do not incorporate every source of model or parameter uncertainty. The ten-month 2026 subset is also reported separately, with especially limited precision.

Use the latest data as a temporal-stability and recent-event extension. Retain the original one-month estimand and the common Student-t AR1 reference. The earlier strengthening suite provides the strongest evidence for a structural improvement in Japan: seasonal AR lowers 2010–2024 rolling RMSE by 2.95%, with a paired block interval of 1.46%–5.34%, although some residual diagnostics still fail. Canada's AR2 gain is only 0.02%, with an interval spanning zero. Keep those as transparent sensitivity results rather than selecting whichever specification produces the largest oil coefficient.

October's $123.93 monthly Brent average and November's price reversal provide useful context. Under the lag-one question, October oil predicts November inflation, which is not yet in the snapshot; November's complete average is also unavailable. We cannot claim the newest event's full inflation response is already measured. Price movements mix supply, demand and policy influences; this observational extension does not isolate a geopolitical shock's causal effect.
