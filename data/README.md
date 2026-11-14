# Data dictionary

The international availability audit adds two OECD snapshots and `country_manifest.json`. Run `python scripts/check_country_coverage.py` to verify them and reproduce `country_cpi_candidate_panel.csv`. That candidate panel is unadjusted and is not a replacement for the seasonally adjusted US baseline. Coverage details and limitations are in [the country audit](../docs/COUNTRY_AVAILABILITY.md).

`raw/` contains the three source snapshots required to reproduce the current analysis. `raw/manifest.json` records source URLs, recorded download timestamps, and SHA-256 hashes. Verify with `python scripts/download_data.py --verify`.

`processed/us_monthly.csv` has one observation per outcome month, January 2000–December 2024 (300 rows). All modeled variables are complete. No values are imputed.

| Column | Meaning |
|---|---|
| date | Outcome month; dates identify months, not publication dates |
| oil | Monthly Brent nominal USD/barrel |
| cpi | Seasonally adjusted US CPI index level |
| demand | IGREA global commodity-market activity index |
| oil_pct | 100 × (oil_t / oil_(t−1) − 1) |
| monthly | 100 × (cpi_t / cpi_(t−1) − 1); not annualized |
| yearly | 100 × (cpi_t / cpi_(t−12) − 1); diagnostic comparison |
| T | oil_pct_(t−1) / 10; treatment |
| C | demand_(t−2) / 100; potential confounder |
| last_monthly | monthly_(t−1); diagnostic prediction benchmark |

The raw files extend beyond the analysis window; the analysis constructs lags before selecting 2000–2024. Source revisions and publication delays matter: this is an analysis of revised snapshots, not a real-time vintage backtest.
