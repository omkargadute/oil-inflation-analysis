# Four-country data dictionary

`processed/multicountry_monthly.csv` is the main analysis panel: US, China, Japan and Canada, 300 complete months per country, 2000–2024. All four CPI series are **unadjusted**; recurring seasonality is modeled in each country's regression. No imputation or cross-flow splicing is used. The previous `us_monthly.csv` and officially adjusted US CPI snapshot are retired from the active repository.

| Column | Meaning |
|---|---|
| date | Outcome month; not the release date |
| country, iso3 | Country label and ISO code |
| cpi | National all-items monthly CPI index, unadjusted |
| adjustment | N: unadjusted |
| flow | Explicit OECD dataflow used for this complete country series |
| oil_usd | Monthly nominal Brent USD/barrel, shared globally |
| global_activity | IGREA industrial-commodity activity proxy, shared globally |
| oil_pct | 100 × (oil_t / oil_(t−1) − 1) |
| monthly | 100 × (cpi_t / cpi_(t−1) − 1), not annualized |
| yearly | 100 × (cpi_t / cpi_(t−12) − 1), descriptive comparison |
| T | oil_pct_(t−1) / 10 |
| C | global_activity_(t−2) / 100 |
| month | Calendar month; indexes the statistical seasonal terms |
| last_monthly | Previous month's inflation; forecast benchmark |
| last_year_same_month | Same month's inflation one year earlier; forecast benchmark |

US/China use the COICOP 1999 OECD flow, Japan/Canada the COICOP 2018 flow. National indices are not perfectly harmonized merely because they share an index reference year. Raw data include 1999 for transformations and lags. `last_year_same_month` is unavailable for January 2000 because January 1999 inflation would need December 1998; this optional benchmark is complete throughout the 2020–2024 evaluation period and is not a modeled-variable gap.

`raw/manifest.json` records Brent and IGREA sources/hashes. `raw/country_manifest.json` records the two OECD snapshots, with exact URLs, timestamps and hashes. Run `python scripts/download_data.py --verify` to verify all four source files. `--refresh` explicitly downloads revisions; rebuild the audit and analysis afterward.

`processed/country_cpi_candidate_panel.csv` and `results/country_coverage.csv` are a broader 17-country availability audit, not the main fitted panel. Run `python scripts/check_country_coverage.py` to reproduce them. See [the availability report](../docs/COUNTRY_AVAILABILITY.md) for detailed limitations.

All sources are revised snapshots; neither notebook nor audit reconstructs real-time publication vintages. Source data retain their providers' applicable terms.
