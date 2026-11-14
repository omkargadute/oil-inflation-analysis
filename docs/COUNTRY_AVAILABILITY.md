# Country expansion: verified data availability

## Finding

**China and Japan can both be added for 2000–2024.** This audit downloaded monthly national all-items CPI indices for 17 countries from two official OECD dataflows. Sixteen have 300 complete monthly inflation observations after alignment with lagged Brent and IGREA predictors. This establishes coverage, not fully harmonized definitions or causal comparability.

The earlier Japanese gap was specific to the old COICOP 1999 series ending June 2021. The correct replacement is **DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL**, which has complete unadjusted Japanese CPI history. No old/new splicing or imputation is required. Earlier failed probes used a different DSD identifier.

## Actual coverage, January 2000–December 2024

Counts are computed month-on-month inflation observations, not just index levels. Queries include 1999 to construct growth rates and lags. Full-history candidates also have 300 year-over-year observations.

| Country | Unadjusted monthly inflation observations | Full-period official adjusted series in queried flows? |
|---|---:|---|
| United States | 300 | Yes |
| China | 300 | No |
| Japan | 300 | No; replacement adjusted index starts January 2010 |
| Canada | 300 | Yes |
| Germany | 300 | Yes |
| France | 300 | Yes |
| United Kingdom | 300 | No; verify CPI versus CPIH definition |
| South Korea | 300 | No |
| Italy | 300 | No |
| Spain | 300 | No |
| Brazil | 300 | No |
| Mexico | 300 | No; use replacement dataflow |
| South Africa | 300 | No |
| Norway | 300 | No |
| Switzerland | 300 | No |
| India | 300 | No; verify population coverage and historical definition |
| Australia | 8 | No; these queries return only nine index levels |

“No” means no complete adjusted series in these queries, not that no such national data exist. Australia's other monthly/quarterly products were not audited here. All available national definitions and breaks still need review before final comparisons.

Details: `results/country_series_coverage.csv`; summary: `results/country_coverage.csv`. The replacement Japanese series has 180 adjusted index levels, January 2010–December 2024, yielding **179 monthly inflation rates**. The old adjusted series has 258 months through June 2021. Neither alone covers 2000–2024; do not silently splice them.

## Recommendation

Start the intended comparison with **US, China, Japan and Canada**: four countries and 1,200 potential country-month observations. Germany and France are optional additions. The same Brent and global-activity series are shared across countries; repeating them does not create independent global shocks.

The simplest direct extension of the existing **officially adjusted monthly specification** is US + Canada + Germany + France. To include China and Japan, choose a consistent seasonality strategy first:

1. **Preserve monthly inflation:** apply a documented seasonal-adjustment procedure consistently to national unadjusted indices, or model seasonal structure explicitly. Validate it, and avoid full-sample adjustment leaking future information into forecast tests. Moving holidays such as Lunar New Year may need special treatment. Explain seasonal nuisance parameters under the course's simple-model scope.
2. **Use year-over-year inflation for the international comparison:** all 16 complete candidates support 300 observations. This reduces ordinary seasonal influence but does not remove every calendar effect, and reintroduces overlapping windows and persistence. It needs an appropriate error model and answers a different question from the monthly baseline.

The current US notebook remains the tested baseline. This audit does not fit new-country models or produce an adjusted international panel. Do not choose the outcome or adjustment method based on which yields significant oil coefficients.

## Causal and measurement limits

Use separate country models with the same three substantive constructs: oil change, inflation and prior global activity. This preserves the conceptual roles, though the intended assignment scope still matters. Pooled country indicators/interactions change the statistical design and need an explicit explanation under the three-variable restriction.

National CPI baskets, energy subsidies, taxes, exchange rates and price regulation can explain coefficient differences. USD Brent is not a local-currency energy-cost measure. Japan and China are large economies; treating global oil/demand as external to domestic conditions is an assumption. Shared global predictors create cross-country dependence; separate intervals alone do not establish a difference between country effects. A 2015 index reference level does not make national indices fully harmonized.

The previously selected UK national series matched CPIH at the October 2022 spot check; verify the intended index before comparing with CPI-based targets. India's long series needs a population-definition crosswalk before assuming it represents the same broad-population concept as every other CPI.

## Reproduce

```sh
python scripts/check_country_coverage.py
python scripts/check_country_coverage.py --refresh
```

The first command verifies saved snapshots and rebuilds the audit. The second explicitly downloads revisions. Exact API URLs, retrieval times and hashes are in `data/raw/country_manifest.json`, separate from baseline US provenance.

The script checks hashes, unique country/flow/adjustment/month keys, positive index values, monthly frequency, index units, growth rates, missing months, and lag alignment. It does not interpolate or stitch flows. `data/processed/country_cpi_candidate_panel.csv` is explicitly **unadjusted** and preserves Australia's missing months. Full national methodological harmonization is not claimed.

## Official sources

- [OECD CPI FAQs](https://www.oecd.org/en/data/insights/data-explainers/2024/07/consumer-price-indices-frequently-asked-questions-faqs.html): index definitions and adjusted-series availability.
- [OECD COICOP 2018 Japan dataflow](https://data-explorer.oecd.org/vis?df%5Bag%5D=OECD.SDD.TPS&df%5Bid%5D=DSD_PRICES_COICOP2018%40DF_PRICES_C2018_ALL&df%5Bvs%5D=1.0&dq=JPN.M.N.CPI.IX._T.N._Z): replacement series; exact query in manifest.
- [FRED China CPI](https://fred.stlouisfed.org/series/CHNCPIALLMINMEI): independent OECD mirror downloaded through April 2025, covering the study window.
- [Statistics Bureau of Japan](https://www.stat.go.jp/data/cpi/1.htm): alternative official national source.
