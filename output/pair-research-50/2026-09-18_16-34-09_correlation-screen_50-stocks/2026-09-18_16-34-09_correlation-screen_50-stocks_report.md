# correlation-screen — 50 stocks — 2026-09-18

Run started: 2026-09-18T16:34:09.174749-04:00 (America/New_York).

Training: 2023-09-19 to 2025-10-22; later diagnostic period: 2025-10-23 to 2026-09-17.

50 requested stocks; 50 with complete positive training prices; 1,225 pair correlations; 14 with training correlation >= 0.80.

Correlation is Pearson correlation of daily log returns from adjusted closing prices, not correlation of price levels. 1 means perfectly matching linear daily-return moves; it does not measure expected profit. No missing prices were filled.

## Method and limits

Same GICS sub-industry, different CIK, training daily log-return correlation >= 0.8, last 126 training returns correlation >= 0.75, training raw Engle-Granger p < 0.05, positive training beta.

The later period is a diagnostic only and is not used for ranking. It has already been examined in this project, so it is not an untouched final test. Any refinements need a new chronological validation/test plan or future paper-trading data.

Engle-Granger is run on price levels, with an intercept, AIC lag selection and alphabetical ticker ordering (first ticker is dependent). Later-period cointegration is separately refitted; it is not a fixed-hedge profitability test. P-values are raw/unadjusted, following a data-based correlation screen. Multiple testing, changing membership and the I(1) assumption still need assessment; passing this screen is not proof of a valid strategy.

This uses the project's fetched present-day constituent snapshot, not historical membership. Eligibility uses training prices only; later missing rows can reduce diagnostic observations.

Sources: [Yahoo Finance](https://finance.yahoo.com/) via [yfinance](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html); [constituents](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies); [Engle-Granger documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html).

## 20 strongest training correlations across all industries

| Pair | Companies | Industry | Train r | Later r | Train coint p | Later coint p | Beta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| CPT/MAA | Camden Property Trust / Mid-America Apartment Communities | Multi-Family Residential REITs | 0.910 | 0.926 | 0.4248 | 0.3116 | 0.761 |
| UDR/VMRK | UDR, Inc. / Vivmark Residential | Multi-Family Residential REITs | 0.907 | 0.823 | 0.08341 | 0.6388 | 0.604 |
| ESS/VMRK | Essex Property Trust / Vivmark Residential | Multi-Family Residential REITs | 0.905 | 0.835 | 0.03932 | 0.2554 | 4.866 |
| CPT/UDR | Camden Property Trust / UDR, Inc. | Multi-Family Residential REITs | 0.897 | 0.868 | 0.07625 | 0.1535 | 3.010 |
| ESS/UDR | Essex Property Trust / UDR, Inc. | Multi-Family Residential REITs | 0.890 | 0.851 | 0.1102 | 0.9578 | 7.883 |
| MAA/UDR | Mid-America Apartment Communities / UDR, Inc. | Multi-Family Residential REITs | 0.872 | 0.861 | 0.2852 | 0.2697 | 3.744 |
| FRT/KIM | Federal Realty Investment Trust / Kimco Realty | Retail REITs | 0.867 | 0.733 | 0.6912 | 0.2025 | 2.818 |
| FRT/REG | Federal Realty Investment Trust / Regency Centers | Retail REITs | 0.863 | 0.802 | 0.6602 | 0.715 | 0.664 |
| CPT/VMRK | Camden Property Trust / Vivmark Residential | Multi-Family Residential REITs | 0.841 | 0.823 | 0.007045 | 0.6391 | 1.853 |
| CPT/ESS | Camden Property Trust / Essex Property Trust | Multi-Family Residential REITs | 0.831 | 0.828 | 0.0001604 | 0.7188 | 0.370 |
| KIM/REG | Kimco Realty / Regency Centers | Retail REITs | 0.827 | 0.786 | 0.4611 | 0.3754 | 0.299 |
| MAA/VMRK | Mid-America Apartment Communities / Vivmark Residential | Multi-Family Residential REITs | 0.818 | 0.832 | 0.3362 | 0.6451 | 2.247 |
| ESS/MAA | Essex Property Trust / Mid-America Apartment Communities | Multi-Family Residential REITs | 0.815 | 0.807 | 0.08368 | 0.9673 | 1.951 |
| A/WAT | Agilent Technologies / Waters Corporation | Life Sciences Tools & Services | 0.810 | 0.642 | 0.7853 | 0.02651 | 0.175 |
| HIG/TRV | Hartford (The) / Travelers Companies (The) | Property & Casualty Insurance | 0.796 | 0.740 | 0.3785 | 0.154 | 0.524 |
| CB/TRV | Chubb Limited / Travelers Companies (The) | Property & Casualty Insurance | 0.755 | 0.725 | 0.4971 | 0.2862 | 0.675 |
| CB/HIG | Chubb Limited / Hartford (The) | Property & Casualty Insurance | 0.753 | 0.723 | 0.3723 | 0.7189 | 1.293 |
| CRL/IQV | Charles River Laboratories / IQVIA | Life Sciences Tools & Services | 0.743 | 0.705 | 0.2317 | 0.7678 | 1.078 |
| FRT/SPG | Federal Realty Investment Trust / Simon Property Group | Retail REITs | 0.738 | 0.653 | 0.5205 | 0.1896 | 0.233 |
| KIM/SPG | Kimco Realty / Simon Property Group | Retail REITs | 0.733 | 0.651 | 0.4003 | 0.2675 | 0.087 |

## Exploratory mean-reversion candidates, selected on training data

| Pair | Companies | Industry | Train r | Later r | Train coint p | Later coint p | Beta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ESS/VMRK | Essex Property Trust / Vivmark Residential | Multi-Family Residential REITs | 0.905 | 0.835 | 0.03932 | 0.2554 | 4.866 |
| CPT/VMRK | Camden Property Trust / Vivmark Residential | Multi-Family Residential REITs | 0.841 | 0.823 | 0.007045 | 0.6391 | 1.853 |
| CPT/ESS | Camden Property Trust / Essex Property Trust | Multi-Family Residential REITs | 0.831 | 0.828 | 0.0001604 | 0.7188 | 0.370 |

## All pairs with training daily-return correlation at least 0.80

| Pair | Companies | Industry | Train r | Later r | Train coint p | Later coint p | Beta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| CPT/MAA | Camden Property Trust / Mid-America Apartment Communities | Multi-Family Residential REITs | 0.910 | 0.926 | 0.4248 | 0.3116 | 0.761 |
| UDR/VMRK | UDR, Inc. / Vivmark Residential | Multi-Family Residential REITs | 0.907 | 0.823 | 0.08341 | 0.6388 | 0.604 |
| ESS/VMRK | Essex Property Trust / Vivmark Residential | Multi-Family Residential REITs | 0.905 | 0.835 | 0.03932 | 0.2554 | 4.866 |
| CPT/UDR | Camden Property Trust / UDR, Inc. | Multi-Family Residential REITs | 0.897 | 0.868 | 0.07625 | 0.1535 | 3.010 |
| ESS/UDR | Essex Property Trust / UDR, Inc. | Multi-Family Residential REITs | 0.890 | 0.851 | 0.1102 | 0.9578 | 7.883 |
| MAA/UDR | Mid-America Apartment Communities / UDR, Inc. | Multi-Family Residential REITs | 0.872 | 0.861 | 0.2852 | 0.2697 | 3.744 |
| FRT/KIM | Federal Realty Investment Trust / Kimco Realty | Retail REITs | 0.867 | 0.733 | 0.6912 | 0.2025 | 2.818 |
| FRT/REG | Federal Realty Investment Trust / Regency Centers | Retail REITs | 0.863 | 0.802 | 0.6602 | 0.715 | 0.664 |
| CPT/VMRK | Camden Property Trust / Vivmark Residential | Multi-Family Residential REITs | 0.841 | 0.823 | 0.007045 | 0.6391 | 1.853 |
| CPT/ESS | Camden Property Trust / Essex Property Trust | Multi-Family Residential REITs | 0.831 | 0.828 | 0.0001604 | 0.7188 | 0.370 |
| KIM/REG | Kimco Realty / Regency Centers | Retail REITs | 0.827 | 0.786 | 0.4611 | 0.3754 | 0.299 |
| MAA/VMRK | Mid-America Apartment Communities / Vivmark Residential | Multi-Family Residential REITs | 0.818 | 0.832 | 0.3362 | 0.6451 | 2.247 |
| ESS/MAA | Essex Property Trust / Mid-America Apartment Communities | Multi-Family Residential REITs | 0.815 | 0.807 | 0.08368 | 0.9673 | 1.951 |
| A/WAT | Agilent Technologies / Waters Corporation | Life Sciences Tools & Services | 0.810 | 0.642 | 0.7853 | 0.02651 | 0.175 |

