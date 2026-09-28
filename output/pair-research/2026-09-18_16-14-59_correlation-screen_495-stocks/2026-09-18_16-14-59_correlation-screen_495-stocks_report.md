# Pairs research: correlation screen

Training: 2023-09-19 to 2025-10-22; later diagnostic period: 2025-10-23 to 2026-09-17.

503 requested stocks; 495 with complete positive training prices; 122,265 pair correlations; 134 with training correlation >= 0.80.

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
| GOOG/GOOGL | Alphabet Inc. (Class C) / Alphabet Inc. (Class A) | Interactive Media & Services (same issuer) | 0.998 | 0.996 | 0.04861 | 0.2255 | 0.994 |
| FOX/FOXA | Fox Corporation (Class B) / Fox Corporation (Class A) | Broadcasting (same issuer) | 0.987 | 0.985 | 0.901 | 0.1027 | 0.913 |
| NWS/NWSA | News Corp (Class B) / News Corp (Class A) | Publishing (same issuer) | 0.959 | 0.961 | 0.6949 | 0.3962 | 1.374 |
| LEN/PHM | Lennar / PulteGroup | Homebuilding | 0.918 | 0.859 | 0.952 | 0.6388 | 0.926 |
| DHI/PHM | D. R. Horton / PulteGroup | Homebuilding | 0.915 | 0.911 | 0.543 | 0.06423 | 1.207 |
| KLAC/LRCX | KLA Corporation / Lam Research | Semiconductor Materials & Equipment | 0.913 | 0.874 | 0.65 | 0.09887 | 0.735 |
| AMAT/KLAC | Applied Materials / KLA Corporation | Semiconductor Materials & Equipment | 0.912 | 0.872 | 0.6946 | 0.3024 | 1.153 |
| AMAT/LRCX | Applied Materials / Lam Research | Semiconductor Materials & Equipment | 0.911 | 0.905 | 0.7749 | 0.3363 | 1.115 |
| CPT/MAA | Camden Property Trust / Mid-America Apartment Communities | Multi-Family Residential REITs | 0.910 | 0.926 | 0.4248 | 0.3116 | 0.761 |
| UDR/VMRK | UDR, Inc. / Vivmark Residential | Multi-Family Residential REITs | 0.907 | 0.823 | 0.08341 | 0.6388 | 0.604 |
| DHI/LEN | D. R. Horton / Lennar | Homebuilding | 0.906 | 0.882 | 0.8902 | 0.08857 | 0.857 |
| ESS/VMRK | Essex Property Trust / Vivmark Residential | Multi-Family Residential REITs | 0.905 | 0.835 | 0.03932 | 0.2554 | 4.866 |
| CPT/UDR | Camden Property Trust / UDR, Inc. | Multi-Family Residential REITs | 0.897 | 0.868 | 0.07625 | 0.1535 | 3.010 |
| DOW/LYB | Dow Inc. / LyondellBasell | Different industries | 0.896 | 0.905 | 0.06566 | 0.9586 | 0.732 |
| CFG/HBAN | Citizens Financial Group / Huntington Bancshares | Regional Banks | 0.893 | 0.830 | 0.8779 | 0.8568 | 3.325 |
| MLM/VMC | Martin Marietta Materials / Vulcan Materials Company | Construction Materials | 0.892 | 0.877 | 0.4971 | 0.5467 | 1.989 |
| HBAN/RF | Huntington Bancshares / Regions Financial Corporation | Regional Banks | 0.892 | 0.839 | 0.1175 | 0.7819 | 0.651 |
| FITB/HBAN | Fifth Third Bancorp / Huntington Bancshares | Regional Banks | 0.891 | 0.881 | 0.6043 | 0.9645 | 2.539 |
| FITB/RF | Fifth Third Bancorp / Regions Financial Corporation | Regional Banks | 0.891 | 0.882 | 0.129 | 0.02796 | 1.704 |
| CFG/FITB | Citizens Financial Group / Fifth Third Bancorp | Regional Banks | 0.890 | 0.869 | 0.9821 | 0.06753 | 1.225 |

## Exploratory mean-reversion candidates, selected on training data

| Pair | Companies | Industry | Train r | Later r | Train coint p | Later coint p | Beta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ESS/VMRK | Essex Property Trust / Vivmark Residential | Multi-Family Residential REITs | 0.905 | 0.835 | 0.03932 | 0.2554 | 4.866 |
| KEY/RF | KeyCorp / Regions Financial Corporation | Regional Banks | 0.865 | 0.878 | 0.001586 | 0.6038 | 0.706 |
| PNC/TFC | PNC Financial Services / Truist Financial | Diversified Banks | 0.864 | 0.843 | 0.02243 | 0.5241 | 4.657 |
| COP/DVN | ConocoPhillips / Devon Energy | Oil & Gas Exploration & Production | 0.860 | 0.820 | 0.04492 | 0.2292 | 1.641 |
| MCO/SPGI | Moody's Corporation / S&P Global | Financial Exchanges & Data | 0.849 | 0.881 | 0.04552 | 0.345 | 1.148 |
| CPT/VMRK | Camden Property Trust / Vivmark Residential | Multi-Family Residential REITs | 0.841 | 0.823 | 0.007045 | 0.6391 | 1.853 |
| CPT/ESS | Camden Property Trust / Essex Property Trust | Multi-Family Residential REITs | 0.831 | 0.828 | 0.0001604 | 0.7188 | 0.370 |
| COP/OXY | ConocoPhillips / Occidental Petroleum | Oil & Gas Exploration & Production | 0.823 | 0.815 | 0.008756 | 0.3892 | 1.225 |
| DUK/SO | Duke Energy / Southern Company | Electric Utilities | 0.823 | 0.865 | 0.02391 | 0.17 | 1.205 |

## All pairs with training daily-return correlation at least 0.80

| Pair | Companies | Industry | Train r | Later r | Train coint p | Later coint p | Beta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| GOOG/GOOGL | Alphabet Inc. (Class C) / Alphabet Inc. (Class A) | Interactive Media & Services (same issuer) | 0.998 | 0.996 | 0.04861 | 0.2255 | 0.994 |
| FOX/FOXA | Fox Corporation (Class B) / Fox Corporation (Class A) | Broadcasting (same issuer) | 0.987 | 0.985 | 0.901 | 0.1027 | 0.913 |
| NWS/NWSA | News Corp (Class B) / News Corp (Class A) | Publishing (same issuer) | 0.959 | 0.961 | 0.6949 | 0.3962 | 1.374 |
| LEN/PHM | Lennar / PulteGroup | Homebuilding | 0.918 | 0.859 | 0.952 | 0.6388 | 0.926 |
| DHI/PHM | D. R. Horton / PulteGroup | Homebuilding | 0.915 | 0.911 | 0.543 | 0.06423 | 1.207 |
| KLAC/LRCX | KLA Corporation / Lam Research | Semiconductor Materials & Equipment | 0.913 | 0.874 | 0.65 | 0.09887 | 0.735 |
| AMAT/KLAC | Applied Materials / KLA Corporation | Semiconductor Materials & Equipment | 0.912 | 0.872 | 0.6946 | 0.3024 | 1.153 |
| AMAT/LRCX | Applied Materials / Lam Research | Semiconductor Materials & Equipment | 0.911 | 0.905 | 0.7749 | 0.3363 | 1.115 |
| CPT/MAA | Camden Property Trust / Mid-America Apartment Communities | Multi-Family Residential REITs | 0.910 | 0.926 | 0.4248 | 0.3116 | 0.761 |
| UDR/VMRK | UDR, Inc. / Vivmark Residential | Multi-Family Residential REITs | 0.907 | 0.823 | 0.08341 | 0.6388 | 0.604 |
| DHI/LEN | D. R. Horton / Lennar | Homebuilding | 0.906 | 0.882 | 0.8902 | 0.08857 | 0.857 |
| ESS/VMRK | Essex Property Trust / Vivmark Residential | Multi-Family Residential REITs | 0.905 | 0.835 | 0.03932 | 0.2554 | 4.866 |
| CPT/UDR | Camden Property Trust / UDR, Inc. | Multi-Family Residential REITs | 0.897 | 0.868 | 0.07625 | 0.1535 | 3.010 |
| DOW/LYB | Dow Inc. / LyondellBasell | Different industries | 0.896 | 0.905 | 0.06566 | 0.9586 | 0.732 |
| CFG/HBAN | Citizens Financial Group / Huntington Bancshares | Regional Banks | 0.893 | 0.830 | 0.8779 | 0.8568 | 3.325 |
| MLM/VMC | Martin Marietta Materials / Vulcan Materials Company | Construction Materials | 0.892 | 0.877 | 0.4971 | 0.5467 | 1.989 |
| HBAN/RF | Huntington Bancshares / Regions Financial Corporation | Regional Banks | 0.892 | 0.839 | 0.1175 | 0.7819 | 0.651 |
| FITB/HBAN | Fifth Third Bancorp / Huntington Bancshares | Regional Banks | 0.891 | 0.881 | 0.6043 | 0.9645 | 2.539 |
| FITB/RF | Fifth Third Bancorp / Regions Financial Corporation | Regional Banks | 0.891 | 0.882 | 0.129 | 0.02796 | 1.704 |
| CFG/FITB | Citizens Financial Group / Fifth Third Bancorp | Regional Banks | 0.890 | 0.869 | 0.9821 | 0.06753 | 1.225 |
| ESS/UDR | Essex Property Trust / UDR, Inc. | Multi-Family Residential REITs | 0.890 | 0.851 | 0.1102 | 0.9578 | 7.883 |
| MET/PRU | MetLife / Prudential Financial | Life & Health Insurance | 0.889 | 0.693 | 0.9322 | 0.677 | 0.639 |
| HBAN/PNC | Huntington Bancshares / PNC Financial Services | Different industries | 0.888 | 0.850 | 0.7505 | 0.4911 | 0.086 |
| CFG/KEY | Citizens Financial Group / KeyCorp | Regional Banks | 0.888 | 0.892 | 0.6431 | 0.8575 | 3.108 |
| CFG/TFC | Citizens Financial Group / Truist Financial | Different industries | 0.888 | 0.832 | 0.7776 | 0.7548 | 1.369 |
| MPC/VLO | Marathon Petroleum / Valero Energy | Oil & Gas Refining & Marketing | 0.883 | 0.864 | 0.05042 | 0.2034 | 1.130 |
| HD/LOW | Home Depot (The) / Lowe's | Home Improvement Retail | 0.881 | 0.878 | 0.5519 | 0.3487 | 1.519 |
| FITB/PNC | Fifth Third Bancorp / PNC Financial Services | Different industries | 0.880 | 0.864 | 0.3745 | 0.0005349 | 0.228 |
| DAL/UAL | Delta Air Lines / United Airlines Holdings | Passenger Airlines | 0.878 | 0.898 | 0.5356 | 0.7951 | 0.351 |
| FITB/USB | Fifth Third Bancorp / U.S. Bancorp | Different industries | 0.878 | 0.855 | 0.2915 | 0.07127 | 1.161 |
| HBAN/TFC | Huntington Bancshares / Truist Financial | Different industries | 0.878 | 0.791 | 0.2325 | 0.1832 | 0.398 |
| CFG/MTB | Citizens Financial Group / M&T Bank | Regional Banks | 0.877 | 0.864 | 0.9482 | 0.124 | 0.260 |
| HBAN/KEY | Huntington Bancshares / KeyCorp | Regional Banks | 0.875 | 0.851 | 0.1102 | 0.4618 | 0.913 |
| MTB/RF | M&T Bank / Regions Financial Corporation | Regional Banks | 0.874 | 0.902 | 0.4176 | 0.2112 | 8.082 |
| CFG/PNC | Citizens Financial Group / PNC Financial Services | Different industries | 0.874 | 0.843 | 0.9882 | 0.07471 | 0.291 |
| CFG/RF | Citizens Financial Group / Regions Financial Corporation | Regional Banks | 0.873 | 0.880 | 0.4427 | 0.3731 | 2.264 |
| MAA/UDR | Mid-America Apartment Communities / UDR, Inc. | Multi-Family Residential REITs | 0.872 | 0.861 | 0.2852 | 0.2697 | 3.744 |
| FITB/TFC | Fifth Third Bancorp / Truist Financial | Different industries | 0.868 | 0.843 | 0.03627 | 0.5359 | 1.079 |
| FRT/KIM | Federal Realty Investment Trust / Kimco Realty | Retail REITs | 0.867 | 0.733 | 0.6912 | 0.2025 | 2.818 |
| DVN/FANG | Devon Energy / Diamondback Energy | Oil & Gas Exploration & Production | 0.866 | 0.828 | 0.4235 | 0.6344 | 0.186 |
| DVN/OXY | Devon Energy / Occidental Petroleum | Oil & Gas Exploration & Production | 0.866 | 0.819 | 0.1513 | 0.1248 | 0.724 |
| HBAN/MTB | Huntington Bancshares / M&T Bank | Regional Banks | 0.866 | 0.830 | 0.1675 | 0.4989 | 0.078 |
| PNC/RF | PNC Financial Services / Regions Financial Corporation | Different industries | 0.866 | 0.835 | 0.01439 | 0.07711 | 7.541 |
| FITB/MTB | Fifth Third Bancorp / M&T Bank | Regional Banks | 0.865 | 0.869 | 0.4534 | 0.09162 | 0.200 |
| KEY/RF | KeyCorp / Regions Financial Corporation | Regional Banks | 0.865 | 0.878 | 0.001586 | 0.6038 | 0.706 |
| HLT/MAR | Hilton Worldwide / Marriott International | Hotels, Resorts & Cruise Lines | 0.865 | 0.824 | 0.8167 | 0.001011 | 1.199 |
| PNC/TFC | PNC Financial Services / Truist Financial | Diversified Banks | 0.864 | 0.843 | 0.02243 | 0.5241 | 4.657 |
| FRT/REG | Federal Realty Investment Trust / Regency Centers | Retail REITs | 0.863 | 0.802 | 0.6602 | 0.715 | 0.664 |
| EXR/PSA | Extra Space Storage / Public Storage | Self-Storage REITs | 0.863 | 0.867 | 0.3171 | 0.04235 | 0.521 |
| CMS/DTE | CMS Energy / DTE Energy | Multi-Utilities | 0.861 | 0.865 | 0.2645 | 0.7326 | 0.488 |
| CMS/WEC | CMS Energy / WEC Energy Group | Different industries | 0.860 | 0.825 | 0.4144 | 0.4525 | 0.529 |
| COP/DVN | ConocoPhillips / Devon Energy | Oil & Gas Exploration & Production | 0.860 | 0.820 | 0.04492 | 0.2292 | 1.641 |
| GS/MS | Goldman Sachs / Morgan Stanley | Investment Banking & Brokerage | 0.859 | 0.864 | 0.06726 | 0.05974 | 5.393 |
| MTB/PNC | M&T Bank / PNC Financial Services | Different industries | 0.858 | 0.859 | 0.1364 | 0.007232 | 1.065 |
| EME/FIX | Emcor / Comfort Systems USA | Construction & Engineering | 0.856 | 0.807 | 0.6878 | 0.1305 | 0.750 |
| MA/V | Mastercard / Visa Inc. | Transaction & Payment Processing Services | 0.856 | 0.859 | 0.4452 | 0.5241 | 1.473 |
| ADI/MCHP | Analog Devices / Microchip Technology | Semiconductors | 0.856 | 0.805 | 0.4995 | 0.487 | -0.105 |
| COP/EOG | ConocoPhillips / EOG Resources | Oil & Gas Exploration & Production | 0.856 | 0.861 | 0.779 | 0.7493 | 0.565 |
| ADI/NXPI | Analog Devices / NXP Semiconductors | Semiconductors | 0.855 | 0.691 | 0.8601 | 0.5612 | 0.540 |
| FITB/KEY | Fifth Third Bancorp / KeyCorp | Regional Banks | 0.854 | 0.843 | 0.2574 | 0.607 | 2.415 |
| HBAN/USB | Huntington Bancshares / U.S. Bancorp | Different industries | 0.853 | 0.832 | 0.3481 | 0.5637 | 0.430 |
| DHI/NVR | D. R. Horton / NVR, Inc. | Homebuilding | 0.853 | 0.732 | 0.7601 | 0.03721 | 0.020 |
| NVR/PHM | NVR, Inc. / PulteGroup | Homebuilding | 0.852 | 0.748 | 0.4545 | 0.6652 | 52.678 |
| PNC/USB | PNC Financial Services / U.S. Bancorp | Diversified Banks | 0.851 | 0.892 | 0.1243 | 0.2509 | 5.014 |
| MCO/SPGI | Moody's Corporation / S&P Global | Financial Exchanges & Data | 0.849 | 0.881 | 0.04552 | 0.345 | 1.148 |
| PSX/VLO | Phillips 66 / Valero Energy | Oil & Gas Refining & Marketing | 0.849 | 0.837 | 0.3278 | 0.2262 | 0.694 |
| KEY/TFC | KeyCorp / Truist Financial | Different industries | 0.848 | 0.830 | 0.2104 | 0.5416 | 0.432 |
| RF/USB | Regions Financial Corporation / U.S. Bancorp | Different industries | 0.848 | 0.841 | 0.2073 | 0.04863 | 0.638 |
| NUE/STLD | Nucor / Steel Dynamics | Steel | 0.848 | 0.873 | 0.7819 | 0.7106 | 0.177 |
| DVN/EOG | Devon Energy / EOG Resources | Oil & Gas Exploration & Production | 0.848 | 0.857 | 0.8944 | 0.5068 | 0.233 |
| KMI/WMB | Kinder Morgan / Williams Companies | Oil & Gas Storage & Transportation | 0.846 | 0.832 | 0.2566 | 0.1741 | 0.448 |
| AEE/LNT | Ameren / Alliant Energy | Different industries | 0.845 | 0.879 | 0.2883 | 0.3511 | 1.735 |
| LEN/NVR | Lennar / NVR, Inc. | Homebuilding | 0.844 | 0.698 | 0.8503 | 0.4001 | 0.018 |
| APA/DVN | APA Corporation / Devon Energy | Oil & Gas Exploration & Production | 0.844 | 0.808 | 0.4296 | 0.6787 | 0.863 |
| KEY/PNC | KeyCorp / PNC Financial Services | Different industries | 0.843 | 0.860 | 0.009837 | 0.4754 | 0.092 |
| COF/SYF | Capital One / Synchrony Financial | Consumer Finance | 0.842 | 0.831 | 0.4588 | 0.3025 | 2.605 |
| CFG/USB | Citizens Financial Group / U.S. Bancorp | Different industries | 0.841 | 0.872 | 0.5619 | 0.3517 | 1.438 |
| CPT/VMRK | Camden Property Trust / Vivmark Residential | Multi-Family Residential REITs | 0.841 | 0.823 | 0.007045 | 0.6391 | 1.853 |
| LNT/WEC | Alliant Energy / WEC Energy Group | Electric Utilities | 0.840 | 0.874 | 0.291 | 0.4475 | 0.520 |
| CCL/NCLH | Carnival Corporation / Norwegian Cruise Line Holdings | Hotels, Resorts & Cruise Lines | 0.840 | 0.830 | 0.8648 | 0.09868 | 1.168 |
| MTB/USB | M&T Bank / U.S. Bancorp | Different industries | 0.839 | 0.863 | 0.5637 | 0.04291 | 5.234 |
| CMS/LNT | CMS Energy / Alliant Energy | Different industries | 0.838 | 0.832 | 0.2489 | 0.9585 | 1.010 |
| MCHP/NXPI | Microchip Technology / NXP Semiconductors | Semiconductors | 0.836 | 0.782 | 0.7114 | 0.5227 | 0.354 |
| ADI/TXN | Analog Devices / Texas Instruments | Semiconductors | 0.835 | 0.819 | 0.8757 | 0.3631 | 1.074 |
| MPC/PSX | Marathon Petroleum / Phillips 66 | Oil & Gas Refining & Marketing | 0.835 | 0.853 | 0.7574 | 0.7032 | 1.202 |
| COP/FANG | ConocoPhillips / Diamondback Energy | Oil & Gas Exploration & Production | 0.834 | 0.787 | 0.3322 | 0.693 | 0.308 |
| AEE/WEC | Ameren / WEC Energy Group | Different industries | 0.833 | 0.863 | 0.06453 | 0.2528 | 0.947 |
| AMT/SBAC | American Tower / SBA Communications | Telecom Tower REITs | 0.833 | 0.702 | 0.4206 | 0.01717 | 0.837 |
| TFC/USB | Truist Financial / U.S. Bancorp | Diversified Banks | 0.832 | 0.867 | 0.1449 | 0.2418 | 1.048 |
| MET/PFG | MetLife / Principal Financial Group | Life & Health Insurance | 0.831 | 0.689 | 0.3345 | 0.4102 | 1.283 |
| CPT/ESS | Camden Property Trust / Essex Property Trust | Multi-Family Residential REITs | 0.831 | 0.828 | 0.0001604 | 0.7188 | 0.370 |
| ARES/KKR | Ares Management / KKR & Co. | Asset Management & Custody Banks | 0.830 | 0.823 | 0.2564 | 0.3192 | 0.888 |
| CCL/RCL | Carnival Corporation / Royal Caribbean Group | Hotels, Resorts & Cruise Lines | 0.830 | 0.798 | 0.1999 | 0.4104 | 0.071 |
| DTE/LNT | DTE Energy / Alliant Energy | Different industries | 0.828 | 0.840 | 0.3445 | 0.1312 | 2.011 |
| RF/TFC | Regions Financial Corporation / Truist Financial | Different industries | 0.828 | 0.861 | 0.02832 | 0.2591 | 0.590 |
| KIM/REG | Kimco Realty / Regency Centers | Retail REITs | 0.827 | 0.786 | 0.4611 | 0.3754 | 0.299 |
| MTB/TFC | M&T Bank / Truist Financial | Different industries | 0.825 | 0.841 | 0.2883 | 0.6729 | 4.893 |
| ED/WEC | Consolidated Edison / WEC Energy Group | Different industries | 0.824 | 0.777 | 0.5624 | 0.5199 | 0.517 |
| KEY/USB | KeyCorp / U.S. Bancorp | Different industries | 0.823 | 0.863 | 0.07643 | 0.6621 | 0.472 |
| COP/OXY | ConocoPhillips / Occidental Petroleum | Oil & Gas Exploration & Production | 0.823 | 0.815 | 0.008756 | 0.3892 | 1.225 |
| DUK/SO | Duke Energy / Southern Company | Electric Utilities | 0.823 | 0.865 | 0.02391 | 0.17 | 1.205 |
| KEY/MTB | KeyCorp / M&T Bank | Regional Banks | 0.821 | 0.852 | 0.1651 | 0.3333 | 0.082 |
| BAC/HBAN | Bank of America / Huntington Bancshares | Different industries | 0.820 | 0.645 | 0.9145 | 0.9024 | 2.889 |
| APA/OXY | APA Corporation / Occidental Petroleum | Oil & Gas Exploration & Production | 0.820 | 0.810 | 0.5985 | 0.7061 | 0.692 |
| EQT/EXE | EQT Corporation / Expand Energy | Oil & Gas Exploration & Production | 0.820 | 0.807 | 0.2558 | 0.6375 | 0.585 |
| OKE/TRGP | Oneok / Targa Resources | Oil & Gas Storage & Transportation | 0.820 | 0.737 | 0.9518 | 0.1529 | 0.269 |
| CMS/ED | CMS Energy / Consolidated Edison | Multi-Utilities | 0.820 | 0.765 | 0.5335 | 0.929 | 0.854 |
| APO/KKR | Apollo Global Management / KKR & Co. | Asset Management & Custody Banks | 0.819 | 0.808 | 0.3622 | 0.2002 | 0.835 |
| PFG/PRU | Principal Financial Group / Prudential Financial | Life & Health Insurance | 0.819 | 0.602 | 0.1273 | 0.7916 | 0.445 |
| MAA/VMRK | Mid-America Apartment Communities / Vivmark Residential | Multi-Family Residential REITs | 0.818 | 0.832 | 0.3362 | 0.6451 | 2.247 |
| BAC/WFC | Bank of America / Wells Fargo | Diversified Banks | 0.815 | 0.795 | 0.3992 | 0.9193 | 0.490 |
| BAC/C | Bank of America / Citigroup | Diversified Banks | 0.815 | 0.725 | 0.4023 | 0.8233 | 0.397 |
| ESS/MAA | Essex Property Trust / Mid-America Apartment Communities | Multi-Family Residential REITs | 0.815 | 0.807 | 0.08368 | 0.9673 | 1.951 |
| HAL/SLB | Halliburton / Schlumberger | Oil & Gas Equipment & Services | 0.814 | 0.738 | 0.2469 | 0.5572 | 0.890 |
| HIG/L | Hartford (The) / Loews Corporation | Different industries | 0.812 | 0.749 | 0.9113 | 0.09623 | 1.821 |
| A/WAT | Agilent Technologies / Waters Corporation | Life Sciences Tools & Services | 0.810 | 0.642 | 0.7853 | 0.02651 | 0.175 |
| RSG/WM | Republic Services / Waste Management | Environmental & Facilities Services | 0.809 | 0.818 | 0.3646 | 0.289 | 1.333 |
| CVX/XOM | Chevron Corporation / ExxonMobil | Integrated Oil & Gas | 0.808 | 0.843 | 0.1456 | 0.1392 | 0.774 |
| SO/WEC | Southern Company / WEC Energy Group | Electric Utilities | 0.807 | 0.767 | 0.6896 | 0.08248 | 0.760 |
| DTE/WEC | DTE Energy / WEC Energy Group | Different industries | 0.807 | 0.831 | 0.296 | 0.05168 | 1.070 |
| BAC/FITB | Bank of America / Fifth Third Bancorp | Different industries | 0.806 | 0.647 | 0.9051 | 0.6643 | 1.045 |
| EVRG/LNT | Evergy / Alliant Energy | Electric Utilities | 0.806 | 0.837 | 0.2198 | 0.9536 | 1.274 |
| DUK/ED | Duke Energy / Consolidated Edison | Different industries | 0.805 | 0.834 | 0.7775 | 0.4071 | 1.508 |
| BX/KKR | Blackstone Inc. / KKR & Co. | Asset Management & Custody Banks | 0.805 | 0.828 | 0.1988 | 0.1187 | 0.860 |
| EOG/FANG | EOG Resources / Diamondback Energy | Oil & Gas Exploration & Production | 0.805 | 0.819 | 0.05656 | 0.07601 | 0.187 |
| AMT/CCI | American Tower / Crown Castle | Telecom Tower REITs | 0.804 | 0.878 | 0.23 | 0.281 | 2.209 |
| FANG/OXY | Diamondback Energy / Occidental Petroleum | Oil & Gas Exploration & Production | 0.804 | 0.790 | 0.3314 | 0.1655 | 1.733 |
| GS/JPM | Goldman Sachs / JPMorgan Chase | Different industries | 0.802 | 0.621 | 0.04914 | 0.5402 | 2.751 |
| DTE/PPL | DTE Energy / PPL Corporation | Different industries | 0.802 | 0.663 | 0.03827 | 0.6575 | 3.245 |
| CMS/PPL | CMS Energy / PPL Corporation | Different industries | 0.802 | 0.697 | 0.01361 | 0.2626 | 1.627 |
| BAC/CFG | Bank of America / Citizens Financial Group | Different industries | 0.802 | 0.675 | 0.06549 | 0.7029 | 0.847 |
| BAC/PNC | Bank of America / PNC Financial Services | Diversified Banks | 0.801 | 0.706 | 0.9693 | 0.6474 | 0.247 |
| APA/COP | APA Corporation / ConocoPhillips | Oil & Gas Exploration & Production | 0.800 | 0.784 | 0.4327 | 0.5937 | 0.493 |
| ETN/VRT | Eaton Corporation / Vertiv | Electrical Components & Equipment | 0.800 | 0.762 | 0.5098 | 0.4462 | 1.398 |

