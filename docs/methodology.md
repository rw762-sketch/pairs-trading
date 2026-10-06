# Methodology

This is an exploratory cluster-assisted pairs-trading study. K-means reduces the number of candidate comparisons; Engle–Granger tests screen their statistical relationship; z-scores drive a fixed-rule long/short simulation. The current official version is specified in [selected_strategy.json](../selected_strategy.json), with results and limitations in [research-results.md](research-results.md).

## Data and chronological separation

Daily Yahoo adjusted prices are cached for a current S&P 500 constituent universe. Training eligibility requires complete, finite, positive prices in the formation period. The universe is not a historical constituent reconstruction: applying it to earlier years introduces survivorship bias. Present adjusted histories are not point-in-time data vintages or executable historical quotes.

For the current official replay, development prices end on **2025-09-16**. The final **252** trading closes, **2025-09-17–2026-09-17**, are reserved from fitting. Six overlapping 252-close formation periods provide persistence p-values; these screens are not six separate trading evaluations. The sixth ends on 2025-03-24. A separate last-252-close fit, **2024-09-13–2025-09-16**, supplies the final candidate screen, hedge ratio and recovery estimates.

This chronological separation prevents test prices from entering those fits. It does **not** make the final period untouched: earlier experiments inspected its returns, and the current signal/selection version was chosen after comparing those results. The specification therefore records `fresh_holdout: false`. A genuinely prospective evaluation requires freezing the complete selection, sizing and trading process before observing new prices.

## Candidate selection

1. Represent each stock by its formation-period daily log-return sequence, standardized by that stock's own mean and standard deviation. Fit K-means with ten requested clusters, seed 42 and ten initializations. Constant-return series are excluded. Test every unique pair within each cluster; no correlation or sector cutoff precedes those tests.
2. Fit `P_A = alpha + beta * P_B + residual` on the final formation period. Run Engle–Granger with an intercept and AIC lag selection. Positive beta and provisional I(1) diagnostics govern trading eligibility after testing. The I(1) screen requires level ADF p ≥ 0.05 and first-difference ADF p < 0.05 for both stocks; it is a diagnostic rather than proof of the assumed process.
3. Require raw cointegration p < 0.05 in at least **3/6** formation periods, including the sixth. Require the separate final formation screen to pass as well. Failed or missing persistence observations count as failures. The active version uses raw p-values; Holm correction is available in the screening module but is not the active rule.
4. Fit an AR(1) to formation residuals. For `0 < phi < 1`, estimate half-life as `−log(2) / log(phi)` and retain values between **1 and 20 trading closes**. Require at least twelve annualized crossings of the formation residual mean. Retain pairs in the screen's deterministic ordering while preventing reuse of any stock.

The screens examine many pairs and the windows overlap. Nominal p-values and pass counts therefore do not provide family-wide error control or independent replications. Estimated half-life describes the fitted historical process; it does not guarantee that an individual trade recovers within that many days.

Implementation: [stock_clustering.py](../stock_clustering.py), [cointegration.py](../cointegration.py), [selection.py](../selection.py), [run_persistence.py](../run_persistence.py).

## Allocation and signals

For each retained pair, let `k` be its number of passing periods and `mean_p` its average p-value across **all six** periods, including nonsignificant ones. Missing p-values are replaced by 1. Compute `score = k / max(mean_p, 1e-6)` and `weight = score / sum(scores)`. This is a heuristic confidence score, not a formal combined p-value or an optimal risk allocation. There is no 20% allocation cap; pair budgets sum to 100%, remain fixed, and are not redistributed when a sleeve is inactive.

With beta fixed from the final pre-trading formation, the spread is `P_A − beta * P_B`. At each close, z equals today's spread minus the **preceding 60 closes'** mean, divided by their sample standard deviation. Subtracting the fixed fitted alpha would give the same rolling z-score because the constant cancels. Prior history warms up the first evaluation signal; the new account still starts flat.

| Event | Close-time rule |
|---|---|
| Enter long spread | `−3.5 < z < −2` |
| Enter short spread | `2 < z < 3.5` |
| Long convergence exit | `z >= −0.5` |
| Short convergence exit | `z <= 0.5` |
| Adverse stop | Long `z <= −3.5`; short `z >= 3.5` |
| Invalid z-score | Exit an existing position |
| Evaluation end | Liquidate at the last close |

Signals execute at the **next close**, including normal and stop exits. There is no 20-day or other maximum holding timeout in the official version. A stop threshold does not cap dollar losses, and z can normalize because its rolling mean or standard deviation changes rather than because the original spread fully recovers.

Implementation: [allocation.py](../allocation.py), [signal_generation.py](../signal_generation.py), [evaluation.py](../evaluation.py).

## Accounting and performance

For an allocated budget `C` and positive beta, a long spread enters with `q_A = C / (P_A + beta * P_B)` and `q_B = −beta * q_A`; a short spread reverses both signs. Gross entry notional across both legs equals `C`. Shares stay fixed until exit. The simulator marks cash plus both signed share positions to market and calculates realized two-leg P&L.

The modeled transaction cost is **0.10% of actual gross notional** on every entry and exit, including both legs. Borrowing costs accrue on the preceding close's short notional at **2% annually**, using calendar days over 365.25. The final liquidation includes fees. Pair accounts are scaled by fixed weights into one $100,000 portfolio, and net trade P&L must reconcile with its final equity change.

CAGR uses actual calendar time. Volatility and Sharpe annualize daily returns using 252 trading days; the reported Sharpe uses a **zero risk-free rate**. A Sharpe above zero does not demonstrate excess return over bank interest or Treasury yields. Cash earns zero in this model. Adjusted-price accounting is a research proxy and does not reproduce all broker dividend cash flows, funding, margin, short availability or executable spreads/slippage. These cost and execution assumptions require sensitivity analysis before making investment claims.

Implementation: [backtester.py](../backtester.py). Timing and accounting checks are in [test_backtester.py](../tests/test_backtester.py), with pipeline, clustering and recovery checks under [tests](../tests).

## Additional evidence and audit scope

The historical 2021 run keeps the current strategy rules fixed, re-estimates pairs and weights using 2017–2020 data, and trades only January–December 2021. The temporary preview uses a different data layout: eight development years, ten overlapping 252-close screens requiring **5/10** including the latest, followed by two years of trading. It does not replace the official six-screen version. Its two retained pairs and concentrated profit warrant further evaluation rather than an inference of general robustness.

Recorded checks recomputed selected fits and weights, reconciled net/gross P&L, replayed metrics, and perturbed later prices to verify unchanged earlier signals and equity. The software suite contains 43 checks, including the original 39 plus four frozen-input replay and integrity checks. These establish implementation properties conditional on fixed rules; they do not correct ex-post model selection, current-constituent bias or repeated use of the evaluation period.

One implementation limitation is the shared evaluator's whole-period finite-price check: a future missing price can place the entire sleeve in cash. All selected sleeves in the reported runs had complete warmup/evaluation prices, so this branch did not affect these results. The historical runner validates retained sleeves after freezing selection and aborts on gaps instead of changing the pairs or weights.

## Reproducing the records

The [compact evidence](../reports/manifest.json) preserves saved CSVs and hashes without private machine paths or full acquisition metadata. Portfolio metrics can be reconciled from included equity/trades; allocation can be inspected alongside selected fits. The separate [official replay bundle](../reproducibility/selected/README.md) contains the frozen adjusted-price snapshot and training screening inputs needed for the default replay. Its [manifest](../reproducibility/selected/manifest.json) records integrity hashes. The earlier-year and temporary preview source-price snapshots are not included; reproducing their price-based fits requires those inputs or newly acquired data.

After installing the requirements, `python main.py` runs the official pinned replay using the bundled frozen inputs when local cache or archived screens are absent. It verifies the bundle and price snapshot hashes and can reproduce the saved +3.900813% result without downloading new prices. [run_selected.py](../run_selected.py) reuses the archived training screens and checks pair identity; it does not perform fresh clustering on every replay. [run_historical_year.py](../run_historical_year.py) supports reconstructing the earlier-year robustness workflow with newly acquired source data. New downloads may revise prices, tickers or observations and need not reproduce the saved snapshots exactly.

The temporary preview evidence is supplied for comparison; it is not a command-line default or a promoted production strategy. Each report's source paths, price snapshot hash, input-plan hash and copied-file hashes are listed in the manifest.
