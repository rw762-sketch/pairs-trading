# Strategies to investigate with the measured pair list

These are research specifications, not tested profitable strategies or live trade signals. The measured tables are in [correlated-pairs.md](2026-09-18_16-14-59_correlation-screen_495-stocks_report.md); rerun the screen from the project folder with `./venv/bin/python screen_pairs.py` (add `--refresh` to redownload the same window).

## What the screen found

Training: September 19, 2023 through October 22, 2025 (526 price observations). Later diagnostic period: October 23, 2025 through September 17, 2026 (226 price observations).

Of 503 requested securities, 495 had complete positive training prices. There were 122,265 possible pairs, 134 with training daily log-return correlation at least 0.80, and 97 strong-correlation pairs involving different issuers in the same GICS sub-industry. This screen checks eligibility only on training data; the existing pipeline removed missing-data stocks using the entire period and retained 494.

Nine different-company same-industry pairs met the exploratory training screen: ESS/VMRK, KEY/RF, PNC/TFC, COP/DVN, MCO/SPGI, CPT/VMRK, CPT/ESS, COP/OXY and DUK/SO. Their later-period Engle-Granger p-values were all above 0.05. These are not confirmed mean-reversion candidates. A nonsignificant result in a shorter sample does not prove the relationship disappeared.

The largest correlations were GOOG/GOOGL, FOX/FOXA and NWS/NWSA, which are share classes of the same companies. The strongest independent-company examples included LEN/PHM, DHI/PHM, KLAC/LRCX, AMAT/KLAC and AMAT/LRCX. High daily-return correlation does not itself establish a stationary spread.

## 1. Fixed-hedge, same-industry mean reversion

Begin with this specification as the simplest benchmark to understand.

- Candidate rule: same GICS sub-industry, different company CIK, training return correlation >= 0.80, last 126 training-return correlation >= 0.75, positive training hedge ratio, and raw training Engle-Granger p < 0.05. This is an exploratory filter requiring multiple-testing and model-assumption review before deployment.
- Model: on formation data, estimate adjusted-price levels as `A = alpha + beta * B + residual`. Freeze the fitted beta for the subsequent evaluation period. Spread is `A - beta * B`.
- Signal: compare the current close's spread with mean and sample standard deviation of the preceding 60 closes. Require 60 valid prior observations and nonzero standard deviation. Compute signals only after that close and simulate execution at the following close for a transparent daily-data baseline; do not fill at the already-observed signal close.
- Entry: long spread at z < -2; short spread at z > 2. For positive beta, long spread means long A and short beta shares of B per share of A; reverse for a short spread.
- Exit: long spread when z >= -0.5; short spread when z <= 0.5. These rules deliberately exit nearer the mean than the entry threshold.
- Risk exits: long if z <= -3.5; short if z >= 3.5; close after 20 trading sessions regardless. Also close at a 2% net loss relative to the allocated pair capital. All exits execute at the next available simulated price; they are not guarantees against larger losses.
- Initial research examples: KEY/RF and CPT/ESS had training cointegration p-values around 0.0016 and 0.00016, respectively, but did not confirm cointegration in the later window. Use them to investigate model failure as well as convergence; do not describe them as validated profitable trades.

The z thresholds, 60-day window, 20-session holding limit and 2% stop are proposed assumptions, not optimized results.

## 2. Monthly re-estimation of the same-industry strategy

This tests whether a changing relationship needs a changing model.

- At each month-end, apply the prespecified screen to the preceding 252 trading days only. Require full-window return correlation >= 0.80 and trailing 60-return correlation >= 0.70, positive beta, and the predefined cointegration filter.
- Fit the hedge ratio using those past prices. Use strategy 1's signal, entry, exit, risk limits and next-close execution for new trades during the following month.
- Freeze a position's hedge ratio until it closes. Updating beta without booking actual rebalancing trades would misstate holdings and profit.
- If a pair fails the next monthly review, disable entries and close an existing position at the following simulated close under the predeclared rule.
- Compare the entire rolling strategy with strategy 1 under identical capital and cost assumptions. Do not keep only the months or pairs that performed well.

This strategy has not been implemented or backtested by the screening script.

## 3. Relative pricing of two share classes

Use GOOG/GOOGL, FOX/FOXA and NWS/NWSA as a separate research group. They should not dominate a list intended to compare independent companies.

- Investigate the economic and voting rights of each class, corporate actions and borrow availability before interpreting a relative-price premium as a mistake.
- Model `log(price_A / price_B)` relative to its own preceding 120-day mean and standard deviation. Its normal level need not be zero. This model differs from the price-level cointegration tests shown in the screen; those tests do not validate this ratio strategy.
- Entry hypothesis: ratio z > 2.5 means short A/long B; z < -2.5 means long A/short B. Use equal dollar exposure for this specification.
- Exit near the mean: a short ratio when z <= 0.5, a long ratio when z >= -0.5. Adverse stop at +4 for a short ratio or -4 for a long ratio; maximum holding period 5 trading sessions. Use the same capital-loss stop and delayed execution as the baseline.
- Require the modeled opportunity to exceed a predefined estimate of the full round-trip costs. Near-identical returns can leave very little margin after trading and borrowing costs.

These settings are hypotheses. High correlation has not established a stationary ratio or positive after-cost return for these pairs.

## Correct accounting and fair comparisons

The current `backtester.py` must be corrected before using its reported return or Sharpe to compare any strategy. The screen deliberately computes no portfolio return.

For a price spread and a positive beta, let G be the allocated gross exposure and use actual entry execution prices:

`q_A = G / (P_A + abs(beta) * P_B)` and `q_B = -beta * q_A` for a long spread; reverse both signs for a short spread.

Track `q_A * change_in_P_A + q_B * change_in_P_B` and debit fees on the actual absolute value traded on both legs at entry and exit. A regression hedge is not automatically dollar-neutral or market-neutral. Fractional quantities in a research simulation are an approximation that must be stated.

Compare 5, 10 and 20 basis points per traded dollar as assumed one-way fee/slippage scenarios, with separate borrow/financing scenarios. These are sensitivity assumptions, not broker quotes. Shorting also raises dividend and corporate-action accounting questions: do not mix total-return-adjusted prices with another dividend cash flow and count the same effect twice. See the [SEC short-selling explanation](https://www.sec.gov/investor/pubs/regsho.htm).

Initially allow one open position per ticker and cap sector allocation, so several overlapping bank or semiconductor pairs do not become hidden concentrated exposure. Predetermine the ranking used when simultaneous signals compete for capital.

The later 30% window has already been inspected, including in this report. Use historical walk-forward comparisons as exploratory research; freeze the eventual methodology before evaluating on a genuinely new future period. Do not repeatedly tune to the same later sample and call it independent validation.

Report turnover, number of closed trades, exposure, net return, drawdown and daily-return Sharpe from the corrected engine. Check sensitivity across plausible nearby parameters and cost assumptions, including losing runs.

## Statistical limits

- The correlation screen concerns adjusted daily log returns. The Engle-Granger tests concern adjusted price levels; both series are assumed I(1). A separate assessment of integration order has not been completed.
- Test direction is alphabetical and fixed before seeing results. Reverse regressions can give different finite-sample results; selecting the best direction after seeing p-values adds another selection step.
- All p-values in the screen are unadjusted and were examined after screening on the same training data. Multiple comparisons can generate false discoveries. Correct the whole prespecified tested family or use a separate validation sample; correcting only the final winners is inadequate.
- Later-period cointegration is separately fitted and is descriptive. It does not establish profitability using the original frozen hedge ratio.
- The universe is the project's fetched current constituent snapshot. It does not correct survivorship bias or reconstruct historical index membership. Verify renamed/reorganized companies and ticker histories before trading research conclusions are carried forward.
- Correlation, stationarity, executable trades and positive after-cost returns are separate questions.

Method references: [Statsmodels Engle-Granger documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html), [Statsmodels multiple-testing correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.fdrcorrection.html), and [yfinance download documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
