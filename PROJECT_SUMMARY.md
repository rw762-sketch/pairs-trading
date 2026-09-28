# Pairs trading project: current results

Updated September 18, 2026. Start here:

- **Learn the method:** [learning book PDF](output/pdf/2026-09-18_quant_project_learning_book.pdf) or [Markdown](output/pdf/2026-09-18_quant_project_learning_book.md). The 25-page book has 14 chapters, 62 glossary entries, 12 worked exercises and a four-week study plan.
- **Understand the latest results:** [project brief PDF](output/pdf/2026-09-18_23-12-20_pairs_trading_project_brief.pdf) or [Markdown](output/pdf/2026-09-18_23-12-20_pairs_trading_project_brief.md). This eight-page brief explains the experiments, charts and additional regression strategy.
- **Inspect the detailed assignment outputs:** [clean research report](results/2026-09-18_21-29-27_pairs-trading-clean-report_494-stocks/2026-09-18_21-29-27_pairs-trading-clean-report_494-stocks_report.md), its [PDF](results/2026-09-18_21-29-27_pairs-trading-clean-report_494-stocks/2026-09-18_21-29-27_pairs-trading-clean-report_494-stocks_report.pdf), and the [dated report catalog](REPORTS.md).

## Report contents

The detailed clean research report contains all 11 outputs in the assignment screenshot: candidate pairs and p-values, a readable table covering all 52 pairs, selected spreads, z-scores with entry/exit thresholds, equity curves, complete trade logs, training/later performance, risk and return metrics, win rate/average trade return/turnover, threshold sensitivity, and cumulative portfolio returns.

Pages 3-4 show after-cost profit for every pair on its actual $1,923.08 allocation. Detailed records are in the report folder's `data/` directory and charts in `figures/`.

It also answers the six research questions about correlation, stability, thresholds, costs, trade concentration and constraints. Completing these outputs does not establish that the strategy is profitable or ready for live trading.

The original PDF's full parameter-optimization requirement is **not finished**. The newer experiments test some bounded changes, but do not complete independent validation-based optimization across cointegration cutoffs, hedge-ratio lookbacks, stops, holding periods and costs. Page 8 of the detailed report identifies the original outstanding work.

## Current data and results

| Item | Result |
|---|---|
| Requested current S&P 500 securities | 503 |
| Retained complete histories | 494 |
| Training window | 2023-09-19 to 2025-10-22; 526 closes |
| Later window | 2025-10-23 to 2026-09-17; 226 closes |
| Original pair combinations | 121,771 |
| Saved raw candidates | 5,734 |
| Selected strategy pairs | 52 |
| Later net total return | -3.62% |
| Later final account | $96,382.91 from $100,000 |
| Later Sharpe / maximum drawdown | -2.34 / -3.87% |
| Later completed pair trades | 246 |

Selection requires a positive beta, the same sub-industry, different issuers, and provisional individual I(1) diagnostics, in addition to the original raw p-value filter. All 52 selected pairs receive equal allocations. The smallest raw p-value and the most raw partners are not rankings of trading quality. No saved pair passes the conservative Bonferroni threshold for the original full search.

## Learning experiment: refresh pairs monthly

The [online learning guide](ONLINE_LEARNING.md) connects published examples to a separate [50-stock comparison](results/2026-09-18_21-43-23_online-learning-walk-forward_50-stocks/2026-09-18_21-43-23_online-learning-walk-forward_50-stocks_report.md). Each monthly screen uses the preceding 252 closes, followed by subsequent trading. All three methods use identical monthly liquidation, trading thresholds and cost assumptions.

| Method | Net return, 2025-10-23 to 2026-09-17 | Completed trades |
|---|---|---|
| Keep initial pairs and hedge ratios | -9.18% | 13 |
| Refresh pairs and hedge ratios monthly | -2.98% | 30 |
| Monthly refresh with Holm multiple-testing correction | 0.00%; stayed in cash | 0 |

Monthly refresh reduced the loss by 6.20 percentage points in this pilot, but remained unprofitable. It changes both eligibility and beta estimates, so their individual effects are not isolated. The stricter filter found no qualifying pairs; inactivity does not establish a profitable signal. This comparison uses a different universe, formation window and liquidation schedule from the 52-pair baseline above. The evaluation period was already inspected; this is not a fresh final holdout or completed parameter optimization.

The dated experiment report includes a readable comparison, account chart, monthly selections, screening p-values, complete trade logs and profit for every selected pair. Run it with `./venv/bin/python run_walk_forward.py`.

## OLS comparison: improve trading decisions

[OLS_STRATEGY_PLAN.md](OLS_STRATEGY_PLAN.md) specifies five versions: the monthly baseline, stable relationship checks, entry confirmation with a cost hurdle, exposure/exit limits, and their combination. The hedge ratio remains an ordinary least-squares regression slope.

The runner uses two earlier windows (2024-09-19 to 2025-03-31 and 2025-04-01 to 2025-10-22) to choose a candidate, records that decision, and then measures all five in the later window. A candidate must earn positive net returns in both earlier windows with sufficient completed trades. If none qualify, the decision is cash. This is a research gate, not proof of profitability; the later period was already inspected.

Run `./venv/bin/python run_ols_improvement.py`. [REPORTS.md](REPORTS.md) links completed dated comparisons, full trade logs and per-pair profits.

The [first completed OLS comparison](results/2026-09-18_22-52-31_ols-strategy-improvement_50-stocks/2026-09-18_22-52-31_ols-strategy-improvement_50-stocks_report.md) found no positive later-period variant. Relationship checks reduced the later loss to -1.46%; entry checks returned -1.96%; risk limits -1.57%; all changes -1.51%, versus the monthly baseline's -2.98%. Earlier-window selection chose the baseline, which then lost money later. No new version is established as a profitable strategy or automatically replaces the baseline.

## Additional strategy: forecast spread recovery with OLS

The [linear error-correction experiment](results/2026-09-18_23-11-02_linear-ecm-forecast-gate_50-stocks/2026-09-18_23-11-02_linear-ecm-forecast-gate_50-stocks_report.md) retains the OLS hedge regression and adds a second linear regression forecasting changes in its residual. An entry gate compares the forecast spread movement **after the next-close entry** with estimated trading and borrowing costs. It preserves the baseline's exits and allocation rules.

In the already-examined later period, 2025-10-23 to 2026-09-17, the additional gate returned **-3.056469% with 29 trades**, versus **-2.975623% with 30 trades** for the monthly baseline. Selection using the two earlier windows chose the baseline. The additional gate did not improve later total return; neither strategy demonstrated profitability, and this period is not an untouched holdout.

The [dated experiment folder](results/2026-09-18_23-11-02_linear-ecm-forecast-gate_50-stocks/) contains the frozen selection, account records, forecasts, trade logs and per-pair profits. Run `./venv/bin/python run_linear_ecm.py` to create another dated run from the cached data.

## How to regenerate

Activate the project environment in your terminal:

```bash
cd "/Users/wuruiyue/Pair trading project"
source venv/bin/activate
```

This only changes Python for that terminal session. `deactivate` leaves it. Your separate class environment is unaffected.

To regenerate the report without rerunning the calculations:

```bash
python clean_report.py
```

To rerun the calculations and produce the same clean report format (activation is optional):

```bash
./venv/bin/python main.py --report
```

Equivalent: `./venv/bin/python build_research_report.py`. This reuses the saved screen and prices, creates a new dated directory and updates REPORTS.md. It does not download a fresh market snapshot. `--no-pdf` produces the Markdown report, plots and CSVs without the PDF. `--help` lists input overrides.

## What changed in the calculation

The backtester now accounts for shares in both legs, actual cash, next-close execution, entry and exit fees, short borrowing and final liquidation. It exports one row per completed round trip. Equity and trade amounts reconcile to a single allocated account.

Baseline: entry z-score 2, exit band 0.5, preceding 60-close rolling statistics, a 20-trading-interval holding cap, 10 basis points per traded dollar in combined fees/slippage, and 2% annual borrow expense. These cost inputs are assumptions.

All 47 tests pass: 15 accounting/signal tests, nine original chronological-selection tests, 12 tests of the OLS improvements, three checks of selection using earlier windows only, and eight additional error-correction tests. Each report run also checks trade profit against account equity, fees, delayed entries, holding limits and final liquidation. The older 16:22 and 17:10 backtest reports used the earlier accounting and should not be used to judge performance. The corrected 17:58 source and this clean report reconcile.

## Limits

Training performance is a fitted diagnostic. The later period has already been inspected during development, so it is not an untouched final holdout. Current membership and full-period data availability introduce selection bias. Adjusted closes, fractional shares and assumed borrowing simplify execution; margin, liquidity limits and market impact are not modeled. Sensitivity plots describe the tested historical periods and do not validate an optimized live strategy.

The legacy `python main.py` download-and-screen path remains available but does not produce all the extended research outputs; use `--report` for this complete saved-data report. `parameter_optimization.py` is an older experimental helper with unfinished methods; this report runs its own explicit sensitivity grid and does not use its optimizer.
