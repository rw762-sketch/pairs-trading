# Learning from published pairs-trading examples

Reviewed September 18, 2026. This note explains which ideas were adapted, what was checked, and what the experiment can establish. Online backtest returns are not performance targets for this project.

## Useful examples and the lesson to take

| Primary source | What it demonstrates | Application here |
|---|---|---|
| [Gatev, Goetzmann and Rouwenhorst: Pairs Trading](https://depot.som.yale.edu/icf/papers/fileuploads/2573/original/08-03.pdf), methodology sections 2.1-2.3 | A 12-month formation period followed by six months of trading, with monthly cohorts. Selection uses distances between normalized total-return histories. | Separate pair formation from subsequent trading. Our 252-close formation and monthly refresh are a proposed adaptation, not a reproduction of their distance strategy. |
| [QuantConnect: dynamic correlation and cointegration pairs](https://www.quantconnect.com/research/15347/intraday-dynamic-pairs-trading-using-correlation-and-cointegration-approach/), Method | Rolling histories and refreshed pair models; parameters associated with open positions; treatment of deselected pairs. | Reconsider eligibility using preceding data. Keep hedge ratios fixed within a trade and define when positions close. Our pilot closes at each monthly boundary. |
| [Statsmodels: rolling regression example](https://www.statsmodels.org/stable/examples/notebooks/generated/rolling_ls.html) | Regression estimates change as a historical window advances. Estimates belong to the window's endpoint. | Fit each month's beta using the preceding 252 closes; use it only for subsequent trading. The example itself uses factor returns, not a pairs-trading backtest. |
| [Statsmodels: multiple-testing correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html) | Methods including Holm adjust for testing a family of hypotheses. | Compare monthly raw p < 0.05 selection with a Holm-adjusted version over the predefined same-industry pair family. |
| [Scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) | Time-ordered evaluation keeps future observations out of earlier training. | Use only dates preceding a trading window for selection, regression and assumption checks. A fresh final holdout is still needed after development. |

## What not to copy blindly

The QuantConnect dynamic tutorial has an entry-direction inconsistency in its prose, and its displayed sizing code does not use the estimated hedge ratio. We keep our verified two-leg accounting rather than copying those fragments. Its intraday setting, universe and reported performance also differ from this daily stock experiment.

The [official QuantConnect cointegration notebook](https://github.com/QuantConnect/Research/blob/master/Analysis/05%20Pairs%20Trading%20Strategy%20Based%20on%20Cointegration.ipynb) fits and illustrates an XOM/CVX relationship over the same historical sample. That is useful for understanding regression residuals. A fitted illustration is not evidence of out-of-sample trading profit.

Gatev and coauthors also examine delayed execution and trading frictions. Our simulations retain a following-close execution delay, both-leg fees, short borrowing costs and idle cash. Their paper's historical results cannot establish that our current implementation is profitable.

## The implemented learning experiment

Use the existing 50-stock sample: ten sub-industries with five stocks each, originally sampled with seed 42 without using later returns. This creates 100 same-industry pair hypotheses. The sample and its current-membership metadata are fixed before this comparison.

Three configurations are specified before simulation:

1. **Fixed selection:** select pairs and beta once from the 252 closes preceding October 23, 2025. Keep that selection for the evaluation period.
2. **Monthly selection:** repeat the same screen and regression at each monthly window, using only the preceding 252 closes.
3. **Monthly selection + Holm:** use the same monthly procedure, with a conservative multiple-testing filter.

Common rules: entry z = +/-2, exit band +/-0.5, preceding 60-close normalization, positive beta, different issuers, provisional I(1) diagnostics, 20-trading-interval maximum holding, 10 bp per traded dollar in fees/slippage, and 2% annual short-borrow expense. Each account begins with $100,000; gains and losses carry forward.

All three close positions at each monthly window's end and allocate the next window's actual account value equally across eligible pairs. This matching controls for monthly liquidation and resizing when comparing selection methods. It differs from the 52-pair baseline's execution schedule, universe and formation window, so the two studies are not an isolated like-for-like comparison. The first and last calendar windows are partial.

No eligible pair means no trade and the account remains cash. A zero return from inactivity is not a profitable trading signal. Each entry executes one close after its signal; holding shares and beta stay fixed within the trade.

Monthly refresh changes both pair eligibility and hedge-ratio estimates. This experiment measures their combined effect; it does not isolate which of those two changes helps or hurts.

The Holm family includes all predefined same-industry hypotheses for that month. A pair that cannot be tested or fails the individual integration-assumption diagnostic is explicitly marked untested/ineligible; a value of one is used only as its conservative correction input. Its actual observed p-value is not fabricated. Corrections are monthly diagnostics under test assumptions, not a guarantee across every repeated monthly search.

## How to run and inspect it

```bash
source venv/bin/activate
python run_walk_forward.py
```

The new dated report includes method comparisons, account curves, monthly selections, all screening p-values, complete trade logs and profit for every selected pair. The experiment plan records inputs and rules before the simulation; it also records the source-file hashes.

Read `walk_forward.py` for selection and simulation; read `run_walk_forward.py` for reporting. The original eight-page 494-stock report remains the baseline.

## Measured result of the first run

The [September 18 experiment report](results/2026-09-18_21-43-23_online-learning-walk-forward_50-stocks/2026-09-18_21-43-23_online-learning-walk-forward_50-stocks_report.md) records these after-cost results for October 23, 2025 to September 17, 2026:

| Method | Net return | Net profit on initial $100,000 | Completed trades |
|---|---|---|---|
| Fixed pairs and hedge ratios | -9.18% | -$9,175.59 | 13 |
| Monthly reselection and refit | -2.98% | -$2,975.62 | 30 |
| Monthly refresh with Holm correction | 0.00% | $0.00 | 0 |

Monthly refresh lost less in this experiment, but it still lost money. The Holm version never selected a pair and stayed in cash. Its zero return does not demonstrate a trading edge. The rules were not changed after observing these results.

All 24 tests passed, including future-data perturbation, account/trade reconciliation, multiple-testing correction and cash-only periods. Tests verify implementation behavior; they do not establish future profitability.

## What remains uncertain

The evaluation period, October 23, 2025 to September 17, 2026, was already inspected during project development. This is a retrospective research comparison, not a fresh final test or a claim of optimal parameters. The 50-stock pilot cannot establish a result for all S&P 500 stocks.

Current constituent/industry metadata and the previously chosen sample retain survivorship and selection limitations. Adjusted closes, available borrowing, fractional shares and simplified execution remain assumptions. A monthly refresh can increase costs or make results worse; use the measured results instead of assuming it is an improvement.

Future work: test stability across multiple earlier windows, validate lookback/stop-loss choices using chronological development folds, cap overlapping stock exposure, and reserve genuinely new observations for final evaluation. Those extensions are not claimed as completed by this pilot.
