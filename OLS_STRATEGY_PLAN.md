# Improve the trading rules while keeping linear regression

The objective is positive profit after costs across several periods. There is no rule that guarantees it. This experiment keeps ordinary least squares (OLS) and tests a small, fixed set of changes around it.

## The model stays simple

Each month estimate:

**Price A = alpha + beta x Price B + residual**

Use the preceding 252 daily closes. Beta is the hedge ratio: for each share of A, use beta shares of B on the opposite side. A positive-beta pair can therefore be traded long A / short B or short A / long B. Alpha is a fitted intercept; subtracting a fixed intercept does not change the rolling z-score.

The spread is Price A - beta x Price B. Compare today's spread with its preceding 60-close average and standard deviation. A z-score of +2 means it is two historical standard deviations above that average. This describes a deviation; it does not prove the spread will recover.

## The three proposed improvements

| Change | Exact rule being tested | Why it might help |
|---|---|---|
| Trust more stable relationships | Both half-window regression slopes must be positive and within 50% of the full-window slope. Fit a second linear regression of the residual on its previous value; require an estimated half-life of 2-20 closes. | Avoid hedge ratios that changed sharply and spreads whose estimated recovery is too slow for the holding period. |
| Wait for a better entry | Require current absolute z-score above 2 but declining from the previous close, with the same sign. The potential move toward the 0.5 exit band must exceed twice estimated round-trip costs. | Avoid entering solely because a deviation keeps widening, and avoid opportunities too small relative to costs. |
| Limit damage and concentration | Exit after an adverse z-score reaches 3.5 or after 10 trading intervals. Do not hold two selected pairs containing the same stock. Allocate at most 20% of monthly starting capital to one pair. | Reduce concentration and time spent in unsuccessful trades. These rules can also cut off trades that would eventually recover. |

The second regression is **residual today = constant + phi x residual yesterday + error**. When 0 < phi < 1, estimated half-life is log(0.5) / log(phi): the model's time for an expected deviation to halve. This is a noisy historical estimate, not a promised recovery time or a separate statistical proof of cointegration.

The cost comparison uses potential convergence distance, not predicted profit. It includes a fee/slippage estimate of 0.1% per traded dollar at entry and exit, plus estimated borrowing for 28 calendar days at 2% annualized on the short leg. Actual backtest borrowing uses actual elapsed calendar days.

Signals, including stops, execute at the next close. Overnight gaps can make a stop lose more than expected. Beta and shares stay fixed within each trade. The allocation cap controls entry size; market moves can change the exposure afterward. Unused capital remains cash with no interest.

## Five versions, fixed before measuring returns

1. **Monthly OLS baseline:** the existing raw-p-value strategy, entry 2, exit 0.5, maximum holding 20 trading intervals.
2. **Stable beta + recovery speed:** baseline plus the relationship checks.
3. **Entry confirmation + cost hurdle:** baseline plus the entry checks.
4. **Stops + allocation limits:** baseline plus the risk rules.
5. **All three changes:** combine the checks and risk rules.

All versions use the same saved 50 stocks and monthly schedule. Candidate pairs must have the same sub-industry, different issuers, provisional individual I(1) compatibility, positive beta and raw Engle-Granger p < 0.05. These are exploratory candidates; screening many pairs creates false-positive risk. The separate Holm-adjusted experiment remains in the project and did not find qualifying trades. Economic filters are not a replacement for multiple-testing correction.

Risk versions select non-overlapping pairs in order of lowest formation p-value, then pair name. Their pair weight is min(1 / selected pair count, 20%). Other versions allocate equally to all eligible pairs. Each closes positions at monthly boundaries before the next refit. Reduced losses under a cash-heavy strategy are not automatically evidence of better predictions.

## How a candidate is chosen

| Stage | Dates | Purpose |
|---|---|---|
| Initial formation history | 2023-09-19 to 2024-09-18 | Supplies the first 252 historical closes; later formation windows roll forward. |
| Earlier testing window 1 | 2024-09-19 to 2025-03-31 | Test each fixed set of rules. |
| Earlier testing window 2 | 2025-04-01 to 2025-10-22 | Check whether the result also holds in a second period. |
| Later comparison | 2025-10-23 to 2026-09-17 | Report results after freezing the earlier-window choice. This period has already been examined. |

Each testing window begins with $100,000; profits and losses carry between months within that window. The separate windows reset capital and positions. A candidate must have positive net returns in both earlier windows, at least three completed trades in each, and ten total. Among eligible versions, choose the highest worse-window return, then mean return, then variant name. If none qualify, the recorded decision is cash and more research.

This small-sample gate is a research rule, not a test of statistical significance. The later period already influenced our research ideas. Recording the choice before this simulation does not make that period an untouched holdout. Current stock membership and previous sample availability filters also introduce historical selection bias.

## Run and inspect

```bash
source venv/bin/activate
python run_ols_improvement.py
```

Every run saves its plan and input/code hashes first, runs the two earlier windows, saves the selected rule, then runs the later comparison. Its dated report includes all five results, risk statistics, an account chart and each selected pair's profit. [REPORTS.md](REPORTS.md) links completed reports.

The implementation is in [ols_improvement.py](ols_improvement.py); the evaluation and report are in [run_ols_improvement.py](run_ols_improvement.py). No existing strategy is automatically replaced with the best-looking later result.

## First measured result

[September 18 comparison](results/2026-09-18_22-52-31_ols-strategy-improvement_50-stocks/2026-09-18_22-52-31_ols-strategy-improvement_50-stocks_report.md), after fees and borrowing:

| Version | Earlier window 1 | Earlier window 2 | Later comparison |
|---|---|---|---|
| Monthly OLS baseline | +1.31% | +1.65% | -2.98% |
| Stable beta + recovery speed | +0.90% | +1.13% | -1.46% |
| Entry confirmation + cost hurdle | +0.50% | +0.44% | -1.96% |
| Stops + allocation limits | -0.31% | +0.43% | -1.57% |
| All three changes | +0.52% | -0.10% | -1.51% |

The earlier-window rule selected the baseline, which subsequently lost money in the later comparison. The relationship filter lost less later, but choosing it now because of that result would use the later period for selection. None of the five was profitable in the later period. The combined version's maximum later drawdown was -1.62%, compared with -4.28% for the baseline; smaller exposure contributes to this risk reduction.

All 39 project tests passed, including accounting, causal selection, the new entry/risk rules and isolation of the earlier-window decision. These checks verify implementation, not future profits.

## Sources and next research step

The [OLS documentation](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html) describes the retained regression; [Engle-Granger documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html) states the integration assumption and test interpretation. [Chronological validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) separates earlier training from later evaluation. [Kitapbayev and Leung](https://arxiv.org/abs/1707.03498) study trading times with costs and finite deadlines. Our rules are practical hypotheses, not a reproduction of that paper's optimal stopping solution.

After this experiment, the useful next step is more independent evidence: longer histories, historical constituent membership, several chronological windows, and forward paper trading with frozen rules. Keep the regression and judge the trading decisions by actual after-cost results.
