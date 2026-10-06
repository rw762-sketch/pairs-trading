# Clustered Pairs Trading Research

A Python research pipeline that reduced candidate pair comparisons by **87%**, automated stock screening and portfolio simulation, and made the resulting backtest reproducible from a fresh clone. It combines K-means grouping, cointegration and persistence checks, OLS spread signals, and explicit two-leg accounting.

The **official version is the weighted 3-of-6 strategy** in [selected_strategy.json](selected_strategy.json). Its saved 2025–2026 historical backtest produced **$3,900.81 simulated net profit on $100,000 initial capital (3.90%)**, with **2.27% maximum drawdown**, using three pairs and 22 trades.

| Impact | Measured outcome |
| --- | --- |
| Smaller screening workload | 124,251 possible pairs → 16,106 within-cluster tests; **87.04% fewer comparisons** |
| Full research workflow | 503-security universe → 499 formation-eligible stocks → 387 initial candidates → 3 retained pairs |
| Historical portfolio result | **3.90% net return**, 2.27% maximum drawdown, and 22 completed trades |
| Repeatable review | One-command frozen replay, saved trade records, **43 automated tests**, and GitHub CI |

The comparison reduction uses the same 499 eligible stocks in the official formation period. See the [screening summary](reports/screening-impact.json) for the counts and calculation.

![Official strategy cumulative return](reports/selected-2025-2026/cumulative-return.png)

## How the project creates value

- **Focused research:** K-means avoided 108,145 candidate comparisons while retaining every within-cluster pair for cointegration testing. Persistence and recovery filters narrowed 387 initial candidates to three portfolio pairs.
- **Automated decisions:** the pipeline turns adjusted price histories into pair selections, hedge ratios, budget weights, trading signals, and dated reports, giving experiments a consistent process.
- **Cost-aware evaluation:** two-leg accounting reconciles $5,783.54 gross P&L, $1,882.72 modeled fees and borrowing, and $3,900.81 net P&L, showing how implementation costs affect the portfolio result.
- **Reproducible evidence:** a fresh clone replays the official result with one command after setup. Hashed inputs, saved selection records, 43 tests, and CI support independent review of timing, sizing, costs, and allocation.

## Reproduce the official result

Use Python 3.13, the version used for local verification (the pinned dependencies require Python 3.12 or newer). Numerical dependency versions are recorded in [constraints-replay.txt](constraints-replay.txt). The frozen replay inputs are included in [reproducibility/selected](reproducibility/selected), so this command does not download market data.

```bash
git clone https://github.com/rw762-sketch/pairs-trading.git
cd pairs-trading
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py'
python main.py
```

The expected result is **3.900813% net return, 22 trades, and three selected pairs**. Each run writes a dated folder under `results/` with an HTML report, equity curve, allocation table, trade log, pair metrics, and performance table. Open `results/index.html` for the latest local report.

The default command replays the frozen selection evidence and recomputes recovery filters, weights, signals, trades, and returns. It does not repeat the full-universe cointegration screen. Input hashes prevent silently substituting a revised price snapshot.

## Official strategy

1. Use a 503-security S&P 500 universe snapshot and Yahoo Finance adjusted daily closes. Cluster stocks into 10 groups using standardized training-period log returns.
2. Fit every within-cluster pair on the final 252 development closes. Require raw cointegration p-value below 0.05, a positive OLS hedge ratio, compatible integration diagnostics, and a finite recovery estimate.
3. Require p-value below 0.05 in **at least three of six overlapping 252-close development formations, including the latest**. These periods check historical stability.
4. Retain pairs with estimated half-life of 1–20 closes and at least 12 annualized spread mean crossings. Prevent a stock appearing in more than one retained pair.
5. Assign each pair a budget proportional to `periods_passed / mean_p_value`, then normalize the budgets to 100%. Average p-values include all six periods; missing tests count as p=1. This is a heuristic allocation rule, not an estimated probability of profit.
6. Trade the fixed OLS spread using its preceding 60-close mean and standard deviation. Enter when `2 < |z| < 3.5`, exit on reversion to the ±0.5 boundary, and stop on an adverse move to ±3.5. Execute at the next close. There is no holding-time limit; open positions close at the evaluation period's end.

The 20-close **half-life screening bound** is separate from a trade holding limit. All three pairs receive a fixed budget; an inactive pair's allocation stays in cash.

| Pair | Significant periods | Assigned budget |
| --- | ---: | ---: |
| MCO–SPGI | 3/6 | 41.54% |
| EVRG–WELL | 4/6 | 40.44% |
| ITW–NXPI | 3/6 | 18.01% |

## Recorded performance

Evaluation dates: **September 17, 2025–September 17, 2026**. Initial capital: **$100,000**.

| Metric | Official result |
| --- | ---: |
| Net portfolio return | 3.90% |
| Gross P&L / initial capital | 5.78% |
| Annualized return | 3.90% |
| Annualized volatility | 3.87% |
| Sharpe ratio, zero risk-free rate | 1.01 |
| Maximum drawdown | 2.27% |
| Completed trades / win rate | 22 / 59.09% |
| Modeled fees + borrowing | $1,882.72 |

Fees are 0.10% of executed gross notional on each entry and exit. Short borrowing is modeled at 2% annually using calendar-day accrual. Cash earns zero interest. A positive zero-risk-free Sharpe does not establish that the strategy beats a cash benchmark.

An earlier-year application of the same rules returned **1.00% net in 2021** with six pairs. A separate ten-year-data preview used eight years of development and two years of evaluation; it is **not the official version**. See [research results](docs/research-results.md) for these comparisons, cost decompositions, and concentration limits.

## Explore the research pipeline

For a new download and a complete screen, explicitly choose an experimental mode:

```bash
python main.py --mode walk-forward
python main.py --mode holdout
python run_historical_year.py --year 2021
```

`walk-forward` re-forms and trades successive periods. `holdout` runs a chronological split within the development data. Both modes reserve the final 252 closes by default; they are separate experiments and do not recreate the official 3-of-6 result. Downloads may differ from the frozen snapshot as the provider revises adjusted history.

## Code map

| Component | Responsibility |
| --- | --- |
| [data_fetcher.py](data_fetcher.py), [data/README.md](data/README.md) | Download/cache prices, determine training eligibility, reserve dates |
| [stock_clustering.py](stock_clustering.py), [cointegration.py](cointegration.py) | K-means candidate groups and every within-cluster test |
| [selection.py](selection.py), [run_persistence.py](run_persistence.py) | Recovery filters and repeated formation-period checks |
| [allocation.py](allocation.py) | Persistence and mean-p-value budget weights |
| [signal_generation.py](signal_generation.py), [backtester.py](backtester.py) | Causal spread signals, fills, positions, costs, and accounting |
| [evaluation.py](evaluation.py), [run_selected.py](run_selected.py) | Portfolio evaluation and the pinned official replay |
| [tests](tests) | Accounting, chronology, selection, and reproducibility checks |

## Research extensions

The project provides an inspectable research workflow with reconciled accounting. Further experiments can compare allocation rules, transaction-cost assumptions, and performance across additional market regimes.

Detailed definitions and assumptions are in [methodology](docs/methodology.md). Earlier implementations remain accessible through Git history; the root code and README describe the current official strategy.
