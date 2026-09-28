# Quick reference

## Open the results

- **Learning book:** [PDF](output/pdf/2026-09-18_quant_project_learning_book.pdf) or [Markdown](output/pdf/2026-09-18_quant_project_learning_book.md) — 25 pages, 14 chapters, 62 glossary entries, 12 worked exercises and a four-week study plan.
- **Latest project brief:** [PDF](output/pdf/2026-09-18_23-12-20_pairs_trading_project_brief.pdf) or [Markdown](output/pdf/2026-09-18_23-12-20_pairs_trading_project_brief.md) — eight pages explaining results and the additional regression strategy.
- **Detailed records:** [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) links the assignment report; [REPORTS.md](REPORTS.md) lists dated reports and charts.

## Use the project environment

From this project folder:

```bash
source venv/bin/activate
python main.py --report
```

Or skip activation and use the environment directly:

```bash
./venv/bin/python main.py --report
```

The report reuses the existing screen and cached prices. Each run gets a new date/time/name folder. The clean PDF includes all selected pairs' profits; complete CSVs are under `data/` and figures under `figures/`. To leave the activated environment, type `deactivate`. This does not remove packages or change your separate class environment.

Install dependencies only into the project environment when needed:

```bash
./venv/bin/python -m pip install -r requirements.txt
```

## Commands

| Purpose | Command |
|---|---|
| Clean the existing results into a new dated report | `./venv/bin/python clean_report.py` |
| Recalculate saved data and create the clean PDF | `./venv/bin/python main.py --report` |
| Same report without PDF | `./venv/bin/python main.py --report --no-pdf` |
| Report input options | `./venv/bin/python main.py --report --help` |
| Compare fixed versus monthly pair selection on 50 saved stocks | `./venv/bin/python run_walk_forward.py` |
| Test five improvements while retaining linear regression | `./venv/bin/python run_ols_improvement.py` |
| Compare the baseline with an OLS spread-forecast entry gate | `./venv/bin/python run_linear_ecm.py` |
| All accounting and chronological-selection tests | `./venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v` |
| Numerical accounting tests, no network | `./venv/bin/python -m unittest discover -s tests -p test_backtester.py -v` |
| Fresh full-universe legacy download and screen | `./venv/bin/python main.py` |

The full-universe screen can take much longer than rebuilding the report. Old `num_stocks` settings are not used to limit the full-universe fetch. To screen the saved 50-stock sample explicitly:

```bash
./venv/bin/python screen_pairs.py --universe output/pair-research-50/universe.json --output output/pair-research-50 --name peer-correlation
```

The monthly-selection experiment is separate from the full-universe baseline. [ONLINE_LEARNING.md](ONLINE_LEARNING.md) explains the online examples, rules and limitations; [REPORTS.md](REPORTS.md) links its dated results and per-pair profits.

[OLS_STRATEGY_PLAN.md](OLS_STRATEGY_PLAN.md) explains the five-version comparison: keep linear regression, test pair stability, better entries and risk limits, and choose a candidate using only the two earlier windows. Any later-period winner remains exploratory.

The [additional OLS error-correction strategy](results/2026-09-18_23-11-02_linear-ecm-forecast-gate_50-stocks/2026-09-18_23-11-02_linear-ecm-forecast-gate_50-stocks_report.md) gates entries using forecast spread movement after execution relative to costs. It returned -3.056469% with 29 trades versus the baseline's -2.975623% with 30 trades in the already-examined later period. Earlier-window selection chose the baseline; no profitability was demonstrated. The [dated folder](results/2026-09-18_23-11-02_linear-ecm-forecast-gate_50-stocks/) holds the full records.

All 47 tests pass, including eight new error-correction tests. These check software behavior; they do not establish a trading edge.

## Read the statistics

- Cointegration: two individually wandering price series have a stationary linear combination. A small test p-value is evidence under assumptions, not a profit probability.
- Correlation: closeness of daily return movements; it does not imply cointegration.
- Beta: slope in the fitted model `price A = alpha + beta x price B + residual`; alpha is the intercept. The project's trading spread is `price A - beta x price B`, so it equals `alpha + residual`. Subtracting a fixed intercept from both spread and rolling mean leaves the z-score unchanged.
- Z-score: today's spread minus the previous 60 closes' average, divided by their standard deviation.
- Sharpe: daily mean return divided by daily volatility, annualized; this report assumes zero risk-free rate.
- Turnover: annual gross traded dollars divided by starting capital. Both legs and both entry/exit count.
- Win rate: fraction of complete trades with positive profit after costs. A high win rate can coexist with losses.

No fixed return or Sharpe target is guaranteed. The current corrected later-period result is negative; see the report for the dates and assumptions.
