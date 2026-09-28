# Report index

## Separate multiple-regression experiment

[Five-predictor entry filter and baseline comparison](results/2026-09-19_00-01-55_multiple-regression_50-stocks/2026-09-19_00-01-55_multiple-regression_50-stocks_report.md)

The original OLS and linear ECM models and reports are retained. Run `./venv/bin/python run_multiple_regression.py` to reproduce this separate experiment. The baseline remained the earlier-fold selection; the later period is reused research data.

## Learning and presentation

- [Quant project learning book](output/pdf/2026-09-18_quant_project_learning_book.pdf) | [Editable Markdown](output/pdf/2026-09-18_quant_project_learning_book.md)
- [Project presentation brief](output/pdf/2026-09-18_23-12-20_pairs_trading_project_brief.pdf) | [Editable Markdown](output/pdf/2026-09-18_23-12-20_pairs_trading_project_brief.md)

The learning book explains the concepts, calculations, procedure and vocabulary. The brief presents measured results, the additional forecast strategy and assignment coverage.

## Current clean report

[2026-09-18_21-29-27_pairs-trading-clean-report_494-stocks](results/2026-09-18_21-29-27_pairs-trading-clean-report_494-stocks/START_HERE.md)

Strategy, required charts and statistics, and profit for every selected pair. Full records are in its data folder; older runs are grouped below.

- [Read the PDF](results/2026-09-18_21-29-27_pairs-trading-clean-report_494-stocks/2026-09-18_21-29-27_pairs-trading-clean-report_494-stocks_report.pdf)
- [Read the Markdown report](results/2026-09-18_21-29-27_pairs-trading-clean-report_494-stocks/2026-09-18_21-29-27_pairs-trading-clean-report_494-stocks_report.md)
- [Profit and statistics for every pair](results/2026-09-18_21-29-27_pairs-trading-clean-report_494-stocks/data/2026-09-18_21-29-27_pairs-trading-clean-report_494-stocks_pair-profits-and-statistics.csv)
- [Complete later-period trade log](results/2026-09-18_21-29-27_pairs-trading-clean-report_494-stocks/data/2026-09-18_21-29-27_pairs-trading-clean-report_494-stocks_later-trade-log.csv)

## Learning experiment

[Rolling pair selection comparison](results/2026-09-18_21-43-23_online-learning-walk-forward_50-stocks/2026-09-18_21-43-23_online-learning-walk-forward_50-stocks_report.md)

A separate pilot with fixed selection, monthly selection and a stricter multiple-testing filter. See [ONLINE_LEARNING.md](ONLINE_LEARNING.md) for the primary examples and adaptations.


## OLS strategy improvement

[Five strategy variants and earlier-window selection](results/2026-09-18_22-52-31_ols-strategy-improvement_50-stocks/2026-09-18_22-52-31_ols-strategy-improvement_50-stocks_report.md)

Keeps linear regression and tests pair stability, entry checks and allocation limits. See [OLS_STRATEGY_PLAN.md](OLS_STRATEGY_PLAN.md) for the exact rules.


## Linear spread forecast experiment

[Forecast entry gate versus the monthly baseline](results/2026-09-18_23-11-02_linear-ecm-forecast-gate_50-stocks/2026-09-18_23-11-02_linear-ecm-forecast-gate_50-stocks_report.md)

Keeps the OLS hedge ratio and tests whether a second linear regression can filter entries using predicted spread recovery and a cost hurdle. The report includes the frozen earlier-window selection, complete trade logs and pair profits.


<details>
<summary>Earlier runs and underlying research files</summary>

Names use the run date and time in America/New_York, report name, and number of stocks used. The historical data period is recorded inside each report or its metadata.

New runs get separate folders. A numeric suffix distinguishes runs started in the same second. Recovered older reports identify their timestamp source in archive metadata.

## 2026-09-18_23-26-04_pairs-trading-backtest_494-stocks

- [a-vtrs-drawdown.png](results/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks_a-vtrs-drawdown.png)
- [a-vtrs-equity.png](results/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks_a-vtrs-equity.png)
- [aapl-cohr-drawdown.png](results/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks_aapl-cohr-drawdown.png)
- [aapl-cohr-equity.png](results/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks_aapl-cohr-equity.png)
- [aapl-incy-drawdown.png](results/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks_aapl-incy-drawdown.png)
- [aapl-incy-equity.png](results/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks_aapl-incy-equity.png)
- [backtest-report.txt](results/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks_backtest-report.txt)
- [backtest-summary.csv](results/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks_backtest-summary.csv)
- [cointegrated-pairs.csv](results/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks_cointegrated-pairs.csv)
- [cointegration-heatmap.png](results/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks_cointegration-heatmap.png)
- [portfolio-equity.png](results/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks_portfolio-equity.png)
- [portfolio-metrics.csv](results/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks_portfolio-metrics.csv)
- [run-metadata.json](results/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks/2026-09-18_23-26-04_pairs-trading-backtest_494-stocks_run-metadata.json)

## 2026-09-18_22-03-47_pairs-trading-backtest_494-stocks

- [a-vtrs-drawdown.png](results/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks_a-vtrs-drawdown.png)
- [a-vtrs-equity.png](results/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks_a-vtrs-equity.png)
- [aapl-cohr-drawdown.png](results/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks_aapl-cohr-drawdown.png)
- [aapl-cohr-equity.png](results/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks_aapl-cohr-equity.png)
- [aapl-incy-drawdown.png](results/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks_aapl-incy-drawdown.png)
- [aapl-incy-equity.png](results/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks_aapl-incy-equity.png)
- [backtest-report.txt](results/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks_backtest-report.txt)
- [backtest-summary.csv](results/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks_backtest-summary.csv)
- [cointegrated-pairs.csv](results/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks_cointegrated-pairs.csv)
- [cointegration-heatmap.png](results/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks_cointegration-heatmap.png)
- [portfolio-equity.png](results/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks_portfolio-equity.png)
- [portfolio-metrics.csv](results/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks_portfolio-metrics.csv)
- [run-metadata.json](results/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks/2026-09-18_22-03-47_pairs-trading-backtest_494-stocks_run-metadata.json)

## 2026-09-18_17-58-16_complete-research-report_494-stocks

- [aep-duk-later-signals.csv](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_aep-duk-later-signals.csv)
- [aep-duk-spread-and-zscore.png](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_aep-duk-spread-and-zscore.png)
- [all-candidate-p-values.csv](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_all-candidate-p-values.csv)
- [all-stock-rankings.csv](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_all-stock-rankings.csv)
- [backtest-equity.png](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_backtest-equity.png)
- [candidate-heatmap.png](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_candidate-heatmap.png)
- [cost-and-constraint-sensitivity.csv](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_cost-and-constraint-sensitivity.csv)
- [cpt-ess-later-signals.csv](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_cpt-ess-later-signals.csv)
- [cpt-ess-spread-and-zscore.png](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_cpt-ess-spread-and-zscore.png)
- [es-fe-later-signals.csv](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_es-fe-later-signals.csv)
- [es-fe-spread-and-zscore.png](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_es-fe-spread-and-zscore.png)
- [later-daily-portfolio.csv](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_later-daily-portfolio.csv)
- [later-trade-log.csv](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_later-trade-log.csv)
- [pair-performance.csv](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_pair-performance.csv)
- [pair-profit-contributions.csv](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_pair-profit-contributions.csv)
- [pair-profit-contributions.png](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_pair-profit-contributions.png)
- [performance-comparison.csv](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_performance-comparison.csv)
- [portfolio-cumulative-returns.png](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_portfolio-cumulative-returns.png)
- [raw-partner-counts.png](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_raw-partner-counts.png)
- [research-report.md](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_research-report.md)
- [research-report.pdf](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_research-report.pdf)
- [run-metadata.json](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_run-metadata.json)
- [selected-pairs-statistics.csv](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_selected-pairs-statistics.csv)
- [stock-integration-diagnostics.csv](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_stock-integration-diagnostics.csv)
- [threshold-sensitivity.csv](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_threshold-sensitivity.csv)
- [threshold-sensitivity.png](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_threshold-sensitivity.png)
- [training-daily-portfolio.csv](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_training-daily-portfolio.csv)
- [training-trade-log.csv](results/2026-09-18_17-58-16_complete-research-report_494-stocks/2026-09-18_17-58-16_complete-research-report_494-stocks_training-trade-log.csv)

## 2026-09-18_17-10-39_pairs-trading-backtest_494-stocks

- [a-vtrs-drawdown.png](results/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks_a-vtrs-drawdown.png)
- [a-vtrs-equity.png](results/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks_a-vtrs-equity.png)
- [aapl-cohr-drawdown.png](results/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks_aapl-cohr-drawdown.png)
- [aapl-cohr-equity.png](results/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks_aapl-cohr-equity.png)
- [aapl-incy-drawdown.png](results/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks_aapl-incy-drawdown.png)
- [aapl-incy-equity.png](results/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks_aapl-incy-equity.png)
- [backtest-report.txt](results/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks_backtest-report.txt)
- [backtest-summary.csv](results/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks_backtest-summary.csv)
- [cointegrated-pairs.csv](results/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks_cointegrated-pairs.csv)
- [cointegration-heatmap.png](results/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks_cointegration-heatmap.png)
- [portfolio-equity.png](results/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks_portfolio-equity.png)
- [portfolio-metrics.csv](results/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks_portfolio-metrics.csv)
- [run-metadata.json](results/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks_run-metadata.json)

## 2026-09-18_16-34-09_correlation-screen_50-stocks

- [report.md](output/pair-research-50/2026-09-18_16-34-09_correlation-screen_50-stocks/2026-09-18_16-34-09_correlation-screen_50-stocks_report.md)
- [results.json](output/pair-research-50/2026-09-18_16-34-09_correlation-screen_50-stocks/2026-09-18_16-34-09_correlation-screen_50-stocks_results.json)
- [universe.json](output/pair-research-50/2026-09-18_16-34-09_correlation-screen_50-stocks/2026-09-18_16-34-09_correlation-screen_50-stocks_universe.json)

## 2026-09-18_16-22-42_pairs-trading-backtest_494-stocks

- [run-metadata.json](results/2026-09-18_16-22-42_pairs-trading-backtest_494-stocks/2026-09-18_16-22-42_pairs-trading-backtest_494-stocks_run-metadata.json)

## 2026-09-18_16-18-44_correlation-screen_50-stocks

- [archive-metadata.json](output/pair-research-50/2026-09-18_16-18-44_correlation-screen_50-stocks/2026-09-18_16-18-44_correlation-screen_50-stocks_archive-metadata.json)
- [report.md](output/pair-research-50/2026-09-18_16-18-44_correlation-screen_50-stocks/2026-09-18_16-18-44_correlation-screen_50-stocks_report.md)
- [results.json](output/pair-research-50/2026-09-18_16-18-44_correlation-screen_50-stocks/2026-09-18_16-18-44_correlation-screen_50-stocks_results.json)
- [run-timing.json](output/pair-research-50/2026-09-18_16-18-44_correlation-screen_50-stocks/2026-09-18_16-18-44_correlation-screen_50-stocks_run-timing.json)
- [sample-details.md](output/pair-research-50/2026-09-18_16-18-44_correlation-screen_50-stocks/2026-09-18_16-18-44_correlation-screen_50-stocks_sample-details.md)

## 2026-09-18_16-14-59_correlation-screen_495-stocks

- [archive-metadata.json](output/pair-research/2026-09-18_16-14-59_correlation-screen_495-stocks/2026-09-18_16-14-59_correlation-screen_495-stocks_archive-metadata.json)
- [report.md](output/pair-research/2026-09-18_16-14-59_correlation-screen_495-stocks/2026-09-18_16-14-59_correlation-screen_495-stocks_report.md)
- [results.json](output/pair-research/2026-09-18_16-14-59_correlation-screen_495-stocks/2026-09-18_16-14-59_correlation-screen_495-stocks_results.json)
- [strategies.md](output/pair-research/2026-09-18_16-14-59_correlation-screen_495-stocks/2026-09-18_16-14-59_correlation-screen_495-stocks_strategies.md)

## 2026-09-18_15-31-49_sp500-stock-universe_503-stocks

- [archive-metadata.json](output/stock-universe/2026-09-18_15-31-49_sp500-stock-universe_503-stocks/2026-09-18_15-31-49_sp500-stock-universe_503-stocks_archive-metadata.json)
- [stock-table.md](output/stock-universe/2026-09-18_15-31-49_sp500-stock-universe_503-stocks/2026-09-18_15-31-49_sp500-stock-universe_503-stocks_stock-table.md)

## 2026-09-18_15-28-37_pairs-trading-backtest_494-stocks

- [a-vtrs-drawdown.png](results/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks_a-vtrs-drawdown.png)
- [a-vtrs-equity.png](results/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks_a-vtrs-equity.png)
- [aapl-cohr-drawdown.png](results/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks_aapl-cohr-drawdown.png)
- [aapl-cohr-equity.png](results/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks_aapl-cohr-equity.png)
- [aapl-incy-drawdown.png](results/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks_aapl-incy-drawdown.png)
- [aapl-incy-equity.png](results/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks_aapl-incy-equity.png)
- [archive-metadata.json](results/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks_archive-metadata.json)
- [backtest-report.txt](results/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks_backtest-report.txt)
- [backtest-summary.csv](results/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks_backtest-summary.csv)
- [cointegrated-pairs.csv](results/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks_cointegrated-pairs.csv)
- [cointegration-heatmap.png](results/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks_cointegration-heatmap.png)
- [portfolio-equity.png](results/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks_portfolio-equity.png)
- [portfolio-metrics.csv](results/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks/2026-09-18_15-28-37_pairs-trading-backtest_494-stocks_portfolio-metrics.csv)

## 2026-09-18_15-24-29_pairs-trading-backtest_25-stocks

- [amd-f-drawdown.png](results/2026-09-18_15-24-29_pairs-trading-backtest_25-stocks/2026-09-18_15-24-29_pairs-trading-backtest_25-stocks_amd-f-drawdown.png)
- [amd-f-equity.png](results/2026-09-18_15-24-29_pairs-trading-backtest_25-stocks/2026-09-18_15-24-29_pairs-trading-backtest_25-stocks_amd-f-equity.png)
- [amzn-meta-drawdown.png](results/2026-09-18_15-24-29_pairs-trading-backtest_25-stocks/2026-09-18_15-24-29_pairs-trading-backtest_25-stocks_amzn-meta-drawdown.png)
- [amzn-meta-equity.png](results/2026-09-18_15-24-29_pairs-trading-backtest_25-stocks/2026-09-18_15-24-29_pairs-trading-backtest_25-stocks_amzn-meta-equity.png)
- [archive-metadata.json](results/2026-09-18_15-24-29_pairs-trading-backtest_25-stocks/2026-09-18_15-24-29_pairs-trading-backtest_25-stocks_archive-metadata.json)
- [ibm-ma-drawdown.png](results/2026-09-18_15-24-29_pairs-trading-backtest_25-stocks/2026-09-18_15-24-29_pairs-trading-backtest_25-stocks_ibm-ma-drawdown.png)
- [ibm-ma-equity.png](results/2026-09-18_15-24-29_pairs-trading-backtest_25-stocks/2026-09-18_15-24-29_pairs-trading-backtest_25-stocks_ibm-ma-equity.png)

</details>
