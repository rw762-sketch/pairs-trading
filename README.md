# Pairs Trading Strategy Using Cointegration

> **Current report:** Open [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for the corrected results and the complete assignment outputs. Run `./venv/bin/python main.py --report` to rebuild them from saved data. The older overview below describes the initial scaffold; its performance expectations, stock-limit settings, output filenames and optimization claims are not a validation of the current strategy. The current report documents the implemented rules and limitations.

A comprehensive Python implementation of a statistical arbitrage strategy that identifies cointegrated stock pairs and executes mean-reversion trades.

## Project Overview

This project implements a pairs trading strategy that:
1. **Identifies cointegrated pairs** - Finds stock pairs whose prices move together long-term using Engle-Granger cointegration test
2. **Generates trading signals** - Calculates spreads and z-scores to identify mean-reversion opportunities
3. **Backtests the strategy** - Simulates trading with transaction costs and generates performance metrics
4. **Validates out-of-sample** - Tests strategy on unseen data to avoid overfitting
5. **Optimizes parameters** - Finds optimal entry/exit thresholds and cointegration filters

## Key Features

- **Statistical Testing**: Engle-Granger two-step cointegration test with ADF validation
- **Sophisticated Signals**: Rolling z-score calculation with dynamic entry/exit thresholds
- **Realistic Backtesting**: Transaction costs, slippage, and holding period constraints
- **Comprehensive Analysis**: Sharpe ratio, drawdown, win rate, turnover metrics
- **Parameter Optimization**: Grid search over parameter combinations
- **Overfitting Detection**: Compare in-sample vs out-of-sample performance
- **Professional Visualizations**: Equity curves, drawdown charts, spread analysis
- **Detailed Reporting**: Trade logs, performance summaries, statistical analysis

## Installation

### Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

### Setup

```bash
# Clone or navigate to project directory
cd Pair\ trading\ project

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt


```

## Quick Start

### Running the Full Pipeline

```bash
python main.py
```

This will:
1. Download 3 years of S&P 500 stock data (30 stocks by default)
2. Find cointegrated pairs in training data (70% of data)
3. Generate trading signals on test data (30% of data)
4. Backtest the strategy with realistic costs
5. Generate visualizations and reports

Results are saved to the `results/` directory:
- `cointegrated_pairs.csv` - List of identified pairs
- `cointegration_heatmap.png` - Visualization of cointegration relationships
- `backtest_summary.csv` - Performance metrics for each pair
- `portfolio_metrics.csv` - Overall portfolio performance
- `backtest_report.txt` - Comprehensive text report
- Individual pair charts (equity curves, drawdown, etc.)

### Custom Configuration

Edit `main.py` to customize parameters:

```python
config = {
    'num_stocks': 50,              # Stocks to analyze
    'start_date': '2021-01-01',    # Data start date
    'end_date': '2024-01-01',      # Data end date
    'train_test_split': 0.7,       # 70% train, 30% test
    'entry_zscore': 2.0,           # Entry signal threshold
    'exit_zscore': 0.0,            # Exit signal threshold
    'transaction_cost': 0.001,     # 0.1% trading cost
    'initial_capital': 100000,     # Starting capital
    'output_dir': 'results'
}
```

## Module Guide

### `data_fetcher.py`
Handles data acquisition and preprocessing.

**Key Functions:**
- `get_sp500_tickers()` - Fetches S&P 500 stock symbols
- `fetch_price_data()` - Downloads historical prices via yfinance
- `clean_price_data()` - Removes stocks with insufficient data
- `split_data()` - Divides data into train/test periods

### `cointegration.py`
Implements cointegration testing for pair identification.

**Key Functions:**
- `engle_granger_test()` - Two-step cointegration test with ADF validation
- `find_cointegrated_pairs()` - Screens all pairs for cointegration
- `rank_pairs()` - Sorts pairs by quality metrics

**Theory:**
- Two price series are cointegrated if their spread (linear combination) is stationary
- Tests use Engle-Granger two-step method: regress Y on X, then test residuals for stationarity
- ADF test checks if residuals have a unit root (non-stationary)

### `signal_generation.py`
Creates trading signals from cointegrated pairs.

**Key Classes:**
- `PairSignalGenerator` - Calculates spread, z-score, and generates entry/exit signals

**Trading Logic:**
- **Long spread when**: z-score < -2 (undervalued - short ticker2, long ticker1)
- **Short spread when**: z-score > +2 (overvalued - long ticker2, short ticker1)
- **Exit when**: z-score reverts to ±0 (mean reversion complete)

### `backtester.py`
Simulates trading execution and calculates performance metrics.

**Key Classes:**
- `PairBacktester` - Runs backtest for a single pair
- Portfolio combination and weighting functions

**Metrics:**
- **Returns**: Total return, annualized return
- **Risk**: Volatility, max drawdown, Sharpe ratio
- **Trading**: Win rate, average trade P&L, turnover, costs

### `visualization.py`
Generates charts and reports.

**Key Functions:**
- `plot_equity_curve()` - Portfolio value over time
- `plot_spread_and_zscore()` - Signals with entry/exit markers
- `plot_drawdown()` - Peak-to-trough declines
- `plot_cointegration_heatmap()` - Pair relationships
- `generate_comprehensive_report()` - Text summary of results

### `parameter_optimization.py`
Tunes strategy parameters for optimal performance.

**Key Classes:**
- `ParameterOptimizer` - Grid search over parameter combinations
- Overfitting detection via in/out-of-sample comparison

## Strategy Parameters to Optimize

### Cointegration
- **`coint_threshold`** (default: 0.05): P-value threshold for cointegration test
  - Lower = stricter requirement for cointegration
  - Typical range: 0.01 to 0.10

### Trading Signals
- **`entry_zscore`** (default: 2.0): Entry threshold (in standard deviations)
  - Higher = wait for larger deviations (fewer trades, less noise)
  - Typical range: 1.5 to 3.0

- **`exit_zscore`** (default: 0.0): Exit threshold
  - Higher = exit later (more mean reversion captured)
  - Typical range: 0.0 to 1.0

### Costs
- **`transaction_cost`** (default: 0.001): Cost per trade (0.1%)
  - Includes bid-ask spread and commissions
  - Check with your broker for actual costs

## Key Insights & Questions

The project guides you to think deeply about:

### 1. Correlation vs Cointegration
- High correlation ≠ cointegration
- Correlation is instantaneous; cointegration is about long-term equilibrium
- Many correlated pairs break down (regime changes, structural breaks)

### 2. Out-of-Sample Reality
- Cointegrated in-sample often breaks down out-of-sample
- Market regimes change; statistical relationships decay
- Watch for "overfitting" where in-sample >> out-of-sample returns

### 3. Transaction Costs Destroy Profits
- Mean-reversion arbitrage has tight margins
- With 0.1% costs, need 0.2% spread profit just to break even
- High turnover pairs often become unprofitable

### 4. Concentration Risk
- Few pairs often drive most returns
- Single-pair failures can significantly impact portfolio
- Diversification across many weak pairs often beats few strong pairs

### 5. Parameter Sensitivity
- Performance is sensitive to entry/exit thresholds
- Optimization on training data often leads to overfitting
- Robust parameters perform better across different market regimes

## Output Files

```
results/
├── cointegrated_pairs.csv           # Identified pairs with p-values
├── cointegration_heatmap.png        # Heatmap of all pair relationships
├── backtest_summary.csv             # Metrics for each pair
├── portfolio_metrics.csv            # Portfolio performance
├── backtest_report.txt              # Comprehensive text report
├── [PAIR]_equity.png                # Equity curve for pair
├── [PAIR]_drawdown.png              # Drawdown chart for pair
└── portfolio_equity.png             # Portfolio equity curve
```

## Python Libraries Used

- **`pandas`**: Data manipulation and analysis
- **`numpy`**: Numerical computing
- **`yfinance`**: Download stock data from Yahoo Finance
- **`statsmodels`**: Statistical testing (cointegration, ADF test)
- **`scipy`**: Scientific functions
- **`matplotlib` / `seaborn`**: Visualization

## Common Issues & Solutions

### Issue: "No pairs found"
- Try lower cointegration threshold (0.10 instead of 0.05)
- Use more stocks in analysis
- Check data has no missing values

### Issue: Poor out-of-sample performance
- Likely overfitting - reduce entry threshold strictness
- Use larger training period (2+ years)
- Check if pairs lose cointegration out-of-sample

### Issue: Data download fails
- Check internet connection
- Yahoo Finance sometimes blocks requests - try later
- Some tickers may not have full data history

### Issue: Slow execution
- Reduce `num_stocks` for initial testing
- Cointegration testing is O(n²) in number of stocks
- Consider sampling pairs instead of testing all

## Future Enhancements

- **Machine Learning**: Use ML to predict which pairs stay cointegrated
- **Multiple Entry Signals**: Combine cointegration with momentum or volatility
- **Risk Management**: Position sizing, dynamic hedge ratios
- **Portfolio Optimization**: Kelly criterion, risk parity weighting
- **Real-time Trading**: Live signal generation and execution
- **Multi-timeframe**: Combine daily and intraday signals

## References

- Engle, R. F., & Granger, C. W. (1987). "Co-integration and error correction: representation, estimation, and testing"
- Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis"
- De Prado, M. L. (2018). "Advances in Financial Machine Learning"

## Disclaimer

This project is for educational purposes only. Past performance does not guarantee future results. Pairs trading involves substantial risk of loss. Always paper trade and back-test thoroughly before risking real capital. Consult a financial advisor before implementing any trading strategy.

## Author

Created as a comprehensive learning project on statistical arbitrage and quantitative trading.

---

For questions or improvements, feel free to modify and extend the code!
