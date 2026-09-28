"""
Configuration file for pairs trading strategy.
Adjust these parameters to customize the strategy.
"""

# Data Configuration
DATA_CONFIG = {
    # Date range for data
    'start_date': '2021-01-01',  # Start of historical data
    'end_date': None,             # None = today; or specify as 'YYYY-MM-DD'
    
    # Number of S&P 500 stocks to analyze (0 = all, otherwise top N by market cap)
    'num_stocks': 50,
    
    # Minimum trading days required per stock
    'min_history_days': 252,  # 1 year
    
    # Data split: training % and test %
    'train_test_split': 0.7,  # 70% train, 30% test
}

# Cointegration Testing Configuration
COINTEGRATION_CONFIG = {
    # P-value threshold for accepting cointegration
    'coint_threshold': 0.05,  # 5% significance level
    
    # ADF test p-value threshold
    'adf_threshold': 0.05,
    
    # Number of pairs to sample for initial testing (None = all pairs)
    'sample_pairs': None,
}

# Trading Signal Configuration
SIGNAL_CONFIG = {
    # Entry threshold (z-score standard deviations)
    # Long when z-score < -entry_zscore
    # Short when z-score > +entry_zscore
    'entry_zscore': 2.0,
    
    # Exit threshold (z-score standard deviations)
    # Exit long when z-score > exit_zscore
    # Exit short when z-score < -exit_zscore
    'exit_zscore': 0.0,
    
    # Lookback window for calculating z-score (in trading days)
    'lookback_window': 60,
    
    # Maximum holding period in days (None = unlimited)
    'max_holding_period': None,
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    # Initial capital per pair
    'initial_capital': 100000,
    
    # Transaction costs (as decimal: 0.001 = 0.1%)
    'transaction_cost': 0.001,  # Typical: 0.05% - 0.15%
    
    # Portfolio weighting method: 'equal' or 'sharpe_ratio'
    'portfolio_weighting': 'equal',
}

# Parameter Optimization Configuration
OPTIMIZATION_CONFIG = {
    # Ranges of parameters to test
    'entry_zscore_range': [1.5, 2.0, 2.5],
    'exit_zscore_range': [0.0, 0.25, 0.5],
    'coint_threshold_range': [0.01, 0.05, 0.10],
    
    # Metric to optimize for: 'Sharpe_Ratio', 'Total_Return', 'Annual_Return', 'Max_Drawdown'
    'optimize_metric': 'Sharpe_Ratio',
    
    # Number of top parameter sets to report
    'top_n_results': 10,
}

# Output Configuration
OUTPUT_CONFIG = {
    # Directory to save results
    'output_dir': 'results',
    
    # Generate visualizations
    'save_plots': True,
    
    # Generate text report
    'generate_report': True,
    
    # Save detailed trade logs
    'save_trade_logs': True,
    
    # Verbosity: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'log_level': 'INFO',
}

# Strategy Validation Configuration
VALIDATION_CONFIG = {
    # Check for overfitting: flag if in-sample Sharpe degrades > threshold
    'sharpe_degradation_threshold': 0.2,
    
    # Flag if return degrades more than this
    'return_degradation_threshold': 0.3,
    
    # Minimum number of trades for a valid backtest
    'min_trades': 3,
    
    # Require out-of-sample testing
    'require_oos_validation': True,
    
    # Check that pairs remain cointegrated on test data
    'validate_oos_cointegration': True,
}


def get_config():
    """Get complete configuration dictionary."""
    return {
        'data': DATA_CONFIG,
        'cointegration': COINTEGRATION_CONFIG,
        'signals': SIGNAL_CONFIG,
        'backtest': BACKTEST_CONFIG,
        'optimization': OPTIMIZATION_CONFIG,
        'output': OUTPUT_CONFIG,
        'validation': VALIDATION_CONFIG,
    }


if __name__ == "__main__":
    import json
    config = get_config()
    print("Current Configuration:")
    print(json.dumps(config, indent=2, default=str))
