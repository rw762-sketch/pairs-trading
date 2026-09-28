"""
Parameter optimization module for pairs trading strategy.
Tests different parameter combinations to find optimal settings.
"""

import pandas as pd
import numpy as np
import itertools
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParameterOptimizer:
    """Optimize trading parameters using grid search."""
    
    def __init__(self, train_data, test_data, pairs_train, pairs_test, initial_capital=100000):
        """
        Initialize optimizer.
        
        Args:
            train_data (pd.DataFrame): Training price data
            test_data (pd.DataFrame): Testing price data
            pairs_train (list): Cointegrated pairs from training data
            pairs_test (list): Cointegrated pairs from testing data
            initial_capital (float): Initial capital
        """
        self.train_data = train_data
        self.test_data = test_data
        self.pairs_train = pairs_train
        self.pairs_test = pairs_test
        self.initial_capital = initial_capital
        self.optimization_results = []
    
    def optimize_parameters(self, entry_thresholds=[1.5, 2.0, 2.5], 
                          exit_thresholds=[0, 0.25, 0.5],
                          coint_thresholds=[0.01, 0.05, 0.10],
                          transaction_cost=0.001):
        """
        Perform grid search over parameters.
        
        Args:
            entry_thresholds (list): Entry z-score thresholds to test
            exit_thresholds (list): Exit z-score thresholds to test
            coint_thresholds (list): Cointegration p-value thresholds
            transaction_cost (float): Transaction cost
        
        Returns:
            pd.DataFrame: Results of all parameter combinations
        """
        from signal_generation import generate_all_signals
        from backtester import backtest_all_pairs, combine_portfolio_results
        
        combinations = list(itertools.product(
            entry_thresholds,
            exit_thresholds,
            coint_thresholds
        ))
        
        logger.info(f"Testing {len(combinations)} parameter combinations")
        
        results = []
        
        for idx, (entry, exit_t, coint) in enumerate(combinations):
            try:
                # Filter pairs based on cointegration threshold
                filtered_pairs = [p for p in self.pairs_train if p['coint_pvalue'] < coint]
                
                if not filtered_pairs:
                    logger.warning(f"No pairs passed cointegration filter with p<{coint}")
                    continue
                
                # Generate signals on test data
                signal_data = generate_all_signals(
                    self.test_data,
                    filtered_pairs,
                    entry_threshold=entry,
                    exit_threshold=exit_t
                )
                
                if not signal_data:
                    continue
                
                # Run backtest
                backtest_results = backtest_all_pairs(signal_data, self.initial_capital, transaction_cost)
                portfolio = combine_portfolio_results(backtest_results)
                
                if portfolio:
                    result = {
                        'Entry_Threshold': entry,
                        'Exit_Threshold': exit_t,
                        'Coint_Threshold': coint,
                        'Num_Pairs': len(filtered_pairs),
                        'Num_Traded_Pairs': len(backtest_results),
                        'Total_Return': portfolio['metrics']['Total_Return'],
                        'Annual_Return': portfolio['metrics']['Annualized_Return'],
                        'Volatility': portfolio['metrics']['Volatility'],
                        'Sharpe_Ratio': portfolio['metrics']['Sharpe_Ratio'],
                        'Max_Drawdown': portfolio['metrics']['Max_Drawdown']
                    }
                    results.append(result)
                    
                    if (idx + 1) % 5 == 0:
                        logger.info(f"Completed {idx + 1}/{len(combinations)} combinations")
            
            except Exception as e:
                logger.warning(f"Error with parameters ({entry}, {exit_t}, {coint}): {e}")
                continue
        
        self.optimization_results = pd.DataFrame(results)
        
        logger.info(f"Optimization complete: tested {len(results)} valid combinations")
        
        return self.optimization_results
    
    def get_best_parameters(self, metric='Sharpe_Ratio', top_n=5):
        """
        Get the best parameter combinations.
        
        Args:
            metric (str): Metric to optimize (Sharpe_Ratio, Total_Return, etc.)
            top_n (int): Number of top results to return
        
        Returns:
            pd.DataFrame: Top parameter combinations
        """
        if self.optimization_results.empty:
            raise ValueError("Must run optimization first")
        
        return self.optimization_results.nlargest(top_n, metric)
    
    def sensitivity_analysis(self, base_params):
        """
        Analyze sensitivity of performance to parameter changes.
        
        Args:
            base_params (dict): Base parameter dictionary
        
        Returns:
            dict: Sensitivity results
        """
        # This would contain code to analyze how performance varies
        # with small changes to each parameter
        pass


def compare_in_sample_vs_out_of_sample(train_results, test_results):
    """
    Compare in-sample vs out-of-sample performance.
    
    Args:
        train_results (dict): Backtest results on training data
        test_results (dict): Backtest results on test data
    
    Returns:
        pd.DataFrame: Comparison table
    """
    comparison = []
    
    for pair_id in train_results.keys():
        if pair_id not in test_results:
            continue
        
        train_metrics = train_results[pair_id]['metrics']
        test_metrics = test_results[pair_id]['metrics']
        
        comparison.append({
            'Pair': pair_id,
            'Train_Sharpe': train_metrics['Sharpe_Ratio'],
            'Test_Sharpe': test_metrics['Sharpe_Ratio'],
            'Sharpe_Degradation': train_metrics['Sharpe_Ratio'] - test_metrics['Sharpe_Ratio'],
            'Train_Return': train_metrics['Annualized_Return'],
            'Test_Return': test_metrics['Annualized_Return'],
            'Return_Degradation': train_metrics['Annualized_Return'] - test_metrics['Annualized_Return']
        })
    
    return pd.DataFrame(comparison)


def detect_overfitting(train_test_comparison, degradation_threshold=0.2):
    """
    Detect potential overfitting based on performance degradation.
    
    Args:
        train_test_comparison (pd.DataFrame): Comparison of in/out-of-sample
        degradation_threshold (float): Threshold for flagging overfitting
    
    Returns:
        pd.DataFrame: Pairs flagged as potentially overfitted
    """
    flagged = train_test_comparison[
        train_test_comparison['Sharpe_Degradation'] > degradation_threshold
    ].copy()
    
    return flagged


def analyze_transaction_cost_impact(backtest_results, cost_scenarios=[0, 0.0005, 0.001, 0.002]):
    """
    Analyze impact of different transaction cost assumptions.
    
    Args:
        backtest_results (dict): Backtest results
        cost_scenarios (list): Transaction costs to test
    
    Returns:
        pd.DataFrame: Analysis of cost impact on returns
    """
    # This would rerun backtests with different cost assumptions
    # and show how performance changes
    pass


if __name__ == "__main__":
    from data_fetcher import fetch_price_data, clean_price_data, split_data
    from cointegration import find_cointegrated_pairs
    
    # Example usage
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'V', 'JNJ', 'WMT', 'PG']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    
    price_data = fetch_price_data(tickers, start_date, end_date)
    price_data = clean_price_data(price_data)
    train_data, test_data = split_data(price_data, train_end_ratio=0.5)
    
    pairs_train = find_cointegrated_pairs(train_data, sample=20)
    pairs_test = find_cointegrated_pairs(test_data, sample=20)
    
    optimizer = ParameterOptimizer(train_data, test_data, pairs_train, pairs_test)
    results = optimizer.optimize_parameters()
    
    print("Optimization Results:")
    print(results)
    
    print("\nBest Parameters:")
    print(optimizer.get_best_parameters(top_n=3))
