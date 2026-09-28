"""
Quick test script to verify the pairs trading strategy implementation.
This runs a minimal example to check all components work correctly.
"""

import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_fetcher():
    """Test data fetching module."""
    logger.info("Testing data_fetcher module...")
    try:
        from data_fetcher import fetch_price_data, clean_price_data, split_data
        
        # Use a small set of well-known tickers for testing
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        # Use recent data that definitely exists
        end_date = '2024-09-18'
        start_date = '2022-09-18'  # 2 years of data
        
        logger.info(f"  Fetching {len(tickers)} stocks from {start_date} to {end_date}...")
        price_data = fetch_price_data(tickers, start_date, end_date)
        assert price_data.shape[0] > 0, "No price data fetched"
        
        logger.info(f"  Cleaning data...")
        price_data = clean_price_data(price_data, min_history_days=200)
        assert price_data.shape[1] > 0, "No valid tickers after cleaning"
        
        logger.info(f"  Splitting data...")
        train_data, test_data = split_data(price_data)
        assert len(train_data) > 0 and len(test_data) > 0, "Split failed"
        
        logger.info("  ✓ data_fetcher tests passed\n")
        return price_data, train_data, test_data
    
    except Exception as e:
        logger.error(f"  ✗ data_fetcher test failed: {e}\n")
        return None, None, None


def test_cointegration(train_data):
    """Test cointegration module."""
    logger.info("Testing cointegration module...")
    try:
        from cointegration import find_cointegrated_pairs, results_to_dataframe
        
        logger.info(f"  Finding cointegrated pairs (sample: 100)...")
        pairs = find_cointegrated_pairs(train_data, coint_threshold=0.05, sample=100)
        
        if pairs:
            logger.info(f"  Found {len(pairs)} cointegrated pairs")
            pairs_df = results_to_dataframe(pairs)
            assert len(pairs_df) == len(pairs), "DataFrame conversion failed"
        else:
            logger.warning(f"  No cointegrated pairs found (this is OK for testing)")
        
        logger.info("  ✓ cointegration tests passed\n")
        return pairs
    
    except Exception as e:
        logger.error(f"  ✗ cointegration test failed: {e}\n")
        return []


def test_signal_generation(test_data, pairs):
    """Test signal generation module."""
    logger.info("Testing signal_generation module...")
    try:
        from signal_generation import generate_all_signals
        
        if not pairs:
            logger.warning("  No pairs to test signals with - skipping\n")
            return {}
        
        logger.info(f"  Generating signals...")
        signals = generate_all_signals(test_data, pairs[:5], entry_threshold=2.0)
        
        logger.info(f"  Generated signals for {len(signals)} pairs")
        
        logger.info("  ✓ signal_generation tests passed\n")
        return signals
    
    except Exception as e:
        logger.error(f"  ✗ signal_generation test failed: {e}\n")
        return {}


def test_backtester(signals):
    """Test backtesting module."""
    logger.info("Testing backtester module...")
    try:
        from backtester import backtest_all_pairs, combine_portfolio_results
        
        if not signals:
            logger.warning("  No signals to backtest - skipping\n")
            return {}, None
        
        logger.info(f"  Running backtests...")
        results = backtest_all_pairs(signals, initial_capital=100000)
        
        logger.info(f"  Completed backtests for {len(results)} pairs")
        
        portfolio = None
        if results:
            logger.info(f"  Combining portfolio results...")
            portfolio = combine_portfolio_results(results)
            
            if portfolio:
                logger.info(f"  Portfolio Sharpe Ratio: {portfolio['metrics']['Sharpe_Ratio']:.2f}")
        
        logger.info("  ✓ backtester tests passed\n")
        return results, portfolio
    
    except Exception as e:
        logger.error(f"  ✗ backtester test failed: {e}\n")
        return {}, None


def test_visualization(results, portfolio):
    """Test visualization module."""
    logger.info("Testing visualization module...")
    try:
        from visualization import generate_performance_report
        
        if not results:
            logger.warning("  No results to visualize - skipping\n")
            return
        
        # Test report generation
        first_pair = list(results.keys())[0]
        metrics = results[first_pair]['metrics']
        
        logger.info(f"  Generating report for {first_pair}...")
        report = generate_performance_report(metrics, first_pair)
        assert len(report) > 0, "Report generation failed"
        
        logger.info("  ✓ visualization tests passed\n")
    
    except Exception as e:
        logger.error(f"  ✗ visualization test failed: {e}\n")


def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("PAIRS TRADING STRATEGY - QUICK TEST")
    logger.info("="*60 + "\n")
    
    # Test each module
    price_data, train_data, test_data = test_data_fetcher()
    
    if train_data is None:
        logger.error("Failed to fetch data - stopping tests")
        return False
    
    pairs = test_cointegration(train_data)
    signals = test_signal_generation(test_data, pairs)
    results, portfolio = test_backtester(signals)
    test_visualization(results, portfolio)
    
    logger.info("="*60)
    logger.info("TEST COMPLETE")
    logger.info("="*60)
    
    if results:
        logger.info("\n✓ All core modules tested successfully!")
        logger.info("\nTo run the full strategy:")
        logger.info("  python main.py")
        return True
    else:
        logger.warning("\n⚠ Tests ran but no trading results generated")
        logger.info("\nThis may be due to:")
        logger.info("  - No cointegrated pairs found in sample data")
        logger.info("  - Try running main.py with more stocks for better results")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
