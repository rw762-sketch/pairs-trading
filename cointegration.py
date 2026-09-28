"""
Cointegration testing module for identifying cointegrated stock pairs.
Uses Engle-Granger two-step method and ADF test.
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def engle_granger_test(price_series1, price_series2, critical_value=0.05):
    """
    Perform Engle-Granger cointegration test on two price series.
    
    Args:
        price_series1 (pd.Series): First price series
        price_series2 (pd.Series): Second price series
        critical_value (float): Significance level for the test
    
    Returns:
        dict: Test results including p-value, hedge ratio, and test statistic
    """
    # Remove any NaN values
    valid_idx = ~(price_series1.isna() | price_series2.isna())
    s1 = price_series1[valid_idx].values
    s2 = price_series2[valid_idx].values
    
    if len(s1) < 10:
        return None
    
    # Run Engle-Granger test
    score, pvalue, _ = coint(s1, s2)
    
    # Estimate hedge ratio using linear regression
    # Z = Y - beta * X, where beta is the hedge ratio
    X = np.column_stack([np.ones(len(s1)), s2])
    params = np.linalg.lstsq(X, s1, rcond=None)[0]
    hedge_ratio = params[1]
    intercept = params[0]
    
    # Calculate spread
    spread = s1 - hedge_ratio * s2
    
    # ADF test on spread
    adf_result = adfuller(spread, autolag='AIC', store=False, regresults=False)
    adf_pvalue = adf_result[1]
    
    return {
        'coint_pvalue': pvalue,
        'adf_pvalue': adf_pvalue,
        'hedge_ratio': hedge_ratio,
        'intercept': intercept,
        'coint_score': score,
        'adf_statistic': adf_result[0],
        'is_cointegrated': pvalue < critical_value and adf_pvalue < critical_value
    }


def find_cointegrated_pairs(price_data, coint_threshold=0.05, adf_threshold=0.05, sample=None):
    """
    Find all cointegrated pairs in the price data.
    
    Args:
        price_data (pd.DataFrame): Price data with tickers as columns
        coint_threshold (float): Cointegration p-value threshold
        adf_threshold (float): ADF test p-value threshold
        sample (int): Number of pairs to sample (for testing). None to test all.
    
    Returns:
        list: List of (ticker1, ticker2, test_results) tuples for cointegrated pairs
    """
    tickers = price_data.columns.tolist()
    n_tickers = len(tickers)
    cointegrated_pairs = []
    
    # Generate all pairs
    total_pairs = n_tickers * (n_tickers - 1) // 2
    logger.info(f"Testing {total_pairs} pairs for cointegration")
    
    pair_count = 0
    for i in range(n_tickers):
        for j in range(i + 1, n_tickers):
            pair_count += 1
            
            if sample and pair_count > sample:
                logger.info(f"Sample limit reached at {sample} pairs")
                break
            
            ticker1, ticker2 = tickers[i], tickers[j]
            
            result = engle_granger_test(
                price_data[ticker1],
                price_data[ticker2],
                critical_value=coint_threshold
            )
            
            if result and result['is_cointegrated']:
                result['ticker1'] = ticker1
                result['ticker2'] = ticker2
                cointegrated_pairs.append(result)
                
                if pair_count % 1000 == 0:
                    logger.info(f"Processed {pair_count} pairs, found {len(cointegrated_pairs)} cointegrated")
    
    logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs")
    
    return cointegrated_pairs


def rank_pairs(cointegrated_pairs, metric='coint_pvalue'):
    """
    Rank cointegrated pairs by a quality metric.
    
    Args:
        cointegrated_pairs (list): List of cointegrated pair results
        metric (str): Metric to sort by ('coint_pvalue', 'adf_pvalue', 'coint_score')
    
    Returns:
        list: Ranked list of pairs
    """
    ranked = sorted(cointegrated_pairs, key=lambda x: x[metric])
    return ranked


def results_to_dataframe(cointegrated_pairs):
    """Convert cointegration test results to a DataFrame."""
    return pd.DataFrame([
        {
            'Ticker1': p['ticker1'],
            'Ticker2': p['ticker2'],
            'Coint_PValue': p['coint_pvalue'],
            'ADF_PValue': p['adf_pvalue'],
            'Hedge_Ratio': p['hedge_ratio'],
            'Coint_Score': p['coint_score'],
            'ADF_Statistic': p['adf_statistic']
        }
        for p in cointegrated_pairs
    ])


if __name__ == "__main__":
    from data_fetcher import fetch_price_data, clean_price_data, split_data
    from datetime import datetime, timedelta
    
    # Example usage
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'V', 'JNJ', 'WMT', 'PG']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    
    price_data = fetch_price_data(tickers, start_date, end_date)
    price_data = clean_price_data(price_data)
    train_data, test_data = split_data(price_data)
    
    pairs = find_cointegrated_pairs(train_data, coint_threshold=0.05, sample=10)
    pairs_df = results_to_dataframe(pairs)
    
    print(f"Found {len(pairs)} cointegrated pairs")
    print(pairs_df.head())
