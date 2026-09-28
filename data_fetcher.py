"""
Data fetching and preprocessing module for pairs trading strategy.
Downloads S&P 500 stock prices and handles data cleaning.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_sp500_tickers():
    """Fetch S&P 500 tickers from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(
        url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    table = next(t for t in tables if 'Symbol' in t.columns)
    # Yahoo Finance uses hyphens for share classes (for example, BRK-B).
    tickers = table['Symbol'].str.replace('.', '-', regex=False).tolist()
    logger.info(f"Fetched {len(tickers)} S&P 500 tickers")
    return tickers


def fetch_price_data(tickers, start_date, end_date):
    """
    Fetch historical price data for given tickers.
    
    Args:
        tickers (list): List of stock tickers
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
    
    Returns:
        pd.DataFrame: Closing prices with tickers as columns and dates as index
    """
    logger.info(f"Fetching price data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=True
        )

        # Yahoo Finance returns a MultiIndex for multiple tickers.
        if isinstance(data.columns, pd.MultiIndex):
            price_field = 'Adj Close' if 'Adj Close' in data.columns.levels[0] else 'Close'
            data = data[price_field]
        elif 'Adj Close' in data.columns:
            data = data['Adj Close']
        elif len(tickers) == 1 and isinstance(data, pd.DataFrame):
            data = data.to_frame()
            data.columns = tickers
        
        # Handle case where single ticker returns Series
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = tickers if len(tickers) == 1 else [data.name or tickers[0]]
        
        return data
    
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise


def clean_price_data(price_data, min_history_days=252):
    """
    Clean price data by removing tickers with insufficient data.
    
    Args:
        price_data (pd.DataFrame): Price data with tickers as columns
        min_history_days (int): Minimum number of trading days required
    
    Returns:
        pd.DataFrame: Cleaned price data
    """
    logger.info(f"Cleaning data: removing tickers with less than {min_history_days} days of history")
    
    # Remove tickers with missing data
    price_data = price_data.dropna(axis=1, how='any')
    
    # Check that we have enough history for each ticker
    min_dates = price_data.count()
    valid_tickers = min_dates[min_dates >= min_history_days].index.tolist()
    
    logger.info(f"Removed {len(price_data.columns) - len(valid_tickers)} tickers with insufficient data")
    logger.info(f"Keeping {len(valid_tickers)} tickers with at least {min_history_days} days")
    
    return price_data[valid_tickers]


def calculate_log_returns(price_data):
    """
    Calculate log returns from price data.
    
    Args:
        price_data (pd.DataFrame): Price data
    
    Returns:
        pd.DataFrame: Log returns
    """
    return np.log(price_data / price_data.shift(1)).dropna()


def split_data(price_data, train_end_ratio=0.7):
    """
    Split data into in-sample (training) and out-of-sample (testing) periods.
    
    Args:
        price_data (pd.DataFrame): Price data
        train_end_ratio (float): Ratio for train/test split
    
    Returns:
        tuple: (train_data, test_data)
    """
    split_index = int(len(price_data) * train_end_ratio)
    train_data = price_data.iloc[:split_index]
    test_data = price_data.iloc[split_index:]
    
    logger.info(f"Data split: {len(train_data)} in-sample, {len(test_data)} out-of-sample")
    
    return train_data, test_data


if __name__ == "__main__":
    # Example usage
    tickers = get_sp500_tickers()[:50]  # Use first 50 for testing
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
    
    price_data = fetch_price_data(tickers, start_date, end_date)
    price_data = clean_price_data(price_data)
    train_data, test_data = split_data(price_data)
    
    print(f"Final dataset shape: {price_data.shape}")
    print(f"Price data head:\n{price_data.head()}")
