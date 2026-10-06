"""Yahoo Finance adjusted-close acquisition with a verified local cache."""

import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf


def load_prices(universe_path, cache_dir, start, end, refresh=False):
    records = json.loads(Path(universe_path).read_text())
    tickers = sorted(row["Yahoo ticker"] for row in records)
    if len(set(tickers)) != len(tickers):
        raise ValueError("Universe tickers must be unique.")
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("Start must precede exclusive end.")
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / "adjusted-prices.pkl"
    metadata = directory / "price-cache-info.json"
    request = {"tickers": tickers, "start": start, "end_exclusive": end}
    cached = not refresh and cache.exists() and metadata.exists()
    info = json.loads(metadata.read_text()) if cached else {}
    if cached and info.get("request") == request:
        prices = pd.read_pickle(cache)
        print(f"Using cached prices: {cache}", flush=True)
    else:
        raw = yf.download(
            tickers, start=start, end=end, auto_adjust=False, progress=False, threads=8
        )
        if (
            not isinstance(raw.columns, pd.MultiIndex)
            or "Adj Close" not in raw.columns.levels[0]
        ):
            raise ValueError(
                "Expected Yahoo adjusted closes; refusing an unadjusted fallback."
            )
        prices = raw["Adj Close"].sort_index().reindex(columns=tickers)
        if prices.empty:
            raise ValueError("Yahoo returned no prices.")
        prices.to_pickle(cache)
        info = {
            "request": request,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "Yahoo Finance via yfinance",
        }
        metadata.write_text(json.dumps(info, indent=2) + "\n")
    if (
        not isinstance(prices.index, pd.DatetimeIndex)
        or not prices.index.is_unique
        or not prices.index.is_monotonic_increasing
        or prices.index.hasnans
    ):
        raise ValueError("Price dates must be unique, finite and chronological.")
    if (
        prices.empty
        or prices.index[0] < pd.Timestamp(start)
        or prices.index[-1] >= pd.Timestamp(end)
    ):
        raise ValueError("Cache observations do not match the requested window.")
    return prices.reindex(columns=tickers), info


def split_prices(prices, train_fraction=0.7, lookback=60):
    """Determine eligibility from training observations only; never fill prices."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie between zero and one.")
    boundary = int(len(prices) * train_fraction)
    if boundary < max(60, lookback + 1) or len(prices) - boundary < 2:
        raise ValueError("Insufficient training or evaluation observations.")
    training = prices.iloc[:boundary]
    valid = np.isfinite(training).all() & (training > 0).all()
    eligible = sorted(training.columns[valid])
    return (
        training[eligible],
        prices.iloc[boundary:][eligible],
        sorted(set(prices.columns) - set(eligible)),
    )


def reserve_testing_period(prices, testing_closes=252, formation=252):
    """Exclude a trailing test block before any development screen or simulation."""
    if testing_closes < 2 or len(prices) - testing_closes < formation + 2:
        raise ValueError("Need at least two test closes and sufficient development history.")
    development = prices.iloc[:-testing_closes].copy()
    reserved = prices.index[-testing_closes:]
    return development, {
        "reserved_closes": len(reserved),
        "reserved_start": str(reserved[0].date()),
        "reserved_end": str(reserved[-1].date()),
        "development_end": str(development.index[-1].date()),
        "fresh_holdout": False,
        "reason": "Historical dates already inspected; reserved for workflow checks, not untouched validation.",
        "prospective_test": "Freeze rules before inspecting future observations; target 252 new trading closes.",
    }
