"""Training-only K-means prefilter for pairs research.

Each stock is one sample; standardized daily log returns are its features.
Cluster membership is a computational filter, not evidence of cointegration.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


def cluster_stocks(training_prices, n_clusters=10, random_state=42):
    """Fit deterministic clusters on supplied training observations only.

    Constant-return stocks are excluded explicitly rather than assigned using
    invalid standardized values. Missing prices are never filled.
    """
    if (
        not isinstance(n_clusters, int)
        or isinstance(n_clusters, bool)
        or n_clusters < 1
    ):
        raise ValueError("n_clusters must be a positive integer")
    prices = training_prices.sort_index(axis=1)
    if len(prices) < 3 or prices.columns.duplicated().any():
        raise ValueError("Need at least three observations and unique ticker columns")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Training prices must be complete, finite and positive")
    returns = np.diff(np.log(values), axis=0).T
    scale = returns.std(axis=1)
    valid = scale > 1e-12
    excluded = prices.columns[~valid].tolist()
    if not valid.any():
        return pd.Series(dtype=int, name="Cluster"), {
            "excluded_tickers": excluded,
            "actual_clusters": 0,
        }
    features = (returns[valid] - returns[valid].mean(axis=1, keepdims=True)) / scale[
        valid, None
    ]
    count = min(n_clusters, len(features), len(np.unique(features, axis=0)))
    # Avoid BLAS/OpenMP oversubscription for this small stock universe.
    with threadpool_limits(limits=1):
        labels = KMeans(
            n_clusters=count, random_state=random_state, n_init=10
        ).fit_predict(features)
    assignments = pd.Series(labels, index=prices.columns[valid], name="Cluster")
    return assignments, {
        "excluded_tickers": excluded,
        "actual_clusters": int(assignments.nunique()),
        "requested_clusters": n_clusters,
        "random_state": random_state,
        "n_init": 10,
        "features": "per-stock standardized training daily log returns",
    }


def within_cluster_pairs(assignments):
    """Unique alphabetical combinations; never construct cross-cluster pairs."""
    return [
        pair
        for _, group in assignments.groupby(assignments)
        for pair in combinations(sorted(group.index), 2)
    ]
