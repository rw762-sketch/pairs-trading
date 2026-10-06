"""Training causality, pair coverage, and cluster feature validation."""

import unittest
import numpy as np
import pandas as pd
from stock_clustering import cluster_stocks, within_cluster_pairs


def prices():
    rng = np.random.default_rng(81)
    common = rng.normal(0, 0.01, (160, 2))
    returns = np.column_stack(
        [common[:, i // 3] + rng.normal(0, 0.0001, 160) for i in range(6)]
    )
    return pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=pd.bdate_range("2020-01-01", periods=160),
        columns=list("ABCDEF"),
    )


class ClusteringTests(unittest.TestCase):
    def test_two_return_groups_only_generate_within_group_pairs(self):
        labels, _ = cluster_stocks(prices(), 2)
        self.assertEqual(labels.A, labels.B)
        self.assertEqual(labels.A, labels.C)
        self.assertEqual(labels.D, labels.E)
        self.assertNotEqual(labels.A, labels.D)
        pairs = within_cluster_pairs(labels)
        self.assertEqual(len(pairs), 6)
        self.assertEqual(len(set(pairs)), 6)
        self.assertTrue(all(labels[a] == labels[b] for a, b in pairs))

    def test_future_prices_and_column_order_cannot_change_training_clusters(self):
        data = prices()
        first, _ = cluster_stocks(data.iloc[:100], 2)
        data.iloc[100:] *= np.arange(1, 7) * 100
        second, _ = cluster_stocks(data.iloc[:100, ::-1], 2)
        pd.testing.assert_series_equal(first, second)

    def test_one_cluster_covers_every_possible_pair(self):
        labels, _ = cluster_stocks(prices(), 1)
        self.assertEqual(len(within_cluster_pairs(labels)), 15)

    def test_constant_stocks_and_invalid_inputs(self):
        data = prices()
        data["G"] = 100.0
        labels, info = cluster_stocks(data, 100)
        self.assertEqual(info["excluded_tickers"], ["G"])
        self.assertNotIn("G", labels.index)
        labels, _ = cluster_stocks(data[["G"]])
        self.assertEqual(within_cluster_pairs(labels), [])
        for count in (0, -1, True):
            with self.assertRaises(ValueError):
                cluster_stocks(data, count)
        data.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            cluster_stocks(data)
