"""All-pair coverage, chronological eligibility and cash-allocation checks."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from cointegration import COLUMNS, screen_pairs
from config import Config
from data_fetcher import split_prices, reserve_testing_period
from main import run_pipeline


def prices():
    rng = np.random.default_rng(71)
    return pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (150, 3)), axis=0)),
        columns=list("ABC"),
        index=pd.bdate_range("2020-01-01", periods=150),
    )


class PipelineTests(unittest.TestCase):
    def test_default_evaluator_disables_holding_timeout(self):
        from evaluation import evaluate_pairs
        from backtester import PairBacktester
        data = prices()
        selected = pd.DataFrame([{"Pair": "A-B", "Ticker1": "A", "Ticker2": "B", "Beta": 1.0, "Weight": 1.0}])
        with patch("evaluation.PairBacktester", wraps=PairBacktester) as engine:
            evaluate_pairs(data, data.index[-30:], selected, Config())
        self.assertIsNone(engine.call_args.kwargs["max_holding_period"])

    def test_reserved_prices_cannot_enter_development(self):
        data = prices()
        development, reservation = reserve_testing_period(data, 40, 60)
        pd.testing.assert_frame_equal(development, data.iloc[:110])
        changed = data.copy()
        changed.iloc[-40:] *= 100
        second, _ = reserve_testing_period(changed, 40, 60)
        pd.testing.assert_frame_equal(development, second)
        self.assertLess(development.index[-1], pd.Timestamp(reservation["reserved_start"]))
        self.assertFalse(reservation["fresh_holdout"])
        with self.assertRaises(ValueError):
            reserve_testing_period(data, 100, 60)

    def test_every_pair_tested_even_with_negative_beta_or_failed_i1(self):
        data = prices()
        diagnostics = pd.DataFrame({"Ticker": list("ABC"), "I1_Compatible": False})
        with (
            patch("cointegration.integration_diagnostics", return_value=diagnostics),
            patch("cointegration.coint", return_value=(0, 0.001, [])) as test,
        ):
            panel, _ = screen_pairs(data, [("A", "B"), ("A", "C"), ("B", "C")])
        self.assertEqual(test.call_count, 3)
        self.assertEqual(set(panel.Pair), {"A-B", "A-C", "B-C"})
        self.assertFalse(panel.Selected.any())

    def test_failed_tests_are_preserved_in_holm_family(self):
        with patch(
            "cointegration.coint",
            side_effect=[(0, 0.01, []), ValueError("failed test")],
        ):
            panel, _ = screen_pairs(prices(), [("A", "B"), ("A", "C")])
        self.assertEqual(len(panel), 2)
        self.assertAlmostEqual(panel.loc[panel.Pair.eq("A-B"), "Holm_P"].iloc[0], 0.02)
        self.assertEqual(panel.loc[panel.Pair.eq("A-C"), "Status"].iloc[0], "failed")

    def test_empty_pair_family_has_no_selected_pairs(self):
        panel, _ = screen_pairs(prices(), [])
        self.assertTrue(panel.empty)
        self.assertEqual(panel.Selected.dtype, bool)

    def test_training_eligibility_does_not_use_later_missing_prices(self):
        data = prices()
        training, later, excluded = split_prices(data)
        data.iloc[-1, 0] = np.nan
        second, _, excluded2 = split_prices(data)
        pd.testing.assert_frame_equal(training, second)
        self.assertEqual(excluded, excluded2)

    def test_zero_selection_produces_complete_cash_account(self):
        with tempfile.TemporaryDirectory() as destination, patch("main.write_report"):
            config = Config(clusters=1, correction="holm", pvalue=1e-20)
            result = run_pipeline(prices(), config, destination)
            self.assertTrue(
                (result["portfolio"]["equity_curve"] == config.capital).all()
            )
            self.assertEqual(result["portfolio"]["metrics"]["Num_Trades"], 0)
            self.assertEqual(len(result["panel"]), 3)
            self.assertTrue((Path(destination) / "cointegration-tests.csv").exists())

    def test_missing_sleeve_retains_cash_instead_of_redistributing(self):
        data = prices()
        data.iloc[-1, 2] = np.nan
        panel = pd.DataFrame(
            [
                {
                    "Pair": "A-B",
                    "Ticker1": "A",
                    "Ticker2": "B",
                    "Beta": 1.0,
                    "Raw_P": 0.001,
                    "Half_Life": 10.0,
                    "Selected": True,
                },
                {
                    "Pair": "A-C",
                    "Ticker1": "A",
                    "Ticker2": "C",
                    "Beta": 1.0,
                    "Raw_P": 0.002,
                    "Half_Life": 10.0,
                    "Selected": True,
                },
            ]
        ).reindex(columns=COLUMNS)
        from evaluation import evaluate_pairs

        panel["Weight"] = 0.2
        dates = data.index[105:]
        portfolio, _, metrics = evaluate_pairs(data, dates, panel, Config(clusters=1))
        self.assertEqual(int(metrics.Status.str.startswith("cash").sum()), 1)
        values = metrics.set_index("Pair")
        expected = (
            values.loc["A-B", "Total_Return"] + values.loc["A-C", "Total_Return"]
        ) * 0.2
        self.assertAlmostEqual(portfolio["metrics"]["Total_Return"], expected)


if __name__ == "__main__":
    unittest.main()
