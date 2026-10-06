"""Recovery bounds, portfolio limits and chronological window causality."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest
import numpy as np
import pandas as pd
from config import Config
from selection import mean_crossings, apply_filters
from walk_forward import schedule_windows, run_walk_forward, exclude_previous_losses, previous_window_losers


def training():
    wave = 2 * np.sin(np.arange(80) * 1.1)
    return pd.DataFrame(
        {"A": 100 + wave, "B": 100.0, "C": 100 + 2 * wave, "D": 100.0},
        index=pd.bdate_range("2020-01-01", periods=80),
    )


def panel():
    return pd.DataFrame(
        [
            {
                "Pair": a + "-" + b,
                "Ticker1": a,
                "Ticker2": b,
                "Alpha": 0.0,
                "Beta": 1.0,
                "Raw_P": 0.001 * (i + 1),
                "Half_Life": 10.0,
                "Selected": True,
            }
            for i, (a, b) in enumerate([("A", "B"), ("A", "C"), ("C", "D")])
        ]
    )


class RecoveryTests(unittest.TestCase):
    def test_previous_loss_uses_net_aggregate_and_unordered_identity(self):
        trades = pd.DataFrame([
            {"Ticker1": "B", "Ticker2": "A", "Net_PnL": -8.0},
            {"Ticker1": "A", "Ticker2": "B", "Net_PnL": 2.0},
            {"Ticker1": "C", "Ticker2": "D", "Net_PnL": 0.0},
        ])
        losses = previous_window_losers(trades)
        self.assertEqual(losses, {("A", "B")})
        eligible, audit = exclude_previous_losses(panel(), losses, 0.2)
        self.assertEqual(set(eligible.Pair), {"A-C", "C-D"})
        self.assertEqual(audit.Previous_Window_Loser.sum(), 1)
        self.assertTrue((eligible.Weight == 0.2).all())
        self.assertEqual(previous_window_losers(trades.iloc[:0]), set())
        eligible, _ = exclude_previous_losses(panel(), set(), 0.2)
        self.assertEqual(len(eligible), 3)

    def test_crossings_are_around_mean_ignore_exact_touches(self):
        self.assertEqual(mean_crossings([-1, 0, 1, 0, -1, 0, 1]), 3)
        self.assertEqual(mean_crossings(np.array([-1, 0, 1, 0, -1, 0, 1]) + 100), 3)
        self.assertEqual(mean_crossings([4, 4, 4]), 0)

    def test_bounds_crossings_and_shared_stock_limit(self):
        result = apply_filters(panel(), training(), Config())
        self.assertTrue(result.iloc[0].Selected)
        self.assertFalse(result.iloc[1].Selected)
        selected = result[result.Selected]
        names = list(selected.Ticker1) + list(selected.Ticker2)
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue((selected.Weight <= 0.2).all())
        self.assertLessEqual(selected.Weight.sum(), 1)
        changed = panel()
        changed["Half_Life"] = [0.9, Config().half_life_max + 0.1, 10.0]
        result = apply_filters(changed, training(), Config())
        self.assertFalse(result.iloc[:2].Selected.any())
        constant = training() * 0 + 100
        self.assertFalse(apply_filters(panel(), constant, Config()).Selected.any())

    def test_unused_early_missing_quotes_do_not_block_later_evaluation(self):
        from evaluation import evaluate_pairs

        data = training()
        data.iloc[0, 0] = np.nan
        chosen = panel().iloc[:1].copy()
        chosen["Weight"] = 0.2
        _, _, metrics = evaluate_pairs(
            data, data.index[60:], chosen, replace(Config(), lookback=10)
        )
        self.assertEqual(metrics.Status.iloc[0], "evaluated")

    def test_schedule_is_nonoverlapping_and_strictly_after_formation(self):
        index = pd.bdate_range("2020-01-01", periods=173)
        schedule = schedule_windows(index, 60, 40)
        combined = pd.DatetimeIndex([])
        for form, trade in schedule:
            self.assertLess(form[-1], trade[0])
            self.assertEqual(len(form), 60)
            combined = combined.append(trade)
        pd.testing.assert_index_equal(combined, index[60:])

    def test_chained_account_and_future_changes_cannot_change_earlier_window(self):
        rng = np.random.default_rng(6)
        values = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (140, 4)), axis=0))
        prices = pd.DataFrame(
            values,
            index=pd.bdate_range("2020-01-01", periods=140),
            columns=list("ABCD"),
        )
        config = replace(
            Config(),
            formation=60,
            trading_window=40,
            lookback=10,
            min_crossings=0.0,
            clusters=1,
        )

        def screen(train, family, *args):
            out = panel()
            out["Status"] = "tested"
            return out, pd.DataFrame()

        with (
            TemporaryDirectory() as one,
            TemporaryDirectory() as two,
            patch("walk_forward.screen_pairs", side_effect=screen),
            patch("walk_forward.write_walk_forward_report"),
        ):
            first = run_walk_forward(prices, config, one)
            modified = prices.copy()
            modified.iloc[100:] *= np.arange(1, 5) * 20
            second = run_walk_forward(modified, config, two)
            for variant in ["baseline", "filtered", "previous_loss_filter"]:
                pd.testing.assert_series_equal(
                    first["portfolios"][variant]["equity_curve"].iloc[:40],
                    second["portfolios"][variant]["equity_curve"].iloc[:40],
                )
                curve = first["portfolios"][variant]["equity_curve"]
                trades = first["portfolios"][variant]["trades"]
                self.assertAlmostEqual(
                    trades.Net_PnL.sum(), curve.iloc[-1] - config.capital
                )
            rows = first["windows"].query('Variant == "filtered"')
            self.assertAlmostEqual(
                rows.Starting_Capital.iloc[1], rows.Ending_Capital.iloc[0]
            )
            self.assertTrue((Path(one) / "plan.json").exists())


if __name__ == "__main__":
    unittest.main()
