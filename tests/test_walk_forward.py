"""Causal selection, full-family correction, and account-carry regression tests."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from walk_forward import (holm_correction, prescribe_pair_family, run_experiment,
                          screen_formation)


def metadata(tickers=('A', 'B')):
    return [{'Yahoo ticker': ticker, 'GICS Sub-Industry': 'Peers', 'CIK': number + 1}
            for number, ticker in enumerate(tickers)]


def oscillating_prices(n=150):
    index = pd.bdate_range('2024-01-01', periods=n)
    days = np.arange(n)
    b = 100 + .02 * days
    return pd.DataFrame({'A': b + 3 * np.sin(days * .8), 'B': b}, index=index)


def eligible_screen(formation, family, alpha=.05):
    out = family.copy()
    out['Beta'] = 1.
    out['Raw_P'] = .001
    out['Correction_Input_P'] = .001
    out['Adjusted_P'] = .001
    out['I1Compatible'] = True
    out['Raw_Eligible'] = True
    out['Holm_Eligible'] = True
    out['Test_Status'] = 'tested'
    return out


class FamilyCorrectionTests(unittest.TestCase):
    def test_holm_known_values_keep_untested_hypothesis_in_family(self):
        result = holm_correction([.01, .04, .03, np.nan])
        np.testing.assert_allclose(result['correction_input'], [.01, .04, .03, 1.])
        np.testing.assert_allclose(result['adjusted'], [.04, .09, .09, 1.])
        np.testing.assert_array_equal(result['reject'], [True, False, False, False])
        np.testing.assert_array_equal(result['is_placeholder'], [False, False, False, True])
        with self.assertRaises(ValueError):
            holm_correction([.1, np.nan], testable=[True, True])

    def test_prescribed_family_depends_only_on_peers_and_issuer_metadata(self):
        rows = metadata(('A', 'B', 'C', 'D'))
        rows[2]['CIK'] = rows[0]['CIK']
        rows[3]['GICS Sub-Industry'] = 'Different peers'
        family = prescribe_pair_family(rows)
        self.assertEqual(family.Pair.tolist(), ['A-B', 'B-C'])

    def test_untestable_formation_retains_nan_observation_and_explicit_one_input(self):
        prices = pd.DataFrame({'A': [100.] * 60, 'B': [50.] * 60},
                              index=pd.bdate_range('2024-01-01', periods=60))
        panel = screen_formation(prices, prescribe_pair_family(metadata()))
        self.assertEqual(len(panel), 1)
        self.assertTrue(panel.Raw_P.isna().all())
        self.assertEqual(panel.Correction_Input_P.iloc[0], 1.)
        self.assertTrue(panel.Correction_Input_Is_Placeholder.iloc[0])
        self.assertEqual(panel.Adjusted_P.iloc[0], 1.)
        self.assertFalse(panel.Raw_Eligible.iloc[0])
        self.assertFalse(panel.Holm_Eligible.iloc[0])
        self.assertEqual(panel.Family_Size.iloc[0], 1)


class WalkForwardTests(unittest.TestCase):
    def test_no_pairs_keeps_all_dates_as_cash_with_finite_zero_metrics(self):
        prices = pd.DataFrame({'A': [100.] * 130, 'B': [50.] * 130},
                              index=pd.bdate_range('2024-01-01', periods=130))
        result = run_experiment(prices, metadata(), evaluation_start=prices.index[61],
                                formation_days=40, lookback=10, initial_capital=12345)
        for variant in result['variants'].values():
            self.assertTrue(variant['equity_curve'].index.equals(prices.index[61:]))
            self.assertTrue((variant['equity_curve'] == 12345).all())
            self.assertTrue(variant['monthly'].Cash_Only.all())
            self.assertTrue((variant['monthly'].Selected_Pairs == 0).all())
            self.assertTrue(variant['trades'].empty)
            self.assertTrue(variant['pair_profits'].empty)
            self.assertEqual(variant['metrics']['Total_Return'], 0)
            self.assertEqual(variant['metrics']['Sharpe_Ratio'], 0)
            self.assertTrue(np.isfinite(list(variant['metrics'].values())).all())

    def test_future_price_perturbation_cannot_change_earlier_selections_or_equity(self):
        rng = np.random.default_rng(801)
        dates = pd.bdate_range('2024-01-01', periods=165)
        b = 100 + np.cumsum(rng.normal(0, .4, len(dates)))
        prices = pd.DataFrame({'A': 5 + 1.2 * b + rng.normal(0, .2, len(dates)), 'B': b}, index=dates)
        changed = prices.copy()
        cutoff = dates[130]
        changed.loc[cutoff:, 'A'] += np.arange(len(changed.loc[cutoff:])) * 3 + 40
        settings = dict(evaluation_start=dates[65], formation_days=60, lookback=10)
        before = run_experiment(prices, metadata(), **settings)
        after = run_experiment(changed, metadata(), **settings)
        # A month's selection is known before that month's first trading close.
        known_before = before['screening'].Formation_End < cutoff
        pd.testing.assert_frame_equal(before['screening'].loc[known_before].reset_index(drop=True),
                                      after['screening'].loc[known_before].reset_index(drop=True))
        for name in before['variants']:
            pd.testing.assert_series_equal(before['variants'][name]['equity_curve'].loc[:dates[129]],
                                           after['variants'][name]['equity_curve'].loc[:dates[129]])
        self.assertTrue((before['screening'].Formation_End < before['screening'].Trading_Start).all())

    def test_capital_chains_and_allocated_trades_reconcile_across_months(self):
        prices = oscillating_prices()
        with patch('walk_forward.screen_formation', side_effect=eligible_screen):
            result = run_experiment(prices, metadata(), evaluation_start=prices.index[61],
                                    formation_days=40, lookback=10, initial_capital=10000,
                                    entry=.8, exit=.25, fee=.001, borrow=.02, max_hold=3)
        for name, variant in result['variants'].items():
            months, trades = variant['monthly'], variant['trades']
            self.assertGreater(len(trades), 0)
            self.assertNotAlmostEqual(months.Ending_Capital.iloc[0], 10000)
            np.testing.assert_allclose(months.Starting_Capital.iloc[1:], months.Ending_Capital.iloc[:-1])
            self.assertAlmostEqual(trades.Net_PnL.sum(), variant['equity_curve'].iloc[-1] - 10000)
            self.assertAlmostEqual(variant['pair_profits'].Net_PnL.sum(), trades.Net_PnL.sum())
            self.assertAlmostEqual(trades.Costs.sum(), variant['metrics']['Total_Costs'])
            self.assertTrue((trades.Bars_Held <= 3).all())
            self.assertTrue((pd.to_datetime(trades.Entry_Date).dt.strftime('%Y-%m') ==
                             pd.to_datetime(trades.Exit_Date).dt.strftime('%Y-%m')).all())
            for month in months.itertuples():
                log = trades.loc[trades.Window == month.Window]
                self.assertAlmostEqual(log.Net_PnL.sum(), month.Net_PnL)
                np.testing.assert_allclose(log.Entry_Gross_Notional, month.Starting_Capital)
                self.assertTrue((log.Entry_Date > month.Trading_Start).all())
                first_value = variant['equity_curve'].loc[month.Trading_Start]
                self.assertAlmostEqual(first_value, month.Starting_Capital)
            for trade in trades.itertuples():
                if trade.Exit_Reason == 'month_end':
                    self.assertEqual(trade.Exit_Date, trade.Trading_End)
                    self.assertLess(trade.Exit_Date, prices.index[-1])
        pd.testing.assert_series_equal(result['variants']['fixed_selection']['equity_curve'],
                                       result['variants']['monthly_raw']['equity_curve'])

    def test_fixed_model_freezes_first_beta_while_monthly_uses_new_past_fit(self):
        prices = oscillating_prices()
        calls = []
        def changing_screen(formation, family, alpha=.05):
            calls.append(formation.index[-1])
            out = eligible_screen(formation, family, alpha)
            out['Beta'] = 1 + .05 * (len(calls) - 1)
            return out
        with patch('walk_forward.screen_formation', side_effect=changing_screen):
            result = run_experiment(prices, metadata(), evaluation_start=prices.index[61],
                                    formation_days=40, lookback=10, entry=.8, exit=.25, max_hold=3)
        fixed = result['variants']['fixed_selection']
        self.assertTrue((fixed['trades'].Hedge_Ratio == 1).all())
        self.assertEqual(fixed['monthly'].Formation_End.nunique(), 1)
        monthly = result['variants']['monthly_raw']
        self.assertGreater(monthly['monthly'].Formation_End.nunique(), 1)
        for trade in monthly['trades'].itertuples():
            self.assertAlmostEqual(trade.Hedge_Ratio, 1 + .05 * (trade.Window - 1))

    def test_two_pairs_share_one_monthly_budget(self):
        prices = oscillating_prices()
        prices['C'] = prices.A + 20
        prices['D'] = prices.B + 20
        rows = metadata(('A', 'B', 'C', 'D'))
        for row in rows[2:]:
            row['GICS Sub-Industry'] = 'Other peers'
        with patch('walk_forward.screen_formation', side_effect=eligible_screen):
            result = run_experiment(prices, rows, evaluation_start=prices.index[61],
                                    formation_days=40, lookback=10, entry=.8, exit=.25, max_hold=3)
        self.assertEqual(result['config']['family_size'], 2)
        variant = result['variants']['monthly_raw']
        self.assertTrue((variant['monthly'].Selected_Pairs == 2).all())
        for month in variant['monthly'].itertuples():
            log = variant['trades'].loc[variant['trades'].Window == month.Window]
            np.testing.assert_allclose(log.Entry_Gross_Notional, month.Starting_Capital / 2)
        self.assertAlmostEqual(variant['trades'].Net_PnL.sum(), variant['equity_curve'].iloc[-1] - 100000)

    def test_reject_insufficient_formation_and_incomplete_metadata(self):
        prices = oscillating_prices()
        with self.assertRaises(ValueError):
            run_experiment(prices, metadata(), evaluation_start=prices.index[10], formation_days=40)
        broken = metadata()
        del broken[0]['CIK']
        with self.assertRaises(ValueError):
            run_experiment(prices, broken)


if __name__ == '__main__':
    unittest.main()
