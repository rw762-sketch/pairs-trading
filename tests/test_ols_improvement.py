"""Synthetic tests of OLS diagnostics, causal entries, and allocated accounting."""

import unittest
import numpy as np
import pandas as pd

from ols_improvement import (RULES, VARIANTS, build_schedule, entry_filter_diagnostics,
                             generate_variant_signals, quality_diagnostics,
                             residual_half_life, run_variant, select_pairs,
                             signals_from_statistics)
from walk_forward import _window_backtest


def synthetic_prices(n=350, two_pairs=False):
    dates = pd.bdate_range('2023-01-02', periods=n)
    day = np.arange(n)
    b = 100 + .02 * day
    spread = .2 * np.sin(day)
    for offset, value in [(9, -6), (10, -5), (11, -4)]:
        spread[day % 15 == offset] = value
    prices = pd.DataFrame({'A': b + spread, 'B': b}, index=dates)
    if two_pairs:
        prices['C'] = prices.A + 20
        prices['D'] = prices.B + 20
    return prices


def metadata(two_pairs=False):
    return [{'Yahoo ticker': ticker, 'GICS Sub-Industry': ('Peer 1' if ticker in ['A', 'B'] else 'Peer 2'),
             'CIK': i + 1} for i, ticker in enumerate(['A', 'B', 'C', 'D'] if two_pairs else ['A', 'B'])]


def panel(two_pairs=False, eligible=True):
    pairs = [('A', 'B'), ('C', 'D')] if two_pairs else [('A', 'B')]
    return pd.DataFrame([{'Pair': a + '-' + b, 'Ticker1': a, 'Ticker2': b, 'Beta': 1.,
                          'Alpha': 0., 'Raw_P': .001, 'Raw_Eligible': eligible,
                          'Quality_Eligible': eligible} for a, b in pairs])


def schedule_for(prices, two_pairs=False, eligible=True):
    dates = prices.index[260:]
    labels = dates.strftime('%Y-%m')
    result = []
    for number, month in enumerate(dict.fromkeys(labels), start=1):
        days = dates[labels == month]
        location = prices.index.get_loc(days[0])
        result.append({'window': number, 'dates': days, 'formation_start': prices.index[location - 252],
                       'formation_end': prices.index[location - 1], 'panel': panel(two_pairs, eligible)})
    return result


class OLSQualityTests(unittest.TestCase):
    def test_ar1_half_life_with_intercept_has_known_analytic_value(self):
        values = [5.]
        for _ in range(59):
            values.append(1. + .8 * values[-1] + (1e-3 if len(values) == 1 else 0))
        # Exact AR(1) away from equilibrium gives an identifiable intercept/slope.
        values = [0.]
        for _ in range(59):
            values.append(1. + .8 * values[-1])
        result = residual_half_life(values)
        self.assertAlmostEqual(result['AR1_Phi'], .8, places=10)
        self.assertAlmostEqual(result['Half_Life'], -np.log(2) / np.log(.8), places=9)
        alternate = [2.]
        for _ in range(39):
            alternate.append(1. - .8 * alternate[-1])
        self.assertTrue(np.isnan(residual_half_life(alternate)['Half_Life']))
        self.assertTrue(np.isnan(residual_half_life([1.] * 60)['Half_Life']))

    def test_stable_ols_relationship_passes_and_beta_break_is_rejected(self):
        rng = np.random.default_rng(27)
        b = 100 + np.cumsum(rng.normal(0, 1., 252))
        residual = np.zeros(252)
        for i in range(1, 252):
            residual[i] = .8 * residual[i - 1] + rng.normal(0, .05)
        prices = pd.DataFrame({'A': 5 + 1.5 * b + residual, 'B': b})
        good = quality_diagnostics(prices, 'A', 'B')
        self.assertTrue(good['Quality_Eligible'])
        self.assertTrue(2 <= good['Half_Life'] <= 20)
        self.assertLess(good['Beta_Instability'], .05)
        broken = pd.DataFrame({'A': np.r_[b[:126], 4 * b[126:]], 'B': b})
        bad = quality_diagnostics(broken, 'A', 'B')
        self.assertFalse(bad['Quality_Eligible'])
        self.assertTrue(bad['Beta_Instability'] > .5 or bad['Quality_Status'] == 'nonpositive_or_invalid_full_beta')

    def test_peer_overlap_selection_is_deterministic_from_training_p_values(self):
        pairs = [('A', 'B', .04), ('A', 'C', .01), ('B', 'D', .02), ('E', 'F', .03)]
        frame = pd.DataFrame([{'Pair': a + '-' + b, 'Ticker1': a, 'Ticker2': b, 'Raw_P': p,
                               'Raw_Eligible': True, 'Quality_Eligible': True} for a, b, p in pairs])
        chosen = select_pairs(frame, 'risk_limits')
        self.assertEqual(chosen.Pair.tolist(), ['A-C', 'B-D', 'E-F'])
        tickers = list(chosen.Ticker1) + list(chosen.Ticker2)
        self.assertEqual(len(tickers), len(set(tickers)))
        self.assertEqual(len(select_pairs(frame, 'baseline')), 4)
        frame.loc[frame.Pair == 'A-C', 'Quality_Eligible'] = False
        self.assertNotIn('A-C', select_pairs(frame, 'combined').Pair.tolist())


class EntryRuleTests(unittest.TestCase):
    def test_confirmation_direction_and_cost_hurdle(self):
        valid = entry_filter_diagnostics(-2.5, -3., 2., 100., 100., 1.)
        self.assertTrue(valid['Entry_Filter_Passed'])
        self.assertAlmostEqual(valid['Convergence_Move_Per_Gross'], .02)
        self.assertAlmostEqual(valid['Estimated_Roundtrip_Cost'], .002 + .02 * 28 / 365.25 * .5)
        self.assertFalse(entry_filter_diagnostics(-3., -2.5, 2., 100., 100., 1.)['Entry_Filter_Passed'])
        self.assertFalse(entry_filter_diagnostics(2.5, -3., 2., 100., 100., 1.)['Entry_Filter_Passed'])
        self.assertFalse(entry_filter_diagnostics(-2.5, -3., .1, 100., 100., 1.)['Entry_Filter_Passed'])
        self.assertFalse(entry_filter_diagnostics(-2., -3., 2., 100., 100., 1.)['Entry_Filter_Passed'])

    def test_entry_filter_does_not_block_exit_at_exact_band(self):
        z = [np.nan, -3, -2.5, -.5, 3, 2.5, .5]
        frame = pd.DataFrame({'ZScore': z, 'Rolling_Std': 2., 'Price1': 100.,
                              'Price2': 100., 'Hedge_Ratio': 1.},
                             index=pd.bdate_range('2024-01-01', periods=len(z)))
        output = signals_from_statistics(frame, [np.nan] + z[:-1], use_entry_filter=True)
        self.assertEqual(output.Signal.tolist(), [0, 0, 1, 0, 0, -1, 0])
        self.assertFalse(output.Entry_Filter_Passed.iloc[3])
        self.assertEqual(output.Signal_Reason.iloc[3], 'mean_reversion')
        self.assertEqual(output.Signal_Reason.iloc[6], 'mean_reversion')

    def test_stop_boundary_exits_and_prevents_new_entry_beyond_stop(self):
        z = [2.5, 3.5, 3.7, 2., -2.5, -3.5]
        frame = pd.DataFrame({'ZScore': z, 'Rolling_Std': 2., 'Price1': 100.,
                              'Price2': 100., 'Hedge_Ratio': 1.},
                             index=pd.bdate_range('2024-01-01', periods=len(z)))
        output = signals_from_statistics(frame, [3.] + z[:-1], stop_zscore=3.5)
        self.assertEqual(output.Signal.tolist(), [-1, 0, 0, 0, 1, 0])
        self.assertEqual(output.Signal_Reason.iloc[1], 'stop_zscore')
        self.assertEqual(output.Signal_Reason.iloc[5], 'stop_zscore')

    def test_previous_z_exists_on_first_evaluation_day(self):
        prices = synthetic_prices()
        dates = prices.index[260:280]
        signals = generate_variant_signals(prices, dates, 'A', 'B', 1., 'entry_filter')
        self.assertTrue(np.isfinite(signals.Previous_ZScore.iloc[0]))
        first_index = prices.index.get_loc(dates[0])
        spread = prices.A - prices.B
        previous_spread = spread.iloc[first_index - 1]
        prior = spread.iloc[first_index - 61:first_index - 1]
        self.assertAlmostEqual(signals.Previous_ZScore.iloc[0], (previous_spread - prior.mean()) / prior.std())


class CausalityAndAccountingTests(unittest.TestCase):
    def test_risk_allocations_keep_idle_cash_and_reconcile(self):
        prices = synthetic_prices(two_pairs=True)
        schedule = schedule_for(prices, two_pairs=True)
        result = run_variant(prices, schedule, 'risk_limits')
        months, trades = result['monthly'], result['trades']
        self.assertGreater(len(trades), 0)
        np.testing.assert_allclose(months.Weight_Per_Pair, .2)
        np.testing.assert_allclose(months.Cash_Weight, .6)
        np.testing.assert_allclose(months.Starting_Capital.iloc[1:], months.Ending_Capital.iloc[:-1])
        self.assertTrue((trades.Bars_Held <= 10).all())
        self.assertAlmostEqual(trades.Net_PnL.sum(), result['equity_curve'].iloc[-1] - 100000)
        self.assertAlmostEqual(result['pair_profits'].Net_PnL.sum(), trades.Net_PnL.sum())
        self.assertAlmostEqual(result['metrics']['Total_Costs'], trades.Costs.sum())
        for month in months.itertuples():
            log = trades.loc[trades.Window == month.Window]
            np.testing.assert_allclose(log.Entry_Gross_Notional, month.Starting_Capital * .2)
            self.assertAlmostEqual(log.Net_PnL.sum(), month.Net_PnL)
            self.assertTrue((log.Entry_Date > month.Trading_Start).all())
            self.assertTrue((log.Exit_Date <= month.Trading_End).all())

    def test_baseline_matches_previous_monthly_raw_engine_with_same_panels(self):
        prices = synthetic_prices(two_pairs=True)
        schedule = schedule_for(prices, two_pairs=True)
        result = run_variant(prices, schedule, 'baseline')
        capital, curves, logs = 100000., [], []
        for window in schedule:
            old = _window_backtest(prices, window['dates'], window['panel'], capital,
                                   2., .5, 60, .001, .02, 20)
            curves.append(old['equity_curve'])
            logs.append(old['trades'])
            capital = old['equity_curve'].iloc[-1]
        expected = pd.concat(curves)
        np.testing.assert_allclose(result['equity_curve'], expected, rtol=1e-12, atol=1e-8)
        self.assertEqual(len(result['trades']), sum(len(log) for log in logs))
        self.assertAlmostEqual(result['trades'].Net_PnL.sum(), capital - 100000, places=7)

    def test_cash_only_windows_are_complete_and_have_no_trades(self):
        prices = synthetic_prices()
        result = run_variant(prices, schedule_for(prices, eligible=False), 'combined', initial_capital=12345)
        self.assertTrue((result['equity_curve'] == 12345).all())
        self.assertTrue((result['monthly'].Cash_Weight == 1).all())
        self.assertTrue((result['monthly'].Selected_Pairs == 0).all())
        self.assertTrue(result['trades'].empty)
        self.assertTrue(result['pair_profits'].empty)
        self.assertEqual(result['metrics']['Total_Return'], 0)
        self.assertTrue(np.isfinite(list(result['metrics'].values())).all())

    def test_future_changes_cannot_change_prior_formation_or_signals(self):
        rng = np.random.default_rng(809)
        dates = pd.bdate_range('2023-01-02', periods=345)
        b = 100 + np.cumsum(rng.normal(0, .6, len(dates)))
        residual = np.zeros(len(dates))
        for i in range(1, len(dates)):
            residual[i] = .8 * residual[i-1] + rng.normal(0, .1)
        prices = pd.DataFrame({'A': 5 + 1.2 * b + residual, 'B': b}, index=dates)
        changed = prices.copy()
        cutoff = dates[312]
        changed.loc[cutoff:, 'A'] += np.arange(len(changed.loc[cutoff:])) * 4 + 25
        before = build_schedule(prices, metadata(), dates[260])
        after = build_schedule(changed, metadata(), dates[260])
        for early, late in zip(before, after):
            if early['formation_end'] < cutoff:
                pd.testing.assert_frame_equal(early['panel'], late['panel'])
        old = run_variant(prices, before, 'baseline')
        new = run_variant(changed, after, 'baseline')
        pd.testing.assert_series_equal(old['equity_curve'].loc[:dates[311]], new['equity_curve'].loc[:dates[311]])
        for window in before:
            self.assertLess(window['formation_end'], window['dates'][0])

    def test_fold_end_is_inclusive_and_insufficient_history_is_rejected(self):
        prices = synthetic_prices()
        schedule = build_schedule(prices, metadata(), prices.index[260], prices.index[301])
        self.assertEqual(schedule[0]['dates'][0], prices.index[260])
        self.assertEqual(schedule[-1]['dates'][-1], prices.index[301])
        self.assertTrue(all(window['formation_end'] < window['dates'][0] for window in schedule))
        with self.assertRaises(ValueError):
            build_schedule(prices, metadata(), prices.index[50])
        with self.assertRaises(ValueError):
            run_variant(prices, [], 'baseline')
        self.assertEqual(set(VARIANTS), {'baseline', 'quality', 'entry_filter', 'risk_limits', 'combined'})
        self.assertEqual(RULES['formation_days'], 252)


if __name__ == '__main__':
    unittest.main()
