"""Deterministic cash/quantity reconciliations and no-lookahead regression tests."""

import unittest
import numpy as np
import pandas as pd

from backtester import PairBacktester, backtest_all_pairs, combine_portfolio_results
from signal_generation import PairSignalGenerator, generate_all_signals


def frame(p1=None, p2=None, signal=None, beta=2.0):
    p1 = [80, 100, 110, 115, 120] if p1 is None else p1
    p2 = [40, 50, 52, 55, 60] if p2 is None else p2
    signal = [1, 1, 0, 0, 0] if signal is None else signal
    result = pd.DataFrame({'Price1': p1, 'Price2': p2, 'Signal': signal,
                           'Hedge_Ratio': beta}, index=pd.bdate_range('2024-01-01', periods=len(p1)))
    result['Spread'] = result.Price1 - beta * result.Price2
    result.attrs.update(ticker1='A', ticker2='B', hedge_ratio=beta)
    return result


class TwoLegAccountingTests(unittest.TestCase):
    def test_hand_calculated_two_leg_roundtrip_and_next_close(self):
        s = frame()
        engine = PairBacktester(s, initial_capital=1000, transaction_cost=0.001)
        curve = engine.run_backtest()
        trade = engine.get_trades_dataframe().iloc[0]
        # Day 0 is a signal only. At day 1 buy 5 A @100 and short 10 B @50.
        self.assertEqual(trade.Entry_Date, s.index[1])
        self.assertEqual(trade.Exit_Date, s.index[3])
        self.assertAlmostEqual(trade.Quantity1, 5)
        self.assertAlmostEqual(trade.Quantity2, -10)
        self.assertAlmostEqual(trade.Entry_Gross_Notional, 1000)
        self.assertAlmostEqual(trade.Gross_PnL, 5 * (115 - 100) - 10 * (55 - 50))
        self.assertAlmostEqual(trade.Entry_Cost, 1)
        self.assertAlmostEqual(trade.Exit_Cost, 1.125)
        self.assertAlmostEqual(trade.Net_PnL, 22.875)
        self.assertAlmostEqual(trade.Trade_Return, 0.022875)
        np.testing.assert_allclose(curve.Portfolio_Value, [1000, 999, 1029, 1022.875, 1022.875])
        np.testing.assert_allclose(curve.Portfolio_Value,
                                   curve.Cash + curve.Quantity1 * s.Price1 + curve.Quantity2 * s.Price2)
        metrics = engine.get_performance_metrics()
        self.assertEqual(metrics['Num_Trades'], 1)
        self.assertAlmostEqual(metrics['Total_Transaction_Costs'], 2.125)
        self.assertAlmostEqual(metrics['Max_Drawdown'], 1022.875 / 1029 - 1)
        self.assertAlmostEqual(metrics['Turnover'], 2125 / 1000 / (4 / 365.25))

    def test_short_spread_pnl_sign(self):
        s = frame(signal=[-1, -1, 0, 0, 0])
        engine = PairBacktester(s, 1000, 0.001)
        engine.run_backtest()
        trade = engine.get_trades_dataframe().iloc[0]
        self.assertAlmostEqual(trade.Quantity1, -5)
        self.assertAlmostEqual(trade.Quantity2, 10)
        self.assertAlmostEqual(trade.Net_PnL, -27.125)

    def test_negative_beta_means_same_sign_legs(self):
        engine = PairBacktester(frame(beta=-2), 1000, 0.001)
        curve = engine.run_backtest()
        trade = engine.get_trades_dataframe().iloc[0]
        self.assertAlmostEqual(trade.Quantity1, 5)
        self.assertAlmostEqual(trade.Quantity2, 10)
        self.assertAlmostEqual(trade.Gross_PnL, 125)
        self.assertAlmostEqual(curve.Portfolio_Value.iloc[-1], 1122.875)

    def test_net_loser_not_counted_as_winner_and_entry_drawdown(self):
        s = frame(p1=[100, 100, 100.2, 100.2], p2=[50] * 4, signal=[1, 0, 0, 0])
        engine = PairBacktester(s, 1000, 0.001)
        engine.run_backtest()
        m = engine.get_performance_metrics()
        self.assertEqual(m['Win_Rate'], 0)
        self.assertAlmostEqual(m['Avg_Trade_PnL'], -1.001)
        self.assertAlmostEqual(m['Total_Transaction_Costs'], 2.001)
        self.assertAlmostEqual(m['Max_Drawdown'], -0.001001)

    def test_final_liquidation_costs_and_reset(self):
        s = frame(signal=[1] * 5)
        engine = PairBacktester(s, 1000, 0.001)
        first = engine.run_backtest().copy()
        trade = engine.get_trades_dataframe().iloc[0]
        self.assertEqual(trade.Exit_Reason, 'end_of_data')
        self.assertEqual(trade.Exit_Date, s.index[-1])
        self.assertAlmostEqual(trade.Exit_Cost, 1.2)
        self.assertEqual(first.Position.iloc[-1], 0)
        pd.testing.assert_frame_equal(first, engine.run_backtest())
        self.assertEqual(len(engine.trades), 1)
        self.assertAlmostEqual(first.Portfolio_Value.iloc[-1] - 1000, trade.Net_PnL)

    def test_no_entry_on_last_bar(self):
        engine = PairBacktester(frame(signal=[0, 0, 0, 1, 1]), 1000, 0.001)
        engine.run_backtest()
        self.assertEqual(len(engine.trades), 0)
        self.assertEqual(engine.get_performance_metrics()['Final_Portfolio_Value'], 1000)

    def test_borrow_uses_short_notional_and_calendar_days(self):
        s = frame(p1=[100] * 4, p2=[50] * 4, signal=[1, 1, 0, 0])
        s.index = pd.to_datetime(['2024-01-04', '2024-01-05', '2024-01-08', '2024-01-09'])
        engine = PairBacktester(s, 1000, 0, annual_borrow_rate=0.36525)
        curve = engine.run_backtest()
        trade = engine.get_trades_dataframe().iloc[0]
        # $500 short at 36.525% p.a. held Fri to Tue: 4 calendar days = $2.
        self.assertAlmostEqual(trade.Borrow_Cost, 2)
        self.assertAlmostEqual(trade.Costs, 2)
        self.assertAlmostEqual(curve.Portfolio_Value.iloc[-1], 998)
        self.assertAlmostEqual(engine.get_performance_metrics()['Total_Costs'], 2)
        self.assertEqual(engine.get_performance_metrics()['Total_Transaction_Costs'], 0)

    def test_max_holding_counts_bars_and_does_not_reenter_stale_signal(self):
        s = frame(p1=[100] * 7, p2=[50] * 7, signal=[1] * 7)
        engine = PairBacktester(s, 1000, 0, max_holding_period=2)
        engine.run_backtest()
        trade = engine.get_trades_dataframe().iloc[0]
        self.assertEqual(len(engine.trades), 1)
        self.assertEqual(trade.Entry_Date, s.index[1])
        self.assertEqual(trade.Exit_Date, s.index[3])
        self.assertEqual(trade.Bars_Held, 2)
        self.assertEqual(trade.Exit_Reason, 'max_holding_period')

    def test_reversal_is_two_closed_roundtrips_and_reconciles(self):
        engine = PairBacktester(frame(signal=[1, -1, -1, 0, 0]), 1000, 0.001)
        curve = engine.run_backtest()
        trades = engine.get_trades_dataframe()
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades.iloc[0].Exit_Reason, 'signal_reversal')
        self.assertEqual(trades.iloc[0].Exit_Date, trades.iloc[1].Entry_Date)
        self.assertAlmostEqual(trades.Net_PnL.sum(), curve.Portfolio_Value.iloc[-1] - 1000)
        self.assertAlmostEqual(trades.Costs.sum(), curve.Transaction_Costs.sum())

    def test_empty_flat_and_bad_data(self):
        empty = frame().iloc[:0]
        engine = PairBacktester(empty, 1000)
        self.assertTrue(engine.run_backtest().empty)
        self.assertEqual(engine.get_performance_metrics()['Final_Portfolio_Value'], 1000)
        self.assertTrue(np.isfinite(list(engine.get_performance_metrics().values())).all())
        flat = PairBacktester(frame(signal=[0] * 5), 1000)
        flat.run_backtest()
        self.assertEqual(flat.get_performance_metrics()['Sharpe_Ratio'], 0)
        invalid_frames = []
        missing = frame()
        missing.loc[missing.index[2], 'Price1'] = np.nan
        invalid_frames.append(missing)
        invalid_frames.append(frame().iloc[::-1])
        duplicate = frame()
        duplicate.index = [duplicate.index[0]] * len(duplicate)
        invalid_frames.append(duplicate)
        for invalid in invalid_frames:
            with self.assertRaises(ValueError):
                PairBacktester(invalid).run_backtest()

    def test_equal_weight_portfolio_scales_equity_trades_and_fees_to_one_account(self):
        first = backtest_all_pairs({'A-B': frame()}, 1000, 0.001)
        second = backtest_all_pairs({'C-D': frame(signal=[-1, -1, 0, 0, 0])}, 2000, 0.001)
        portfolio = combine_portfolio_results(first | second, initial_capital=1000)
        self.assertAlmostEqual(portfolio['trades'].Entry_Gross_Notional.sum(), 1000)
        self.assertAlmostEqual(portfolio['metrics']['Total_Transaction_Costs'], 2.125)
        self.assertAlmostEqual(portfolio['equity_curve'].iloc[-1], 997.875)
        self.assertAlmostEqual(portfolio['trades'].Net_PnL.sum(), -2.125)
        self.assertAlmostEqual(portfolio['metrics']['Win_Rate'], 0.5)
        with self.assertRaises(ValueError):
            combine_portfolio_results(first | second, equal_weight=False)
        explicit = combine_portfolio_results(first | second, equal_weight=False,
                                              weights={'A-B': 1, 'C-D': 0})
        self.assertAlmostEqual(explicit['equity_curve'].iloc[-1], 1022.875)
        self.assertEqual(len(explicit['trades']), 1)


class SignalTimingTests(unittest.TestCase):
    def test_zscore_uses_prior_window_and_requested_lookback(self):
        prices = pd.DataFrame({'A': [11, 12, 13, 14, 30, 16], 'B': [10] * 6},
                              index=pd.bdate_range('2024-01-01', periods=6))
        generator = PairSignalGenerator(prices, 'A', 'B', hedge_ratio=1, lookback=3)
        result = generator.get_signals()
        self.assertTrue(result.ZScore.iloc[:3].isna().all())
        # Prior spreads [1,2,3]: mean2, sample std1. Today spread4 => z2.
        self.assertAlmostEqual(result.ZScore.iloc[3], 2)
        self.assertAlmostEqual(result.Rolling_Mean.iloc[4], 3)
        self.assertAlmostEqual(result.ZScore.iloc[4], 17)
        all_signals = generate_all_signals(prices, [{'ticker1': 'A', 'ticker2': 'B', 'hedge_ratio': 1}], lookback=3)
        pd.testing.assert_series_equal(result.ZScore, all_signals['A-B'].ZScore)

    def test_estimated_beta_does_not_use_future_and_warmup_is_flat(self):
        prices = pd.DataFrame({'A': [21, 23, 25, 27, 29, 31, 33], 'B': [10, 11, 12, 13, 14, 15, 16]},
                              index=pd.bdate_range('2024-01-01', periods=7))
        changed = prices.copy()
        changed.loc[changed.index[5:], 'A'] = [1000, 2000]
        before = PairSignalGenerator(prices, 'A', 'B', lookback=3)
        after = PairSignalGenerator(changed, 'A', 'B', lookback=3)
        self.assertAlmostEqual(before.hedge_ratio, 2)
        self.assertAlmostEqual(before.hedge_ratio, after.hedge_ratio)
        pd.testing.assert_series_equal(before.get_signals().ZScore.iloc[:5], after.get_signals().ZScore.iloc[:5])
        self.assertTrue((before.get_signals().Signal.iloc[:3] == 0).all())

    def test_symmetric_exit_band_and_exact_threshold(self):
        prices = pd.DataFrame({'A': [100] * 7, 'B': [50] * 7}, index=pd.bdate_range('2024-01-01', periods=7))
        generator = PairSignalGenerator(prices, 'A', 'B', hedge_ratio=1, lookback=2)
        generator.calculate_zscore()
        generator.z_score = pd.Series([-2.5, -0.7, -0.5, 2.5, 0.7, 0.5, 0], index=prices.index)
        result = generator.get_signals(entry_threshold=2, exit_threshold=0.5)
        self.assertEqual(result.Signal.tolist(), [1, 1, 0, -1, -1, 0, 0])

    def test_prior_history_can_warm_up_evaluation_with_flat_start(self):
        prices = pd.DataFrame({'A': [11, 12, 13, 14, 30, 16], 'B': [10] * 6},
                              index=pd.bdate_range('2024-01-01', periods=6))
        generator = PairSignalGenerator(prices, 'A', 'B', 1, 3)
        generator.calculate_zscore()
        for attr in ['price_data', 'spread', 'z_score', 'moving_mean', 'moving_std']:
            setattr(generator, attr, getattr(generator, attr).iloc[4:])
        result = generator.get_signals()
        self.assertAlmostEqual(result.ZScore.iloc[0], 17)
        self.assertEqual(result.Signal.iloc[0], -1)
        # The new sample starts flat: first signal can execute only on its next bar.
        engine = PairBacktester(result, 1000)
        self.assertEqual(engine.run_backtest().Position.iloc[0], 0)


if __name__ == '__main__':
    unittest.main()
