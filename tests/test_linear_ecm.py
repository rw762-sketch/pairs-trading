"""Synthetic numerical and causal checks for the additional linear ECM gate."""

import unittest
import numpy as np
import pandas as pd

from linear_ecm import (MODEL_COLUMNS, apply_forecast_gate, build_ecm_schedule,
                         companion_stability, fit_spread_ecm, forecast_residual,
                         gate_diagnostics, run_ecm_variant)
from ols_improvement import run_variant


def model(a=0., k=-.2, g=0.):
    result = {name: np.nan for name in MODEL_COLUMNS}
    result.update(ECM_Intercept=a, ECM_Level_Coeff=k, ECM_Delta_Coeff=g,
                  ECM_Rank=3, ECM_Observations=250, ECM_Valid=True, ECM_Status='valid')
    return result


def fixture():
    dates = pd.bdate_range('2023-01-02', periods=350)
    day = np.arange(len(dates))
    b = 100 + .02 * day
    residual = .2 * np.sin(day)
    for offset, value in [(9, -6), (10, -5), (11, -4)]:
        residual[day % 15 == offset] = value
    prices = pd.DataFrame({'A': b + residual, 'B': b}, index=dates)
    evaluation = dates[260:]
    labels = evaluation.strftime('%Y-%m')
    schedule = []
    for month in dict.fromkeys(labels):
        trading = evaluation[labels == month]
        location = dates.get_loc(trading[0])
        panel = pd.DataFrame([{'Pair': 'A-B', 'Ticker1': 'A', 'Ticker2': 'B', 'Alpha': 0.,
                               'Beta': 1., 'Raw_P': .001, 'Raw_Eligible': True,
                               'Quality_Eligible': True, **model()}])
        schedule.append({'dates': trading, 'formation_start': dates[location - 252],
                         'formation_end': dates[location - 1], 'panel': panel})
    return prices, schedule


class ECMNumericsTests(unittest.TestCase):
    def test_known_linear_equation_recovers_ols_coefficients(self):
        a, k, g = .1, -.2, .15
        residual = [8., 6.3]
        for _ in range(78):
            change = a + k * residual[-1] + g * (residual[-1] - residual[-2])
            residual.append(residual[-1] + change)
        fitted = fit_spread_ecm(residual)
        self.assertTrue(fitted['ECM_Valid'])
        self.assertEqual(fitted['ECM_Rank'], 3)
        self.assertAlmostEqual(fitted['ECM_Intercept'], a, places=10)
        self.assertAlmostEqual(fitted['ECM_Level_Coeff'], k, places=10)
        self.assertAlmostEqual(fitted['ECM_Delta_Coeff'], g, places=10)
        self.assertLess(fitted['ECM_Residual_RMSE'], 1e-10)
        constant = fit_spread_ecm([1.] * 252)
        self.assertFalse(constant['ECM_Valid'])
        self.assertEqual(constant['ECM_Status'], 'rank_deficient')

    def test_companion_stability_convention_and_boundary(self):
        stable = companion_stability(-.2, .1)
        self.assertTrue(stable['stable'])
        self.assertAlmostEqual(stable['AR2_Phi1'], .9)
        self.assertAlmostEqual(stable['AR2_Phi2'], -.1)
        self.assertTrue((np.abs(stable['roots']) < 1).all())
        # Roots of 1 - phi1*z - phi2*z^2 are reciprocal companion eigenvalues.
        ar_roots = np.roots([-stable['AR2_Phi2'], -stable['AR2_Phi1'], 1.])
        self.assertTrue((np.abs(ar_roots) > 1).all())
        self.assertFalse(companion_stability(0., 0.)['stable'])
        self.assertFalse(companion_stability(.2, .1)['stable'])
        self.assertFalse(companion_stability(-.2, 1.2)['stable'])

    def test_forecast_excludes_the_execution_gap(self):
        fitted = model()
        path = forecast_residual(fitted, 10., -1., 6)
        np.testing.assert_allclose(path, 10 * .8 ** np.arange(1, 7))
        gate = gate_diagnostics(fitted, 10., -1., 2.5, 100., 100., 1.)
        expected = 8. - 10 * .8 ** 6
        self.assertAlmostEqual(gate['Projected_Signed_Move'], expected)
        self.assertNotAlmostEqual(gate['Projected_Signed_Move'], 10 - 10 * .8 ** 6)
        self.assertAlmostEqual(gate['Projected_Move_Per_Gross'], expected / 200)
        self.assertTrue(gate['Entry_Gate_Passed'])

    def test_costs_use_the_correct_short_leg_and_wrong_direction_is_blocked(self):
        short = gate_diagnostics(model(), 10., -1., 2.5, 150., 50., 1.)
        long = gate_diagnostics(model(), -10., 1., -2.5, 150., 50., 1.)
        self.assertAlmostEqual(short['Estimated_Roundtrip_Cost'], .002 + .02 * 7 / 365.25 * .75)
        self.assertAlmostEqual(long['Estimated_Roundtrip_Cost'], .002 + .02 * 7 / 365.25 * .25)
        wrong = gate_diagnostics(model(), 10., -1., -2.5, 150., 50., 1.)
        self.assertLess(wrong['Projected_Signed_Move'], 0)
        self.assertFalse(wrong['Entry_Gate_Passed'])
        self.assertFalse(gate_diagnostics(model(), .01, 0., 2.5, 150., 50., 1.)['Entry_Gate_Passed'])
        self.assertFalse(gate_diagnostics(dict(model(), ECM_Valid=False), 10., 0., 2.5, 150., 50., 1.)['Entry_Gate_Passed'])

    def test_gate_never_blocks_standard_exit(self):
        z = [-2.5, -.5, 2.5, .5]
        frame = pd.DataFrame({'ZScore': z, 'Price1': 100., 'Price2': 100., 'Hedge_Ratio': 1.},
                             index=pd.bdate_range('2024-01-01', periods=4))
        result = apply_forecast_gate(frame, [-10., -1., 10., 1.], [1., 0., -1., 0.], model())
        self.assertEqual(result.Signal.tolist(), [1, 0, -1, 0])
        self.assertFalse(result.Entry_Gate_Passed.iloc[1])
        self.assertEqual(result.Signal_Reason.iloc[1], 'mean_reversion')
        self.assertEqual(result.Signal_Reason.iloc[3], 'mean_reversion')


class ECMIntegrationTests(unittest.TestCase):
    def test_account_carries_and_every_execution_has_a_prior_gate(self):
        prices, schedule = fixture()
        result = run_ecm_variant(prices, schedule)
        self.assertGreater(len(result['trades']), 0)
        self.assertAlmostEqual(result['trades'].Net_PnL.sum(), result['equity_curve'].iloc[-1] - 100000)
        self.assertAlmostEqual(result['pair_profits'].Net_PnL.sum(), result['trades'].Net_PnL.sum())
        self.assertAlmostEqual(result['trades'].Costs.sum(), result['metrics']['Total_Costs'])
        months = result['monthly']
        np.testing.assert_allclose(months.Starting_Capital.iloc[1:], months.Ending_Capital.iloc[:-1])
        for month in months.itertuples():
            trades = result['trades'].loc[result['trades'].Window == month.Window]
            np.testing.assert_allclose(trades.Entry_Gross_Notional, month.Starting_Capital)
            self.assertTrue((trades.Entry_Date > month.Trading_Start).all())
            self.assertTrue((trades.Exit_Date <= month.Trading_End).all())
        forecasts = result['forecast_diagnostics']
        for trade in result['trades'].itertuples():
            dates = schedule[trade.Window - 1]['dates']
            before = dates[dates.get_loc(trade.Entry_Date) - 1]
            gate = forecasts.loc[(forecasts.Window == trade.Window) & (forecasts.Date == before)].iloc[0]
            self.assertTrue(gate.Entry_Gate_Passed)

    def test_baseline_is_identical_and_invalid_models_leave_allocations_as_cash(self):
        prices, schedule = fixture()
        actual = run_ecm_variant(prices, schedule, 'baseline')
        expected = run_variant(prices, schedule, 'baseline')
        pd.testing.assert_series_equal(actual['equity_curve'], expected['equity_curve'])
        pd.testing.assert_frame_equal(actual['trades'], expected['trades'])
        for window in schedule:
            window['panel']['ECM_Valid'] = False
        cash = run_ecm_variant(prices, schedule)
        self.assertTrue((cash['equity_curve'] == 100000).all())
        self.assertTrue(cash['trades'].empty)
        self.assertEqual(cash['metrics']['Total_Return'], 0.)

    def test_future_prices_cannot_change_formation_models_or_earlier_forecasts(self):
        rng = np.random.default_rng(332)
        dates = pd.bdate_range('2023-01-02', periods=345)
        b = 100 + np.cumsum(rng.normal(0, .5, len(dates)))
        s = np.zeros(len(dates))
        for i in range(2, len(dates)):
            s[i] = s[i-1] - .2 * s[i-1] + .1 * (s[i-1] - s[i-2]) + rng.normal(0, .12)
        prices = pd.DataFrame({'A': 5 + 1.2 * b + s, 'B': b}, index=dates)
        metadata = [{'Yahoo ticker': t, 'GICS Sub-Industry': 'Peers', 'CIK': i+1} for i,t in enumerate(['A','B'])]
        changed = prices.copy()
        cutoff = dates[312]
        changed.loc[cutoff:, 'A'] += 50 + np.arange(len(changed.loc[cutoff:])) * 5
        before = build_ecm_schedule(prices, metadata, dates[260])
        after = build_ecm_schedule(changed, metadata, dates[260])
        for a,b in zip(before,after):
            if a['formation_end'] < cutoff:
                pd.testing.assert_frame_equal(a['panel'],b['panel'])
        earlier = run_ecm_variant(prices,before)
        later = run_ecm_variant(changed,after)
        pd.testing.assert_series_equal(earlier['equity_curve'].loc[:dates[311]],later['equity_curve'].loc[:dates[311]])
        a,b=earlier['forecast_diagnostics'],later['forecast_diagnostics']
        if len(a):
            pd.testing.assert_frame_equal(a.loc[a.Date<cutoff].reset_index(drop=True),
                                          b.loc[b.Date<cutoff].reset_index(drop=True))


if __name__ == '__main__':
    unittest.main()
