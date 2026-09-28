import unittest
import numpy as np
import pandas as pd
from multiple_regression import FEATURES, features, training_data, fit_model, predict, apply_gate, RegressionGate
from ols_improvement import run_variant
from tests.test_linear_ecm import fixture


class MultipleRegressionTests(unittest.TestCase):
    def test_known_coefficients_and_rank_rejection(self):
        rng = np.random.default_rng(8)
        x = pd.DataFrame(rng.normal(size=(200, 5)), columns=FEATURES)
        coefficients = np.array([.1, -.2, .3, .4, -.5])
        training = x.assign(target=.03 + x.to_numpy() @ coefficients)
        model = fit_model(training)
        self.assertTrue(model['valid'])
        np.testing.assert_allclose(predict(model, x), training.target, atol=1e-12)
        training[FEATURES[1]] = training[FEATURES[0]]
        self.assertFalse(fit_model(training)['valid'])
        self.assertFalse(fit_model(training.iloc[:20])['valid'])

    def test_causal_features_and_label_boundary(self):
        rng = np.random.default_rng(19)
        prices = pd.DataFrame(100 + rng.normal(size=(320, 2)).cumsum(axis=0),
                              index=pd.bdate_range('2023-01-01', periods=320), columns=['A', 'B'])
        x, spread, gross = features(prices, 'A', 'B', 0., 1.)
        changed = prices.copy()
        changed.iloc[252:] *= 3
        altered, _, _ = features(changed, 'A', 'B', 0., 1.)
        pd.testing.assert_frame_equal(x.iloc[:252], altered.iloc[:252])
        training = training_data(prices.iloc[:252], 'A', 'B', 0., 1.)
        self.assertEqual(training.index[-1], prices.index[245])
        i = prices.index.get_loc(training.index[0])
        self.assertAlmostEqual(training.target.iloc[0], (spread.iloc[i+6] - spread.iloc[i+1]) / gross.iloc[i])
        first = fit_model(training)
        second = fit_model(training_data(changed.iloc[:252], 'A', 'B', 0., 1.))
        self.assertEqual(first, second)

    def test_gate_sign_costs_and_exit(self):
        signals = pd.DataFrame({'ZScore': [-2.5, -1., -.4, 2.5], 'Price1': [150.]*4,
                                'Price2': [50.]*4, 'Hedge_Ratio': [1.]*4})
        result = apply_gate(signals, np.array([.02, -.02, np.nan, .02]))
        self.assertEqual(result.Signal.tolist(), [1, 1, 0, 0])
        self.assertAlmostEqual(result.Estimated_Roundtrip_Cost.iloc[0], .002 + .02*7/365.25*.25)
        self.assertFalse(apply_gate(signals, np.full(4, np.nan)).Signal.any())

    def test_engine_extension_preserves_baseline_and_reconciles(self):
        prices, schedule = fixture()
        base = run_variant(prices, schedule, 'baseline')
        identity = run_variant(prices, schedule, 'baseline', signal_transform=lambda signals, *args: signals)
        pd.testing.assert_series_equal(base['equity_curve'], identity['equity_curve'])
        pd.testing.assert_frame_equal(base['trades'], identity['trades'])
        gate = RegressionGate()
        result = run_variant(prices, schedule, 'baseline', signal_transform=gate)
        self.assertTrue(len(gate.models))
        self.assertAlmostEqual(result['trades'].Net_PnL.sum(), result['equity_curve'].iloc[-1] - 100000)


if __name__ == '__main__':
    unittest.main()
