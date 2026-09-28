"""The choice must depend on earlier evidence, with a meaningful cash fallback."""
import unittest

from run_ols_improvement import select_validation_rule


def result(ret, trades):
    return {'metrics': {'Total_Return': ret, 'Num_Trades': trades}}


class SelectionTests(unittest.TestCase):
    def test_stability_beats_one_large_winning_fold(self):
        folds = {'validation_1': {'steady': result(.03, 5), 'uneven': result(.4, 8)},
                 'validation_2': {'steady': result(.03, 5), 'uneven': result(.01, 8)}}
        self.assertEqual(select_validation_rule(folds)['selected_variant'], 'steady')

    def test_inactivity_small_samples_or_negative_fold_cannot_win(self):
        folds = {'validation_1': {'idle': result(0, 0), 'thin': result(.5, 2), 'loss': result(-.01, 10)},
                 'validation_2': {'idle': result(0, 0), 'thin': result(.5, 20), 'loss': result(.5, 10)}}
        self.assertIsNone(select_validation_rule(folds)['selected_variant'])

    def test_later_metrics_rejected_at_selection_boundary(self):
        with self.assertRaises(ValueError):
            select_validation_rule({'validation_1': {}, 'validation_2': {}, 'later': {}})


if __name__ == '__main__':
    unittest.main()
