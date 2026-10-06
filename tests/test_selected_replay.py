"""Exact selected-strategy replay from committed inputs, without local caches."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import run_selected


ROOT = Path(__file__).resolve().parents[1]


class SelectedReplayTests(unittest.TestCase):
    def make_clone(self, destination):
        clone = Path(destination)
        shutil.copyfile(ROOT / 'selected_strategy.json', clone / 'selected_strategy.json')
        shutil.copytree(ROOT / 'reproducibility/selected', clone / 'reproducibility/selected')
        return clone

    def test_fresh_clone_reproduces_official_result_without_local_cache(self):
        with tempfile.TemporaryDirectory() as destination:
            clone = self.make_clone(destination)
            expected = json.loads((clone / 'reproducibility/selected/manifest.json').read_text())['expected_result']
            with patch('run_selected.ROOT', clone):
                result = run_selected.run_selected_strategy(clone / 'data/cache', clone / 'results')
            self.assertFalse((clone / 'data/cache/adjusted-prices.pkl').exists())
            self.assertEqual(set(result['selected'].Pair), {'EVRG-WELL', 'MCO-SPGI', 'ITW-NXPI'})
            self.assertEqual(len(result['selected']), expected['num_pairs'])
            self.assertAlmostEqual(result['selected'].Weight.sum(), 1.0)
            self.assertAlmostEqual(result['portfolio']['metrics']['Total_Return'], expected['total_return'], places=12)
            self.assertEqual(result['portfolio']['metrics']['Num_Trades'], expected['num_trades'])
            curve = result['portfolio']['equity_curve']
            self.assertEqual(str(curve.index[0].date()), expected['evaluation_start'])
            self.assertEqual(str(curve.index[-1].date()), expected['evaluation_end'])
            self.assertTrue((result['output'] / 'report.html').exists())

    def test_missing_explicit_alternative_cache_does_not_use_bundle(self):
        with tempfile.TemporaryDirectory() as destination:
            clone = self.make_clone(destination)
            with patch('run_selected.ROOT', clone):
                chosen = run_selected.load_selected_strategy()
                with self.assertRaisesRegex(FileNotFoundError, 'price snapshot is missing'):
                    run_selected.resolve_replay_inputs(chosen, clone / 'another-cache')

    def test_changed_local_snapshot_is_rejected_instead_of_overridden(self):
        with tempfile.TemporaryDirectory() as destination:
            clone = self.make_clone(destination)
            cache = clone / 'data/cache'
            cache.mkdir(parents=True)
            (cache / 'adjusted-prices.pkl').write_bytes(b'changed local prices')
            with patch('run_selected.ROOT', clone), patch('run_selected.pd.read_pickle') as reader:
                with self.assertRaisesRegex(ValueError, 'original price snapshot'):
                    run_selected.run_selected_strategy(cache, clone / 'results')
                reader.assert_not_called()

    def test_changed_bundled_screen_fails_integrity_verification(self):
        with tempfile.TemporaryDirectory() as destination:
            clone = self.make_clone(destination)
            screen = clone / 'reproducibility/selected/persistence-tests.csv'
            screen.write_bytes(screen.read_bytes() + b'\nchanged screen\n')
            with patch('run_selected.ROOT', clone), patch('run_selected.pd.read_pickle') as reader:
                with self.assertRaisesRegex(ValueError, 'SHA256 verification: persistence-tests.csv'):
                    run_selected.run_selected_strategy(clone / 'data/cache', clone / 'results')
                reader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
