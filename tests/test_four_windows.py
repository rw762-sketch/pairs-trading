import unittest
import pandas as pd
from run_four_windows import annual_blocks

class FourWindowTests(unittest.TestCase):
    def test_all_dates_used_once_no_overlap_balanced_lengths(self):
        index=pd.bdate_range('2021-09-17',periods=1005)
        blocks=annual_blocks(index)
        joined=blocks[0].append(blocks[1:])
        pd.testing.assert_index_equal(index,joined)
        self.assertTrue(joined.is_unique)
        self.assertEqual(len(blocks),4)
        self.assertLessEqual(max(map(len,blocks))-min(map(len,blocks)),1)
        for before,after in zip(blocks,blocks[1:]):self.assertLess(before[-1],after[0])
