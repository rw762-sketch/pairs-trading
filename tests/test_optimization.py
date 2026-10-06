import unittest
import pandas as pd
from dataclasses import replace
from config import Config
from run_optimization import select

class OptimizationTests(unittest.TestCase):
    def test_three_passes_no_latest_requirement_and_strict_cutoff(self):
        from tests.test_recovery import panel, training
        evidence=pd.DataFrame([
            [.01,.02,.03,.2,.2,.2],
            [.01,.02,.2,.2,.2,.2],
            [.01,.02,.05,.2,.2,.2],
        ],index=panel().Pair)
        data=panel()
        data['I1_Compatible']=True
        data['Status']='tested'
        selected=select(data,evidence,training(),replace(Config(),min_crossings=0))
        self.assertEqual(selected.loc[selected.Selected,'Pair'].tolist(),['A-B'])
        strict=select(data,evidence,training(),replace(Config(),pvalue=.01,min_crossings=0))
        self.assertFalse(strict.Selected.any())
