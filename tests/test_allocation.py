import unittest
import numpy as np
import pandas as pd
from allocation import persistence_weights

class AllocationTests(unittest.TestCase):
    def test_more_passes_and_lower_mean_receive_larger_scores(self):
        frame=pd.DataFrame([[.01,.01,.01,.01],[.01,.01,.06,.06],[.02,.02,.06,.06]],index=['A','B','C'])
        result=persistence_weights(frame)
        self.assertGreater(result.loc['A','Weight'],result.loc['B','Weight'])
        self.assertGreater(result.loc['B','Weight'],result.loc['C','Weight'])
        self.assertAlmostEqual(result.Weight.sum(),1.0)
        self.assertGreater(result.loc['A','Weight'],.2)

    def test_missing_and_zero_values_are_conservative_and_finite(self):
        frame=pd.DataFrame([[0.,0.],[.01,np.nan]],index=['A','B'])
        result=persistence_weights(frame)
        self.assertEqual(result.loc['B','Pass_Count'],1)
        self.assertAlmostEqual(result.loc['B','Mean_P'],.505)
        self.assertTrue(np.isfinite(result.Weight).all())
        with self.assertRaises(ValueError):persistence_weights(pd.DataFrame([[1.2,.01]]))
