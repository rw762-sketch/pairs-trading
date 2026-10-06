import unittest
import pandas as pd
from run_persistence import persistent_mask

class PersistenceTests(unittest.TestCase):
    def test_five_passes_and_latest_required_missing_not_pass(self):
        frame=pd.DataFrame([
            [.01,.02,.03,.04,.06,.02],
            [.01,.02,.03,.04,.02,.06],
            [.01,.02,.03,.06,.06,.02],
            [.01,.02,.03,.04,float('nan'),.02],
        ])
        self.assertEqual(persistent_mask(frame,require_latest=True).tolist(),[True,False,False,True])
        self.assertEqual(persistent_mask(frame,required=4,require_latest=True).tolist(),[True,False,True,True])

    def test_latest_failure_does_not_exclude_without_constraint(self):
        frame = pd.DataFrame([[.01, .02, .2, .2, .2, .2], [.01, .2, .2, .2, .2, .2]])
        self.assertEqual(persistent_mask(frame, required=2).tolist(), [True, False])
        self.assertEqual(persistent_mask(frame, required=2, require_latest=True).tolist(), [False, False])
