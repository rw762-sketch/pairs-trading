"""Training-evidence allocation scores; no evaluation returns enter weights."""
import numpy as np
import pandas as pd


def persistence_weights(pvalues, cutoff=0.05, floor=1e-6):
    """Normalize pass-count / mean p-value over every training window.

    Missing test results count as p=1, not as passes. Mean p is descriptive,
    not a combined statistical significance test. The denominator floor only
    prevents division by zero; no weight cap is imposed.
    """
    if not 0 < cutoff < 1 or not np.isfinite(floor) or floor <= 0:
        raise ValueError('Require a valid p cutoff and positive finite floor.')
    if not pvalues.index.is_unique or pvalues.shape[1] == 0:
        raise ValueError('Require unique pair identifiers and training-window columns.')
    values=pvalues.astype(float)
    observed=values.to_numpy()
    if np.isinf(observed).any() or ((values < 0) | (values > 1)).any().any():
        raise ValueError('p-values must be between zero and one or missing.')
    values=values.fillna(1.0)
    passes=values.lt(cutoff).sum(axis=1)
    mean=values.mean(axis=1)
    score=passes/mean.clip(lower=floor)
    if not np.isfinite(score.sum()) or score.sum()<=0:
        raise ValueError('No positive persistence scores available for allocation.')
    return pd.DataFrame({'Pass_Count':passes,'Mean_P':mean,'Allocation_Score':score,'Weight':score/score.sum()})
