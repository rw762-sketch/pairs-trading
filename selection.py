"""Training-only recovery filters and deterministic shared-stock limits."""

import numpy as np
import pandas as pd


def mean_crossings(residual):
    """Count changes of sign around the training mean; exact touches do not count."""
    values = np.asarray(residual, dtype=float)
    signs = np.sign(values - values.mean())
    signs = signs[signs != 0]
    return int(np.count_nonzero(np.diff(signs))) if len(signs) > 1 else 0


def apply_filters(panel, training, config):
    result = panel.copy()
    result["Base_Selected"] = result.Selected.astype(bool)
    counts = []
    rates = []
    for row in result.itertuples():
        if np.isfinite([row.Alpha, row.Beta]).all():
            residual = (
                training[row.Ticker1] - row.Alpha - row.Beta * training[row.Ticker2]
            )
            count = mean_crossings(residual)
            rate = count * 252 / max(len(training) - 1, 1)
        else:
            count = 0
            rate = 0.0
        counts.append(count)
        rates.append(rate)
    result["Mean_Crossings"] = counts
    result["Crossings_Per_Year"] = rates
    result["Recovery_Eligible"] = (
        result.Base_Selected
        & result.Half_Life.between(config.half_life_min, config.half_life_max)
        & (result.Crossings_Per_Year >= config.min_crossings)
    )
    accepted = []
    used = set()
    for row in result.itertuples():
        allow = bool(
            row.Recovery_Eligible
            and row.Ticker1 not in used
            and row.Ticker2 not in used
        )
        accepted.append(allow)
        if allow:
            used.update([row.Ticker1, row.Ticker2])
    result["Selected"] = pd.Series(accepted, index=result.index, dtype=bool)
    n = int(result.Selected.sum())
    result["Weight"] = np.where(
        result.Selected, min(1 / n, config.max_pair_weight) if n else 0.0, 0.0
    )
    return result
