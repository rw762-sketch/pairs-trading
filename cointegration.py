"""Engle-Granger testing for EVERY prescribed within-cluster pair."""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.stats.multitest import multipletests

COLUMNS = [
    "Pair",
    "Ticker1",
    "Ticker2",
    "Raw_P",
    "Holm_P",
    "Alpha",
    "Beta",
    "AR1_Phi",
    "Half_Life",
    "I1_Compatible",
    "Status",
    "Error",
    "Selected",
]


def half_life(residual):
    x = np.asarray(residual, dtype=float)
    design = np.column_stack([np.ones(len(x) - 1), x[:-1]])
    coefficients, _, rank, _ = np.linalg.lstsq(design, x[1:], rcond=None)
    phi = float(coefficients[1]) if rank == 2 else np.nan
    return phi, float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else np.nan


def integration_diagnostics(training):
    rows = []
    for ticker in training:
        level = difference = np.nan
        error = ""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                level = float(adfuller(training[ticker], autolag="AIC")[1])
                difference = float(
                    adfuller(training[ticker].diff().dropna(), autolag="AIC")[1]
                )
        except (ValueError, np.linalg.LinAlgError) as exc:
            error = str(exc)
        rows.append(
            {
                "Ticker": ticker,
                "Level_ADF_P": level,
                "Diff_ADF_P": difference,
                "I1_Compatible": bool(level >= 0.05 and difference < 0.05),
                "Error": error,
            }
        )
    return pd.DataFrame(
        rows, columns=["Ticker", "Level_ADF_P", "Diff_ADF_P", "I1_Compatible", "Error"]
    )


def screen_pairs(
    training, candidate_pairs, pvalue=0.05, correction="raw", progress=False
):
    """No correlation or industry filter precedes the cointegration tests.

    The positive-beta and I(1) diagnostics govern trading eligibility AFTER all
    pairs have been tested. Failed tests remain in the saved family with status.
    Holm includes the entire family; failed tests use conservative p=1 inputs.
    """
    if correction not in {"raw", "holm"} or not 0 < pvalue < 1:
        raise ValueError("Use correction raw/holm and a p-value cutoff in (0,1).")
    pairs = list(candidate_pairs)
    if any(a == b or a not in training or b not in training for a, b in pairs):
        raise ValueError("Pairs require two distinct available tickers.")
    if len({frozenset(pair) for pair in pairs}) != len(pairs):
        raise ValueError("Duplicate candidate pairs.")
    diagnostics = integration_diagnostics(training)
    compatible = diagnostics.set_index("Ticker").I1_Compatible.to_dict()
    rows = []
    for number, (a, b) in enumerate(pairs, 1):
        row = {
            "Pair": f"{a}-{b}",
            "Ticker1": a,
            "Ticker2": b,
            "Raw_P": np.nan,
            "Alpha": np.nan,
            "Beta": np.nan,
            "AR1_Phi": np.nan,
            "Half_Life": np.nan,
            "Status": "tested",
            "Error": "",
            "I1_Compatible": compatible.get(a, False) and compatible.get(b, False),
        }
        try:
            av = training[a].to_numpy(dtype=float)
            bv = training[b].to_numpy(dtype=float)
            alpha, beta = np.linalg.lstsq(
                np.column_stack([np.ones(len(bv)), bv]), av, rcond=None
            )[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = float(coint(av, bv, trend="c", autolag="aic")[1])
            if not np.isfinite([alpha, beta, raw]).all():
                raise ValueError("Nonfinite fit or cointegration result.")
            phi, life = half_life(av - alpha - beta * bv)
            row.update(
                Raw_P=raw,
                Alpha=float(alpha),
                Beta=float(beta),
                AR1_Phi=phi,
                Half_Life=life,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            row.update(Status="failed", Error=str(exc))
        rows.append(row)
        if progress and (number % 500 == 0 or number == len(pairs)):
            print(f"Cointegration: {number:,}/{len(pairs):,} pairs tested", flush=True)
    panel = pd.DataFrame(rows).reindex(columns=COLUMNS)
    if len(panel):
        panel["Holm_P"] = multipletests(panel.Raw_P.fillna(1), method="holm")[1]
    cutoff = panel.Holm_P if correction == "holm" else panel.Raw_P
    panel["Selected"] = (
        panel.Status.eq("tested")
        & (cutoff < pvalue)
        & (panel.Beta > 0)
        & panel.I1_Compatible
        & np.isfinite(panel.Half_Life.astype(float))
    ).astype(bool)
    panel = panel.sort_values(
        ["Raw_P", "Half_Life", "Pair"], na_position="last"
    ).reset_index(drop=True)
    return panel, diagnostics
