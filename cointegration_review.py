"""Readable, auditable summaries of a saved cointegration screen.

The saved pair file contains passing candidates only. Missing combinations have
no recorded p-value here, and raw counts are not a ranking of trading quality.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


PAIR_COLUMNS = {
    'ticker1': 'Ticker1', 'ticker2': 'Ticker2',
    'coint_pvalue': 'Coint_PValue', 'adf_pvalue': 'ADF_PValue',
    'hedge_ratio': 'Hedge_Ratio', 'coint_score': 'Coint_Score',
    'adf_statistic': 'ADF_Statistic', 'intercept': 'Intercept',
}


def normalize_pair_results(pairs):
    """Accept legacy dictionaries or the exported CSV's column names."""
    frame = pd.DataFrame(pairs).copy().rename(columns=PAIR_COLUMNS)
    required = ['Ticker1', 'Ticker2', 'Coint_PValue']
    if frame.empty:
        return frame.reindex(columns=list(dict.fromkeys(required + list(frame.columns))))
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing pair columns: {sorted(missing)}')
    if frame[required].isna().any().any():
        raise ValueError('Pair tickers and p-values must not be missing.')
    frame['Coint_PValue'] = pd.to_numeric(frame['Coint_PValue'], errors='raise')
    if not frame['Coint_PValue'].between(0, 1).all():
        raise ValueError('Cointegration p-values must be between zero and one.')
    if (frame['Ticker1'] == frame['Ticker2']).any():
        raise ValueError('Self-pairs are not cointegration candidates.')
    keys = frame.apply(lambda row: tuple(sorted((row.Ticker1, row.Ticker2))), axis=1)
    if keys.duplicated().any():
        raise ValueError('Repeated unordered pairs would inflate partner counts.')
    return frame.drop(columns=[c for c in frame if str(c).startswith('Unnamed:')])


def _metadata_by_ticker(metadata):
    if isinstance(metadata, dict):
        return metadata
    frame = pd.DataFrame(metadata)
    if frame.empty:
        return {}
    ticker_column = next((c for c in ['Yahoo ticker', 'Ticker', 'ticker', 'Symbol']
                          if c in frame), None)
    if ticker_column is None:
        raise ValueError('Stock metadata needs a ticker column.')
    return {str(row[ticker_column]).replace('.', '-'): row
            for row in frame.to_dict('records')}


def build_cointegration_review(pairs_df, universe_metadata, total_tested=None):
    """Return raw partner rankings, lowest-p pairs, and interpretation cautions.

    ``total_tested`` must be the number of tests originally run, including those
    whose nonpassing p-values were not saved. No missing p-values are imputed.
    Only saved candidates with raw p < .05 contribute to partner counts.
    """
    pairs = normalize_pair_results(pairs_df)
    metadata = _metadata_by_ticker(universe_metadata)
    if total_tested is not None and total_tested < len(pairs):
        raise ValueError('total_tested cannot be smaller than the saved pair count.')
    candidates = pairs.loc[pairs['Coint_PValue'] < .05].copy()
    all_tickers = sorted(set(pairs['Ticker1']) | set(pairs['Ticker2']))
    records = []
    for ticker in all_tickers:
        rows = candidates.loc[(candidates['Ticker1'] == ticker) |
                              (candidates['Ticker2'] == ticker)]
        partners = sorted((set(rows['Ticker1']) | set(rows['Ticker2'])) - {ticker})
        stock = metadata.get(ticker, {})
        records.append({
            'Ticker': ticker,
            'Company': stock.get('Security', stock.get('Company', ticker)),
            'Sector': stock.get('GICS Sector', stock.get('Sector', 'Unknown')),
            'Industry': stock.get('GICS Sub-Industry', stock.get('Industry', 'Unknown')),
            'Partner_Count': len(partners),
            'Min_PValue': rows['Coint_PValue'].min(),
            'Median_PValue': rows['Coint_PValue'].median(),
            'Partners': ', '.join(partners),
        })
    stock_columns = ['Ticker', 'Company', 'Sector', 'Industry', 'Partner_Count',
                     'Min_PValue', 'Median_PValue', 'Partners']
    ranked = pd.DataFrame(records, columns=stock_columns).sort_values(
        ['Partner_Count', 'Ticker'], ascending=[False, True], ignore_index=True)
    lowest = candidates.sort_values(['Coint_PValue', 'Ticker1', 'Ticker2']).reset_index(drop=True)
    for number in [1, 2]:
        lowest[f'Company{number}'] = lowest[f'Ticker{number}'].map(
            lambda t: metadata.get(t, {}).get('Security', metadata.get(t, {}).get('Company', t)))
        lowest[f'Industry{number}'] = lowest[f'Ticker{number}'].map(
            lambda t: metadata.get(t, {}).get('GICS Sub-Industry',
                                            metadata.get(t, {}).get('Industry', 'Unknown')))
    lowest['Same_Industry'] = ((lowest['Industry1'] == lowest['Industry2']) &
                              (lowest['Industry1'] != 'Unknown'))
    lowest['Positive_Beta'] = (lowest['Hedge_Ratio'] > 0
                               if 'Hedge_Ratio' in lowest else False)
    bonferroni = .05 / total_tested if total_tested else None
    diagnostics = {
        'saved_pair_count': len(pairs),
        'raw_p_below_0_05_count': len(candidates),
        'stocks_in_saved_pairs': len(all_tickers),
        'total_tests_in_original_screen': total_tested,
        'bonferroni_threshold': bonferroni,
        'saved_pairs_below_bonferroni': (int((pairs['Coint_PValue'] < bonferroni).sum())
                                       if bonferroni is not None else None),
        'negative_beta_count': (int((lowest['Hedge_Ratio'] < 0).sum())
                                if 'Hedge_Ratio' in lowest else None),
        'same_industry_count': int(lowest['Same_Industry'].sum()),
        'cautions': [
            'Counts describe saved training candidates with raw p < 0.05, not confirmed trading opportunities.',
            'A small p-value is evidence against no cointegration under test assumptions, not the probability of profit.',
            'Testing many pairs creates multiple-comparison risk; the Bonferroni threshold is a conservative diagnostic.',
            'Engle-Granger assumes both input price series are I(1); stationary price-level hubs can produce misleading rankings.',
            'The original screen used one regression orientation per pair; finite-sample p-values depend on that orientation.',
            'Unrecorded pairs have unknown p-values in this saved file; do not replace them with 1.',
            'Negative beta gives same-direction legs in a spread position; it is not the usual stock long/short hedge.',
            'Current index membership and full-period missing-data filtering introduce survivorship and selection limitations.',
        ],
        'sources': [
            'https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html',
            'https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html',
        ],
    }
    return {'ranked_stocks': ranked, 'lowest_p_pairs': lowest, 'diagnostics': diagnostics}


def integration_diagnostics(train_prices):
    """Screen price levels and first differences for I(1) compatibility.

    ADF uses a constant and AIC lag selection. Level p >= .05 and difference
    p < .05 are a provisional diagnostic only: failure to reject a unit root
    does not prove one. This function must receive training observations only.
    """
    rows = []
    for ticker in train_prices.columns:
        values = train_prices[ticker].astype(float)
        error = ''
        level_p = difference_p = np.nan
        try:
            if not np.isfinite(values).all() or len(values) < 30:
                raise ValueError('Need at least 30 complete finite training prices.')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                level_p = float(adfuller(values, regression='c', autolag='AIC')[1])
                difference_p = float(adfuller(values.diff().dropna(), regression='c', autolag='AIC')[1])
        except (ValueError, np.linalg.LinAlgError) as exc:
            error = str(exc)
        rows.append({
            'Ticker': ticker,
            'Level_ADF_PValue': level_p,
            'Diff_ADF_PValue': difference_p,
            'I1_Compatible': bool(level_p >= .05 and difference_p < .05),
            'Observations': len(values),
            'Error': error,
        })
    return pd.DataFrame(rows, columns=['Ticker', 'Level_ADF_PValue', 'Diff_ADF_PValue',
                                      'I1_Compatible', 'Observations', 'Error'])
