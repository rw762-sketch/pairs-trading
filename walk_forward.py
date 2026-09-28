"""Prespecified, monthly walk-forward comparison on a fixed peer-stock universe.

All model selection sees preceding formation observations only. The same fixed
same-industry/different-issuer hypothesis family is recorded in every window.
Each variant liquidates at every window end and carries its account capital
forward. This is a research comparison; no variant is chosen by its final return.
"""

from itertools import combinations
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint

from backtester import (PairBacktester, TRADE_COLUMNS, combine_portfolio_results,
                        performance_metrics)
from signal_generation import PairSignalGenerator

VARIANT_NAMES = ('fixed_selection', 'monthly_raw', 'monthly_holm')
SCREEN_COLUMNS = [
    'Pair', 'Ticker1', 'Ticker2', 'Industry', 'Raw_P', 'Correction_Input_P',
    'Adjusted_P', 'Beta', 'Alpha', 'I1Compatible', 'Raw_Eligible', 'Holm_Eligible',
    'Test_Status', 'Test_Error', 'Correction_Input_Is_Placeholder', 'Family_Size',
    'Level_ADF_P1', 'Diff_ADF_P1', 'Level_ADF_P2', 'Diff_ADF_P2',
]
PAIR_PROFIT_COLUMNS = ['Pair', 'Net_PnL', 'Num_Trades', 'Costs', 'Selected_Windows']


def _metadata_frame(metadata):
    if isinstance(metadata, pd.DataFrame):
        frame = metadata.copy()
    elif isinstance(metadata, dict):
        frame = pd.DataFrame([dict(value, **{'Yahoo ticker': key})
                              for key, value in metadata.items()])
    else:
        frame = pd.DataFrame(metadata)
    ticker_column = next((key for key in ('Yahoo ticker', 'Ticker', 'ticker', 'Symbol') if key in frame), None)
    industry_column = next((key for key in ('GICS Sub-Industry', 'Industry', 'industry') if key in frame), None)
    if ticker_column is None or industry_column is None or 'CIK' not in frame:
        raise ValueError('Metadata must identify each ticker, its GICS sub-industry, and issuer CIK')
    if frame[[ticker_column, industry_column, 'CIK']].isna().any().any():
        raise ValueError('Ticker, industry, and issuer metadata must be complete before testing')
    frame = frame.rename(columns={ticker_column: 'Ticker', industry_column: 'Industry'})
    frame['Ticker'] = frame.Ticker.astype(str).str.replace('.', '-', regex=False)
    frame['Industry'] = frame.Industry.astype(str).str.strip()
    frame['CIK'] = frame.CIK.map(lambda value: str(int(value)))
    if frame.Ticker.duplicated().any() or (frame.Ticker.str.len() == 0).any() or (frame.Industry.str.len() == 0).any():
        raise ValueError('Metadata must contain unique tickers and nonempty industries')
    return frame.sort_values('Ticker').reset_index(drop=True)


def prescribe_pair_family(metadata):
    """Fix peer combinations from metadata before inspecting any test p-values."""
    stocks = _metadata_frame(metadata)
    rows = []
    for a, b in combinations(stocks.itertuples(), 2):
        if a.Industry == b.Industry and a.CIK != b.CIK:
            rows.append({'Pair': f'{a.Ticker}-{b.Ticker}', 'Ticker1': a.Ticker,
                         'Ticker2': b.Ticker, 'Industry': a.Industry})
    return pd.DataFrame(rows, columns=['Pair', 'Ticker1', 'Ticker2', 'Industry'])


def holm_correction(raw_pvalues, testable=None, alpha=.05):
    """Holm correction over the complete prescribed family.

    Untested hypotheses contribute a conservative input of 1.0. Their observed
    raw p-value remains missing in the screening table; the placeholder is never
    presented as a measured statistical result. The returned adjusted values
    belong to this full-family correction, including the explicit placeholders.
    """
    raw = np.asarray(raw_pvalues, dtype=float)
    valid = np.isfinite(raw) if testable is None else np.asarray(testable, dtype=bool)
    if raw.ndim != 1 or valid.shape != raw.shape:
        raise ValueError('Raw p-values and testable mask must be matching vectors')
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between zero and one')
    if (valid & (~np.isfinite(raw) | (raw < 0) | (raw > 1))).any():
        raise ValueError('Observed testable p-values must be finite and within [0, 1]')
    correction_input = np.where(valid, raw, 1.0)
    if len(raw):
        reject, adjusted, _, _ = multipletests(correction_input, alpha=alpha, method='holm')
    else:
        reject, adjusted = np.array([], dtype=bool), np.array([], dtype=float)
    return {'correction_input': correction_input, 'adjusted': adjusted,
            'reject': reject & valid, 'is_placeholder': ~valid}


def screen_formation(formation, pair_family, alpha=.05):
    """Test only the supplied formation observations, retaining every hypothesis."""
    if len(formation) < 30:
        raise ValueError('Formation needs at least 30 historical observations')
    diagnostics = {}
    for ticker in sorted(set(pair_family.Ticker1) | set(pair_family.Ticker2)):
        values = formation[ticker].to_numpy(dtype=float)
        level = difference = np.nan
        error = ''
        if not np.isfinite(values).all() or (values <= 0).any():
            error = 'Formation contains missing, nonfinite or nonpositive prices'
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FutureWarning)
                    level = float(adfuller(values, regression='c', autolag='AIC')[1])
                    difference = float(adfuller(np.diff(values), regression='c', autolag='AIC')[1])
            except (ValueError, np.linalg.LinAlgError) as exc:
                error = str(exc)
        compatible = bool(np.isfinite(level) and np.isfinite(difference) and level >= alpha and difference < alpha)
        diagnostics[ticker] = {'level': level, 'difference': difference, 'compatible': compatible, 'error': error}
    rows = []
    for pair in pair_family.itertuples():
        da, db = diagnostics[pair.Ticker1], diagnostics[pair.Ticker2]
        compatible = da['compatible'] and db['compatible']
        raw = beta = intercept = np.nan
        status = 'not_tested_i1_incompatible'
        error = '; '.join(value for value in (da['error'], db['error']) if value)
        if error:
            status = 'not_tested_invalid_formation'
        elif compatible:
            try:
                a = formation[pair.Ticker1].to_numpy(dtype=float)
                b = formation[pair.Ticker2].to_numpy(dtype=float)
                intercept, beta = np.linalg.lstsq(np.column_stack([np.ones(len(b)), b]), a, rcond=None)[0]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    raw = float(coint(a, b, trend='c', autolag='AIC')[1])
                if not np.isfinite(raw) or not np.isfinite(beta):
                    raise ValueError('Nonfinite cointegration result or hedge ratio')
                status = 'tested'
                error = '; '.join(str(w.message) for w in caught)
            except (ValueError, np.linalg.LinAlgError) as exc:
                status, error, raw = 'test_error', str(exc), np.nan
        rows.append({
            'Pair': pair.Pair, 'Ticker1': pair.Ticker1, 'Ticker2': pair.Ticker2,
            'Industry': pair.Industry, 'Raw_P': raw, 'Beta': beta, 'Alpha': intercept,
            'I1Compatible': compatible, 'Test_Status': status, 'Test_Error': error,
            'Level_ADF_P1': da['level'], 'Diff_ADF_P1': da['difference'],
            'Level_ADF_P2': db['level'], 'Diff_ADF_P2': db['difference'],
        })
    frame = pd.DataFrame(rows).reindex(columns=SCREEN_COLUMNS)
    observed = frame.Test_Status.eq('tested').to_numpy()
    correction = holm_correction(frame.Raw_P.to_numpy(dtype=float), observed, alpha)
    frame['Correction_Input_P'] = correction['correction_input']
    frame['Adjusted_P'] = correction['adjusted']
    frame['Correction_Input_Is_Placeholder'] = correction['is_placeholder']
    frame['Family_Size'] = len(pair_family)
    frame['Raw_Eligible'] = observed & (frame.Raw_P < alpha) & (frame.Beta > 0)
    frame['Holm_Eligible'] = correction['reject'] & (frame.Beta > 0)
    return frame


def _window_backtest(prices, dates, selected, capital, entry, exit,
                     lookback, fee, borrow, max_hold):
    """One flat-to-flat month with a causal warmup and one total capital budget."""
    if selected.empty:
        curve = pd.Series(capital, index=dates, name='Portfolio_Value', dtype=float)
        trades = pd.DataFrame(columns=TRADE_COLUMNS + ['Portfolio_Weight'])
        return {'equity_curve': curve, 'trades': trades,
                'metrics': performance_metrics(curve, trades, capital)}
    # Cut the history at this month's end; no signal can inspect a later month.
    history = prices.loc[:dates[-1]].tail(len(dates) + lookback)
    results = {}
    for row in selected.itertuples():
        generator = PairSignalGenerator(history, row.Ticker1, row.Ticker2,
                                         hedge_ratio=row.Beta, lookback=lookback)
        generator.calculate_zscore()
        for attribute in ('price_data', 'spread', 'z_score', 'moving_mean', 'moving_std'):
            setattr(generator, attribute, getattr(generator, attribute).loc[dates])
        signals = generator.get_signals(entry_threshold=entry, exit_threshold=exit)
        engine = PairBacktester(signals, initial_capital=capital, transaction_cost=fee,
                                max_holding_period=max_hold, annual_borrow_rate=borrow)
        curve = engine.run_backtest()
        if len(curve) and curve.Gross_Exposure.iloc[-1] != 0:
            raise AssertionError('Window must finish with both legs liquidated')
        results[row.Pair] = {'equity_curve': curve, 'trades': engine.get_trades_dataframe(),
                             'metrics': engine.get_performance_metrics(), 'initial_capital': capital}
    combined = combine_portfolio_results(results, initial_capital=capital)
    if not np.isclose(combined['trades'].Net_PnL.sum(), combined['equity_curve'].iloc[-1] - capital,
                      rtol=1e-10, atol=1e-7):
        raise AssertionError('Allocated trade P&L must reconcile to the monthly account')
    return combined


def run_experiment(prices, metadata, evaluation_start='2025-10-23', formation_days=252,
                   initial_capital=100000, entry=2, exit=.5, lookback=60,
                   fee=.001, borrow=.02, max_hold=20):
    """Compare three prescribed selection variants, without fitting to outcomes.

    ``prices`` must include prior formation/warmup data and evaluation dates;
    ``metadata`` fixes the stock universe, peer industries and issuer identities.
    The fixed variant uses the first formation's raw selection and betas forever.
    The other two reselect/refit each calendar month using the preceding window.
    All variants close positions each month for a matched selection comparison.
    """
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError('Prices require finite, unique, chronologically sorted dates')
    if prices.columns.duplicated().any():
        raise ValueError('Price tickers must be unique')
    for name, value, minimum in [('formation_days', formation_days, 30), ('lookback', lookback, 2)]:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError('initial_capital must be finite and positive')
    if not np.isfinite(entry) or not np.isfinite(exit) or not 0 <= exit < entry:
        raise ValueError('Require 0 <= exit < entry')
    if not np.isfinite(fee) or fee < 0 or not np.isfinite(borrow) or borrow < 0:
        raise ValueError('Fee and borrowing assumptions must be finite and nonnegative')
    if max_hold is not None and (not isinstance(max_hold, (int, np.integer)) or isinstance(max_hold, bool) or max_hold < 1):
        raise ValueError('max_hold must be a positive integer of trading bars')
    stocks = _metadata_frame(metadata)
    missing = set(stocks.Ticker) - set(prices.columns)
    if missing:
        raise ValueError(f'Prices missing universe tickers: {sorted(missing)}')
    prices = prices.loc[:, stocks.Ticker].astype(float).copy()
    family = prescribe_pair_family(metadata)
    start = pd.Timestamp(evaluation_start)
    if prices.index.tz is not None and start.tzinfo is None:
        start = start.tz_localize(prices.index.tz)
    evaluation_dates = prices.index[prices.index >= start]
    if len(evaluation_dates) == 0:
        raise ValueError('No evaluation observations on or after evaluation_start')
    first_row = prices.index.get_loc(evaluation_dates[0])
    if first_row < max(formation_days, lookback):
        raise ValueError('Insufficient preceding data for formation and signal warmup')
    month_keys = evaluation_dates.strftime('%Y-%m')
    windows = [evaluation_dates[month_keys == key] for key in dict.fromkeys(month_keys)]
    all_screening, scheduled = [], []
    variant_storage = {name: {'curves': [], 'logs': [], 'monthly': [], 'selected_counts': {},
                              'capital': float(initial_capital)} for name in VARIANT_NAMES}
    frozen_selection = None
    first_formation_start = first_formation_end = None
    for number, dates in enumerate(windows, start=1):
        first_trade_row = prices.index.get_loc(dates[0])
        formation = prices.iloc[first_trade_row - formation_days:first_trade_row]
        if len(formation) != formation_days or formation.index[-1] >= dates[0]:
            raise AssertionError('Formation must contain only strictly earlier observations')
        panel = screen_formation(formation, family)
        panel['Window'] = number
        panel['Formation_Start'] = formation.index[0]
        panel['Formation_End'] = formation.index[-1]
        panel['Trading_Start'] = dates[0]
        panel['Trading_End'] = dates[-1]
        all_screening.append(panel)
        scheduled.append({'Window': number, 'Formation_Start': formation.index[0].isoformat(),
                          'Formation_End': formation.index[-1].isoformat(),
                          'Trading_Start': dates[0].isoformat(), 'Trading_End': dates[-1].isoformat()})
        if frozen_selection is None:
            frozen_selection = panel.loc[panel.Raw_Eligible].copy()
            first_formation_start, first_formation_end = formation.index[0], formation.index[-1]
        selections = {'fixed_selection': frozen_selection,
                      'monthly_raw': panel.loc[panel.Raw_Eligible],
                      'monthly_holm': panel.loc[panel.Holm_Eligible]}
        for name, selected in selections.items():
            storage = variant_storage[name]
            capital = storage['capital']
            if capital <= 0:
                raise ValueError(f'{name} exhausted its capital before window {number}; insolvency requires a separate model')
            result = _window_backtest(prices, dates, selected, capital, entry, exit, lookback,
                                      fee, borrow, max_hold)
            curve, trades = result['equity_curve'], result['trades'].copy()
            if number < len(windows) and not trades.empty:
                trades.loc[trades.Exit_Reason == 'end_of_data', 'Exit_Reason'] = 'month_end'
            trades['Variant'] = name
            trades['Window'] = number
            trades['Trading_Start'] = dates[0]
            trades['Trading_End'] = dates[-1]
            final = float(curve.iloc[-1])
            for pair in selected.Pair:
                storage['selected_counts'][pair] = storage['selected_counts'].get(pair, 0) + 1
            storage['curves'].append(curve)
            if not trades.empty:
                storage['logs'].append(trades)
            storage['monthly'].append({
                'Variant': name, 'Window': number,
                'Formation_Start': first_formation_start if name == 'fixed_selection' else formation.index[0],
                'Formation_End': first_formation_end if name == 'fixed_selection' else formation.index[-1],
                'Trading_Start': dates[0], 'Trading_End': dates[-1],
                'Selected_Pairs': len(selected), 'Starting_Capital': capital,
                'Ending_Capital': final, 'Net_PnL': final - capital,
                'Num_Trades': len(trades), 'Total_Costs': float(trades.Costs.sum()),
                'Return': final / capital - 1, 'Cash_Only': selected.empty,
            })
            storage['capital'] = final
    variants = {}
    for name, storage in variant_storage.items():
        curve = pd.concat(storage['curves']).rename('Portfolio_Value')
        trades = (pd.concat(storage['logs'], ignore_index=True) if storage['logs'] else
                  pd.DataFrame(columns=TRADE_COLUMNS + ['Portfolio_Weight', 'Variant', 'Window', 'Trading_Start', 'Trading_End']))
        monthly = pd.DataFrame(storage['monthly'])
        if not curve.index.equals(evaluation_dates):
            raise AssertionError('Every evaluation close must appear exactly once')
        if not np.allclose(monthly.Starting_Capital.iloc[1:], monthly.Ending_Capital.iloc[:-1], rtol=1e-12, atol=1e-8):
            raise AssertionError('Monthly capital must chain without resetting the account')
        if not np.isclose(trades.Net_PnL.sum(), curve.iloc[-1] - initial_capital, rtol=1e-10, atol=1e-7):
            raise AssertionError('All allocated trade P&L must reconcile to the complete account')
        records = []
        for pair, count in sorted(storage['selected_counts'].items()):
            subset = trades.loc[trades.Pair == pair]
            records.append({'Pair': pair, 'Net_PnL': float(subset.Net_PnL.sum()),
                            'Num_Trades': len(subset), 'Costs': float(subset.Costs.sum()),
                            'Selected_Windows': count})
        pair_profits = pd.DataFrame(records, columns=PAIR_PROFIT_COLUMNS)
        variants[name] = {'equity_curve': curve, 'trades': trades, 'monthly': monthly,
                          'pair_profits': pair_profits,
                          'metrics': performance_metrics(curve, trades, float(initial_capital))}
    screening = pd.concat(all_screening, ignore_index=True)
    return {'variants': variants, 'screening': screening, 'config': {
        'tickers': stocks.Ticker.tolist(), 'stock_count': len(stocks),
        'family_size': len(family), 'variant_names': list(VARIANT_NAMES),
        'scheduled_windows': scheduled, 'formation_days': formation_days,
        'evaluation_start': evaluation_dates[0].isoformat(), 'evaluation_end': evaluation_dates[-1].isoformat(),
        'initial_capital': float(initial_capital), 'entry': entry, 'exit': exit,
        'lookback': lookback, 'fee': fee, 'borrow': borrow, 'max_hold': max_hold,
        'selection_alpha': .05, 'adjustment': 'Holm per monthly prescribed family',
        'all_variants_liquidate_each_window': True,
        'variants': {
            'fixed_selection': 'Raw selection and betas frozen from the first formation; monthly forced closures',
            'monthly_raw': 'Raw p < .05 selection and beta refit each month',
            'monthly_holm': 'Monthly refit with Holm family correction at .05',
        },
        'correction_note': 'Untestable hypotheses retain raw p = NaN and use explicitly labeled correction input 1.0; correction does not cover repeated testing across months',
        'limitations': 'Fixed current peer universe and historical adjusted closes; provisional I(1) diagnostics, assumed borrowing/fractional shares, no liquidity or margin-call model; no untouched final holdout claim',
    }}
