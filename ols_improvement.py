"""Five prespecified monthly OLS variants, evaluated without changing the baseline.

Only earlier formation prices select pairs and fit hedge ratios. Entry quality
and risk rules are hypotheses to evaluate, not assurances of positive returns.
All execution occurs on a subsequent close and every month finishes flat.
"""

import numpy as np
import pandas as pd

from backtester import PairBacktester, TRADE_COLUMNS, performance_metrics
from signal_generation import PairSignalGenerator
from walk_forward import _metadata_frame, prescribe_pair_family, screen_formation

# Fixed before validation or later-period results are read.
RULES = {
    'formation_days': 252, 'lookback': 60, 'entry': 2.0, 'exit': .5,
    'fee': .001, 'borrow': .02, 'baseline_max_hold': 20, 'risk_max_hold': 10,
    'risk_stop_zscore': 3.5, 'risk_max_pair_weight': .2,
    'half_life_min': 2.0, 'half_life_max': 20.0, 'beta_instability_max': .5,
    'convergence_cost_multiple': 2.0, 'estimated_borrow_calendar_days': 28,
}
VARIANTS = {
    'baseline': {'label': 'Monthly OLS baseline', 'quality': False, 'entry_filter': False, 'risk_limits': False},
    'quality': {'label': 'OLS relationship quality', 'quality': True, 'entry_filter': False, 'risk_limits': False},
    'entry_filter': {'label': 'OLS with entry confirmation and cost hurdle', 'quality': False, 'entry_filter': True, 'risk_limits': False},
    'risk_limits': {'label': 'OLS with exposure and exit limits', 'quality': False, 'entry_filter': False, 'risk_limits': True},
    'combined': {'label': 'OLS with all three changes', 'quality': True, 'entry_filter': True, 'risk_limits': True},
}
QUALITY_COLUMNS = ['AR1_Phi', 'Half_Life', 'Beta_First_Half', 'Beta_Last_Half',
                   'Beta_Instability', 'Quality_Eligible', 'Quality_Status']
PAIR_PROFIT_COLUMNS = ['Pair', 'Net_PnL', 'Num_Trades', 'Costs', 'Selected_Windows']


def ols_fit(a, b):
    """Fit A = intercept + beta * B, rejecting unidentified/nonfinite inputs."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('OLS needs matching complete finite vectors with at least three rows')
    matrix = np.column_stack([np.ones(len(b)), b])
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, a, rcond=None)
    if rank < 2:
        raise ValueError('OLS predictor does not identify an intercept and slope')
    return float(coefficients[0]), float(coefficients[1])


def residual_half_life(residual):
    """OLS AR(1) with intercept; half-life is defined only for 0 < phi < 1."""
    values = np.asarray(residual, dtype=float)
    if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
        raise ValueError('AR(1) needs at least four finite residual observations')
    try:
        _, phi = ols_fit(values[1:], values[:-1])
    except ValueError:
        return {'AR1_Phi': np.nan, 'Half_Life': np.nan}
    half_life = -np.log(2) / np.log(phi) if 0 < phi < 1 else np.nan
    return {'AR1_Phi': float(phi), 'Half_Life': float(half_life)}


def quality_diagnostics(formation, ticker1, ticker2, beta=None, alpha=None):
    """Formation-only residual persistence and half-window OLS slope stability."""
    result = {key: np.nan for key in QUALITY_COLUMNS}
    result.update(Quality_Eligible=False, Quality_Status='invalid_formation')
    try:
        a, b = formation[ticker1].to_numpy(dtype=float), formation[ticker2].to_numpy(dtype=float)
        if len(a) < 8 or (a <= 0).any() or (b <= 0).any():
            raise ValueError('Formation must contain enough strictly positive prices')
        if beta is None or alpha is None:
            alpha, beta = ols_fit(a, b)
        if not np.isfinite(beta) or not np.isfinite(alpha) or beta <= 0:
            result['Quality_Status'] = 'nonpositive_or_invalid_full_beta'
            return result
        residual = a - float(alpha) - float(beta) * b
        result.update(residual_half_life(residual))
        half = len(formation) // 2
        _, first_beta = ols_fit(a[:half], b[:half])
        _, last_beta = ols_fit(a[-half:], b[-half:])
        instability = max(abs(first_beta / beta - 1), abs(last_beta / beta - 1))
        result.update(Beta_First_Half=first_beta, Beta_Last_Half=last_beta,
                      Beta_Instability=float(instability))
        persistence_ok = (0 < result['AR1_Phi'] < 1 and
                          RULES['half_life_min'] <= result['Half_Life'] <= RULES['half_life_max'])
        stability_ok = first_beta > 0 and last_beta > 0 and instability <= RULES['beta_instability_max']
        result['Quality_Eligible'] = bool(persistence_ok and stability_ok)
        result['Quality_Status'] = ('eligible' if result['Quality_Eligible'] else
                                    ('half_life_outside_bounds' if not persistence_ok else 'unstable_beta'))
    except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
        result['Quality_Status'] = f'invalid_formation: {exc}'
    return result


def build_schedule(prices, metadata, start, end=None):
    """Build raw candidate/quality panels from 252 closes before each month.

    The first/last calendar months may be partial. The specified fixed peer
    family is constructed from metadata before any cointegration test is run.
    Returned dictionaries contain dates, formation_start/end, panel, and window.
    """
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError('Price dates must be complete, unique and chronological')
    if prices.columns.duplicated().any():
        raise ValueError('Price tickers must be unique')
    stocks = _metadata_frame(metadata)
    if not set(stocks.Ticker).issubset(prices.columns):
        raise ValueError('Every prescribed universe ticker must have a price column')
    family = prescribe_pair_family(metadata)
    beginning = pd.Timestamp(start)
    ending = pd.Timestamp(end) if end is not None else prices.index[-1]
    if prices.index.tz is not None:
        if beginning.tzinfo is None:
            beginning = beginning.tz_localize(prices.index.tz)
        if ending.tzinfo is None:
            ending = ending.tz_localize(prices.index.tz)
    dates = prices.index[(prices.index >= beginning) & (prices.index <= ending)]
    if len(dates) == 0:
        raise ValueError('The requested evaluation window contains no price rows')
    if prices.index.get_loc(dates[0]) < RULES['formation_days']:
        raise ValueError('Need 252 preceding formation closes before the first trading window')
    labels = dates.strftime('%Y-%m')
    schedule = []
    for number, label in enumerate(dict.fromkeys(labels), start=1):
        trading_dates = dates[labels == label]
        location = prices.index.get_loc(trading_dates[0])
        formation = prices.loc[:, stocks.Ticker].iloc[location - RULES['formation_days']:location]
        if formation.index[-1] >= trading_dates[0]:
            raise AssertionError('Formation must end strictly before the trading window')
        panel = screen_formation(formation, family)
        diagnostics = []
        for row in panel.itertuples():
            if row.Raw_Eligible:
                diagnostics.append(quality_diagnostics(formation, row.Ticker1, row.Ticker2,
                                                       beta=row.Beta, alpha=row.Alpha))
            else:
                info = {key: np.nan for key in QUALITY_COLUMNS}
                info.update(Quality_Eligible=False, Quality_Status='not_raw_eligible')
                diagnostics.append(info)
        quality = pd.DataFrame(diagnostics, columns=QUALITY_COLUMNS)
        panel = pd.concat([panel.reset_index(drop=True), quality], axis=1)
        panel['Window'] = number
        panel['Formation_Start'] = formation.index[0]
        panel['Formation_End'] = formation.index[-1]
        panel['Trading_Start'] = trading_dates[0]
        panel['Trading_End'] = trading_dates[-1]
        schedule.append({'window': number, 'dates': trading_dates, 'formation_start': formation.index[0],
                         'formation_end': formation.index[-1], 'panel': panel,
                         'family_size': len(family), 'stock_count': len(stocks)})
    return schedule


def select_pairs(panel, variant):
    """Select from formation results only; risk variants prohibit ticker reuse."""
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    rule = VARIANTS[variant]
    selected = panel.loc[panel.Raw_Eligible].copy()
    if rule['quality']:
        selected = selected.loc[selected.Quality_Eligible]
    selected = selected.sort_values(['Raw_P', 'Pair']).reset_index(drop=True)
    if rule['risk_limits']:
        indices, used = [], set()
        for index, row in selected.iterrows():
            if row.Ticker1 not in used and row.Ticker2 not in used:
                indices.append(index)
                used.update([row.Ticker1, row.Ticker2])
        selected = selected.loc[indices].reset_index(drop=True)
    return selected


def entry_filter_diagnostics(z, previous_z, spread_std, price1, price2, beta):
    """A convergence/cost hurdle, not a predicted return or guarantee of profit.

    The 28-calendar-day borrowing horizon stays fixed across variants. The
    estimated convergence movement does not account for execution delay or
    model failure. The actual backtest uses observed subsequent prices.
    """
    values = [z, previous_z, spread_std, price1, price2, beta]
    if not np.isfinite(values).all() or spread_std <= 0 or price1 <= 0 or price2 <= 0 or beta <= 0:
        return {'Entry_Confirmation_Passed': False, 'Convergence_Move_Per_Gross': np.nan,
                'Estimated_Roundtrip_Cost': np.nan, 'Entry_Filter_Passed': False}
    gross = price1 + beta * price2
    short_fraction = beta * price2 / gross if z < 0 else price1 / gross
    movement = (abs(z) - RULES['exit']) * spread_std / gross
    costs = 2 * RULES['fee'] + RULES['borrow'] * (RULES['estimated_borrow_calendar_days'] / 365.25) * short_fraction
    confirmed = abs(z) > RULES['entry'] and z * previous_z > 0 and abs(z) < abs(previous_z)
    return {'Entry_Confirmation_Passed': bool(confirmed),
            'Convergence_Move_Per_Gross': float(movement), 'Estimated_Roundtrip_Cost': float(costs),
            'Entry_Filter_Passed': bool(confirmed and movement >= RULES['convergence_cost_multiple'] * costs)}


def signals_from_statistics(signals, previous_z, *, use_entry_filter=False, stop_zscore=None):
    """Apply local entry/exit state logic; entry filters never block an exit.

    Input rolling statistics already use preceding closes. Returned positions
    are decisions, not executions; PairBacktester applies the one-close delay.
    """
    signals = signals.copy()
    previous = np.asarray(previous_z, dtype=float)
    if previous.shape != (len(signals),):
        raise ValueError('Previous z-scores must align exactly with signal rows')
    position = 0
    positions, reasons, diagnostics = [], [], []
    values = signals[['ZScore', 'Rolling_Std', 'Price1', 'Price2', 'Hedge_Ratio']].to_numpy(dtype=float)
    for index, (z, std, p1, p2, beta) in enumerate(values):
        diagnostic = entry_filter_diagnostics(z, previous[index], std, p1, p2, beta)
        diagnostics.append(diagnostic)
        reason = ''
        if not np.isfinite(z):
            if position:
                reason = 'invalid_zscore'
            position = 0
        elif position == 0:
            within_stop = stop_zscore is None or abs(z) < stop_zscore
            entry_allowed = not use_entry_filter or diagnostic['Entry_Filter_Passed']
            if within_stop and entry_allowed:
                if z < -RULES['entry']:
                    position = 1
                elif z > RULES['entry']:
                    position = -1
                if position:
                    reason = 'entry'
        else:
            stopped = stop_zscore is not None and ((position == 1 and z <= -stop_zscore) or
                                                  (position == -1 and z >= stop_zscore))
            reverted = ((position == 1 and z >= -RULES['exit']) or
                        (position == -1 and z <= RULES['exit']))
            if stopped or reverted:
                reason = 'stop_zscore' if stopped else 'mean_reversion'
                position = 0
        positions.append(position)
        reasons.append(reason)
    signals['Signal'] = positions
    signals['Signal_Reason'] = reasons
    signals['Signal_Change'] = signals.Signal.diff().fillna(signals.Signal)
    signals['Previous_ZScore'] = previous
    for key in ['Entry_Confirmation_Passed', 'Convergence_Move_Per_Gross',
                'Estimated_Roundtrip_Cost', 'Entry_Filter_Passed']:
        signals[key] = [item[key] for item in diagnostics]
    signals.attrs.update(entry_threshold=RULES['entry'], exit_threshold=RULES['exit'],
                         stop_zscore=stop_zscore, use_entry_filter=use_entry_filter)
    return signals


def generate_variant_signals(prices, dates, ticker1, ticker2, beta, variant):
    """Build a flat-start monthly signal stream with 61 prior closes of history."""
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    rule = VARIANTS[variant]
    history = prices.loc[:dates[-1]].tail(len(dates) + RULES['lookback'] + 1)
    generator = PairSignalGenerator(history, ticker1, ticker2, hedge_ratio=beta, lookback=RULES['lookback'])
    generator.calculate_zscore()
    previous_z = generator.z_score.shift(1).loc[dates].to_numpy()
    for attribute in ('price_data', 'spread', 'z_score', 'moving_mean', 'moving_std'):
        setattr(generator, attribute, getattr(generator, attribute).loc[dates])
    signals = generator.get_signals(entry_threshold=RULES['entry'], exit_threshold=RULES['exit'])
    return signals_from_statistics(signals, previous_z, use_entry_filter=rule['entry_filter'],
                                   stop_zscore=RULES['risk_stop_zscore'] if rule['risk_limits'] else None)


def run_variant(prices, schedule, variant, initial_capital=100000, signal_transform=None):
    """Run one fixed rule set; allocations and idle cash share one real account."""
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant: {variant}')
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError('initial_capital must be finite and positive')
    if not schedule:
        raise ValueError('An evaluation schedule is required')
    rule = VARIANTS[variant]
    capital = float(initial_capital)
    curves, logs, month_rows, selection_rows = [], [], [], []
    selected_counts = {}
    previous_end = None
    for number, window in enumerate(schedule, start=1):
        dates, panel = window['dates'], window['panel']
        if (len(dates) == 0 or not dates.is_unique or not dates.is_monotonic_increasing or
                window['formation_end'] >= dates[0] or
                (previous_end is not None and dates[0] <= previous_end)):
            raise ValueError('Schedule must be chronological, disjoint, and strictly after formation')
        if capital <= 0:
            raise ValueError('Account has exhausted capital; insolvency needs a separate model')
        previous_end = dates[-1]
        selected = select_pairs(panel, variant)
        count = len(selected)
        weight = min(1 / count, RULES['risk_max_pair_weight']) if count and rule['risk_limits'] else (1 / count if count else 0.)
        cash_weight = max(0., 1. - count * weight)
        monthly_curve = pd.Series(capital * cash_weight, index=dates, name='Portfolio_Value', dtype=float)
        monthly_logs = []
        for row in selected.itertuples():
            allocation = capital * weight
            selected_counts[row.Pair] = selected_counts.get(row.Pair, 0) + 1
            selection_rows.append({'Variant': variant, 'Window': number, 'Pair': row.Pair,
                                   'Ticker1': row.Ticker1, 'Ticker2': row.Ticker2,
                                   'Formation_Start': window['formation_start'], 'Formation_End': window['formation_end'],
                                   'Trading_Start': dates[0], 'Trading_End': dates[-1], 'Beta': row.Beta,
                                   'Raw_P': row.Raw_P, 'Weight': weight, 'Starting_Allocation': allocation})
            signals = generate_variant_signals(prices, dates, row.Ticker1, row.Ticker2, row.Beta, variant)
            if signal_transform is not None:
                signals = signal_transform(signals.copy(), prices, window, row)
                if not signals.index.equals(dates):
                    raise ValueError('Transformed signals must preserve trading dates')
            engine = PairBacktester(signals, initial_capital=allocation, transaction_cost=RULES['fee'],
                                    annual_borrow_rate=RULES['borrow'],
                                    max_holding_period=RULES['risk_max_hold'] if rule['risk_limits'] else RULES['baseline_max_hold'])
            curve = engine.run_backtest()
            if curve.Gross_Exposure.iloc[-1] != 0:
                raise AssertionError('Each month must end with both legs closed')
            monthly_curve += curve.Portfolio_Value
            trades = engine.get_trades_dataframe()
            if not trades.empty:
                trades['Pair'] = row.Pair
                trades['Portfolio_Weight'] = weight
                trades['Variant'] = variant
                trades['Window'] = number
                trades['Trading_Start'] = dates[0]
                trades['Trading_End'] = dates[-1]
                if number < len(schedule):
                    trades.loc[trades.Exit_Reason == 'end_of_data', 'Exit_Reason'] = 'month_end'
                monthly_logs.append(trades)
        monthly_trades = (pd.concat(monthly_logs, ignore_index=True) if monthly_logs else
                          pd.DataFrame(columns=TRADE_COLUMNS + ['Portfolio_Weight', 'Variant', 'Window', 'Trading_Start', 'Trading_End']))
        final = float(monthly_curve.iloc[-1])
        if not np.isclose(monthly_trades.Net_PnL.sum(), final - capital, rtol=1e-10, atol=1e-7):
            raise AssertionError('Allocated trade profit does not reconcile to monthly capital')
        month_rows.append({
            'Variant': variant, 'Window': number, 'Formation_Start': window['formation_start'],
            'Formation_End': window['formation_end'], 'Trading_Start': dates[0], 'Trading_End': dates[-1],
            'Eligible_Pairs': int(panel.Raw_Eligible.sum()), 'Selected_Pairs': count,
            'Weight_Per_Pair': weight, 'Cash_Weight': cash_weight,
            'Starting_Capital': capital, 'Ending_Capital': final, 'Net_PnL': final - capital,
            'Num_Trades': len(monthly_trades), 'Total_Costs': float(monthly_trades.Costs.sum()),
            'Return': final / capital - 1,
        })
        curves.append(monthly_curve)
        if not monthly_trades.empty:
            logs.append(monthly_trades)
        capital = final
    equity = pd.concat(curves).rename('Portfolio_Value')
    trades = (pd.concat(logs, ignore_index=True) if logs else
              pd.DataFrame(columns=TRADE_COLUMNS + ['Portfolio_Weight', 'Variant', 'Window', 'Trading_Start', 'Trading_End']))
    monthly = pd.DataFrame(month_rows)
    if not np.allclose(monthly.Starting_Capital.iloc[1:], monthly.Ending_Capital.iloc[:-1], rtol=1e-12, atol=1e-8):
        raise AssertionError('Monthly account capital must carry forward')
    if not np.isclose(trades.Net_PnL.sum(), equity.iloc[-1] - initial_capital, rtol=1e-10, atol=1e-7):
        raise AssertionError('All allocated trade profits must reconcile to the total account')
    records = []
    for pair, count in sorted(selected_counts.items()):
        log = trades.loc[trades.Pair == pair]
        records.append({'Pair': pair, 'Net_PnL': float(log.Net_PnL.sum()), 'Num_Trades': len(log),
                        'Costs': float(log.Costs.sum()), 'Selected_Windows': count})
    return {'equity_curve': equity, 'trades': trades, 'monthly': monthly,
            'pair_profits': pd.DataFrame(records, columns=PAIR_PROFIT_COLUMNS),
            'selections': pd.DataFrame(selection_rows, columns=[
                'Variant', 'Window', 'Pair', 'Ticker1', 'Ticker2', 'Formation_Start', 'Formation_End',
                'Trading_Start', 'Trading_End', 'Beta', 'Raw_P', 'Weight', 'Starting_Allocation']),
            'metrics': performance_metrics(equity, trades, float(initial_capital)),
            'settings': {'variant': variant, **rule, **RULES, 'initial_capital': float(initial_capital)}}
