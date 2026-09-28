"""One extra linear-regression strategy: a spread ECM forecast entry gate.

This is a single equation for an already estimated pair residual, not a full
multivariate VECM. Baseline selection, allocations, costs and exits are retained.
"""

import numpy as np
import pandas as pd

from backtester import PairBacktester, TRADE_COLUMNS, performance_metrics
from ols_improvement import (RULES as BASE_RULES, build_schedule, generate_variant_signals,
                             run_variant as run_ols_variant, select_pairs)

ECM_RULES = {
    'formation_days': 252, 'entry_z': 2.0, 'exit_z': .5,
    'forecast_entry_step': 1, 'forecast_exit_step': 6,
    'forecast_intervals_after_entry': 5, 'cost_multiple': 2.0,
    'estimated_borrow_calendar_days': 7, 'fee': .001, 'borrow': .02,
    'max_holding_bars': 20, 'require_negative_level_coefficient': True,
    'require_companion_eigenvalues_inside_unit_circle': True,
    'require_full_rank': True,
}
SOURCES = {
    'ols': 'https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html',
    'vecm_context': 'https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.vecm.VECM.html',
    'ar_root_convention': 'https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.AutoRegResults.roots.html',
}
MODEL_COLUMNS = [
    'ECM_Intercept', 'ECM_Level_Coeff', 'ECM_Delta_Coeff', 'ECM_Rank', 'ECM_Observations',
    'ECM_Residual_RMSE', 'AR2_Phi1', 'AR2_Phi2', 'Companion_Root1_Real', 'Companion_Root1_Imag',
    'Companion_Root2_Real', 'Companion_Root2_Imag', 'ECM_Max_Root_Modulus', 'ECM_Valid', 'ECM_Status',
]
GATE_COLUMNS = [
    'Forecast_Residual_t1', 'Forecast_Residual_t6', 'Projected_Signed_Move',
    'Projected_Move_Per_Gross', 'Estimated_Roundtrip_Cost', 'Entry_Gate_Passed',
]
PAIR_PROFIT_COLUMNS = ['Pair', 'Net_PnL', 'Num_Trades', 'Costs', 'Selected_Windows']


def companion_stability(level_coefficient, delta_coefficient):
    """Companion eigenvalues inside the unit circle equal AR roots outside it.

    Δs_t = a + k*s_(t-1) + g*Δs_(t-1) implies
    s_t = a + (1+k+g)*s_(t-1) - g*s_(t-2).
    """
    k, g = float(level_coefficient), float(delta_coefficient)
    if not np.isfinite([k, g]).all():
        return {'AR2_Phi1': np.nan, 'AR2_Phi2': np.nan, 'roots': np.array([np.nan, np.nan]),
                'max_modulus': np.nan, 'stable': False}
    phi1, phi2 = 1 + k + g, -g
    roots = np.roots([1., -phi1, -phi2])
    maximum = float(np.abs(roots).max())
    return {'AR2_Phi1': phi1, 'AR2_Phi2': phi2, 'roots': roots,
            'max_modulus': maximum, 'stable': bool(maximum < 1.)}


def fit_spread_ecm(residual):
    """Fit a three-column OLS model using only supplied formation residuals."""
    result = {column: np.nan for column in MODEL_COLUMNS}
    result.update(ECM_Valid=False, ECM_Status='invalid_input', ECM_Rank=0, ECM_Observations=0)
    values = np.asarray(residual, dtype=float)
    if values.ndim != 1 or len(values) < 10 or not np.isfinite(values).all():
        return result
    delta = np.diff(values)
    target = delta[1:]
    design = np.column_stack([np.ones(len(target)), values[1:-1], delta[:-1]])
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        result['ECM_Status'] = 'ols_failure'
        return result
    a, k, g = map(float, coefficients)
    result.update(ECM_Intercept=a, ECM_Level_Coeff=k, ECM_Delta_Coeff=g,
                  ECM_Rank=int(rank), ECM_Observations=len(target),
                  ECM_Residual_RMSE=float(np.sqrt(np.mean((target - design @ coefficients) ** 2))))
    dynamics = companion_stability(k, g)
    r1, r2 = dynamics['roots']
    result.update(AR2_Phi1=dynamics['AR2_Phi1'], AR2_Phi2=dynamics['AR2_Phi2'],
                  Companion_Root1_Real=float(np.real(r1)), Companion_Root1_Imag=float(np.imag(r1)),
                  Companion_Root2_Real=float(np.real(r2)), Companion_Root2_Imag=float(np.imag(r2)),
                  ECM_Max_Root_Modulus=dynamics['max_modulus'])
    if rank != 3:
        result['ECM_Status'] = 'rank_deficient'
    elif not np.isfinite(coefficients).all():
        result['ECM_Status'] = 'nonfinite_coefficients'
    elif k >= 0:
        result['ECM_Status'] = 'nonnegative_level_coefficient'
    elif not dynamics['stable']:
        result['ECM_Status'] = 'unstable_dynamics'
    else:
        result.update(ECM_Valid=True, ECM_Status='valid')
    return result


def forecast_residual(model, residual, delta, steps=6):
    """Conditional mean path at t+1,...,t+steps, with innovations set to zero."""
    a, k, g = (float(model[key]) for key in ('ECM_Intercept', 'ECM_Level_Coeff', 'ECM_Delta_Coeff'))
    if not isinstance(steps, (int, np.integer)) or isinstance(steps, bool) or steps < 1:
        raise ValueError('steps must be a positive integer')
    if not np.isfinite([a, k, g, residual, delta]).all():
        raise ValueError('Forecast coefficients and current residual/delta must be finite')
    value, difference = float(residual), float(delta)
    path = []
    for _ in range(steps):
        difference = a + k * value + g * difference
        value += difference
        path.append(value)
    return np.asarray(path)


def gate_diagnostics(model, residual, delta, zscore, price1, price2, beta):
    """Evaluate an entry forecast without counting the untradable t-to-t+1 move."""
    result = {column: np.nan for column in GATE_COLUMNS}
    result['Entry_Gate_Passed'] = False
    if (not bool(model.get('ECM_Valid', False)) or
            not np.isfinite([residual, delta, zscore, price1, price2, beta]).all() or
            price1 <= 0 or price2 <= 0 or beta <= 0):
        return result
    direction = 1 if zscore < -ECM_RULES['entry_z'] else (-1 if zscore > ECM_RULES['entry_z'] else 0)
    path = forecast_residual(model, residual, delta, ECM_RULES['forecast_exit_step'])
    predicted_entry = path[ECM_RULES['forecast_entry_step'] - 1]
    predicted_exit = path[ECM_RULES['forecast_exit_step'] - 1]
    gross = price1 + beta * price2
    # A short-spread entry shorts A; a long-spread entry shorts beta shares of B.
    short_fraction = price1 / gross if direction == -1 else beta * price2 / gross
    costs = (2 * ECM_RULES['fee'] + ECM_RULES['borrow'] *
             ECM_RULES['estimated_borrow_calendar_days'] / 365.25 * short_fraction)
    move = direction * (predicted_exit - predicted_entry)
    ratio = move / gross
    result.update(Forecast_Residual_t1=predicted_entry, Forecast_Residual_t6=predicted_exit,
                  Projected_Signed_Move=move, Projected_Move_Per_Gross=ratio,
                  Estimated_Roundtrip_Cost=costs,
                  Entry_Gate_Passed=bool(direction != 0 and ratio > ECM_RULES['cost_multiple'] * costs))
    return result


def apply_forecast_gate(signals, residuals, deltas, model):
    """Gate new entries only; the standard mean-reversion exits stay unchanged."""
    output = signals.copy()
    residuals, deltas = np.asarray(residuals), np.asarray(deltas)
    if residuals.shape != (len(signals),) or deltas.shape != (len(signals),):
        raise ValueError('Residual and delta vectors must align with signal rows')
    diagnostics, positions, reasons = [], [], []
    position = 0
    for i, row in enumerate(signals.itertuples()):
        gate = gate_diagnostics(model, residuals[i], deltas[i], row.ZScore, row.Price1, row.Price2, row.Hedge_Ratio)
        diagnostics.append(gate)
        reason = ''
        if not np.isfinite(row.ZScore):
            if position:
                reason = 'invalid_zscore'
            position = 0
        elif position == 0:
            if gate['Entry_Gate_Passed']:
                position = 1 if row.ZScore < -ECM_RULES['entry_z'] else -1
                reason = 'entry_ecm_forecast'
        elif ((position == 1 and row.ZScore >= -ECM_RULES['exit_z']) or
              (position == -1 and row.ZScore <= ECM_RULES['exit_z'])):
            position, reason = 0, 'mean_reversion'
        positions.append(position)
        reasons.append(reason)
    output['Signal'] = positions
    output['Signal_Reason'] = reasons
    output['Signal_Change'] = output.Signal.diff().fillna(output.Signal)
    output['Residual'] = residuals
    output['Residual_Delta'] = deltas
    for column in GATE_COLUMNS:
        output[column] = [row[column] for row in diagnostics]
    output.attrs.update(ecm_model=dict(model), entry_threshold=2., exit_threshold=.5)
    return output


def build_ecm_schedule(prices, metadata, start, end=None):
    """Append ECM diagnostics to the unchanged monthly OLS candidate schedule."""
    schedule = build_schedule(prices, metadata, start, end)
    for window in schedule:
        formation = prices.loc[window['formation_start']:window['formation_end']]
        rows = []
        for pair in window['panel'].itertuples():
            if pair.Raw_Eligible:
                residual = formation[pair.Ticker1] - pair.Alpha - pair.Beta * formation[pair.Ticker2]
                rows.append(fit_spread_ecm(residual))
            else:
                model = {column: np.nan for column in MODEL_COLUMNS}
                model.update(ECM_Valid=False, ECM_Status='not_raw_eligible')
                rows.append(model)
        window['panel'] = pd.concat([window['panel'].reset_index(drop=True),
                                     pd.DataFrame(rows, columns=MODEL_COLUMNS)], axis=1)
    return schedule


def run_ecm_variant(prices, schedule, variant='ecm_gate', initial_capital=100000):
    """Keep the baseline unchanged or run exactly the additional ECM entry gate."""
    if variant == 'baseline':
        result = run_ols_variant(prices, schedule, 'baseline', initial_capital)
        result['forecast_diagnostics'] = pd.DataFrame()
        return result
    if variant != 'ecm_gate':
        raise ValueError('variant must be baseline or ecm_gate')
    if not schedule or not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError('A schedule and positive finite initial capital are required')
    capital = float(initial_capital)
    curves, logs, monthly_rows, selections, forecasts = [], [], [], [], []
    counts = {}
    previous_end = None
    for number, window in enumerate(schedule, start=1):
        dates = window['dates']
        if (len(dates) == 0 or not dates.is_unique or not dates.is_monotonic_increasing or
                window['formation_end'] >= dates[0] or (previous_end is not None and dates[0] <= previous_end)):
            raise ValueError('Trading windows must be disjoint and strictly after formation')
        if capital <= 0:
            raise ValueError('Account has exhausted capital; insolvency requires a separate model')
        previous_end = dates[-1]
        selected = select_pairs(window['panel'], 'baseline')
        n = len(selected)
        weight = 1 / n if n else 0.
        curve = pd.Series(0. if n else capital, index=dates, name='Portfolio_Value')
        month_logs = []
        for row in selected.itertuples():
            counts[row.Pair] = counts.get(row.Pair, 0) + 1
            model = {column: getattr(row, column) for column in MODEL_COLUMNS}
            signals = generate_variant_signals(prices, dates, row.Ticker1, row.Ticker2, row.Beta, 'baseline')
            residual = prices[row.Ticker1] - row.Alpha - row.Beta * prices[row.Ticker2]
            delta = residual.diff()
            signals = apply_forecast_gate(signals, residual.loc[dates].to_numpy(), delta.loc[dates].to_numpy(), model)
            diagnostics = signals.reset_index(names='Date')
            diagnostics['Pair'], diagnostics['Window'], diagnostics['Variant'] = row.Pair, number, variant
            for column in MODEL_COLUMNS:
                diagnostics[column] = model[column]
            forecasts.append(diagnostics)
            selections.append({'Variant': variant, 'Window': number, 'Pair': row.Pair,
                               'Ticker1': row.Ticker1, 'Ticker2': row.Ticker2, 'Beta': row.Beta, 'Alpha': row.Alpha,
                               'Formation_Start': window['formation_start'], 'Formation_End': window['formation_end'],
                               'Trading_Start': dates[0], 'Trading_End': dates[-1], 'Raw_P': row.Raw_P,
                               'Weight': weight, 'Starting_Allocation': capital * weight, **model})
            engine = PairBacktester(signals, capital * weight, ECM_RULES['fee'],
                                    ECM_RULES['max_holding_bars'], ECM_RULES['borrow'])
            account = engine.run_backtest()
            if account.Gross_Exposure.iloc[-1] != 0:
                raise AssertionError('Every monthly sleeve must finish flat')
            curve += account.Portfolio_Value
            trades = engine.get_trades_dataframe()
            if not trades.empty:
                for trade in trades.itertuples():
                    decision_index = dates.get_loc(trade.Entry_Date) - 1
                    if decision_index < 0 or not signals.Entry_Gate_Passed.iloc[decision_index]:
                        raise AssertionError('Every entry must follow a passing previous-close forecast gate')
                trades['Pair'], trades['Variant'], trades['Window'] = row.Pair, variant, number
                trades['Portfolio_Weight'] = weight
                trades['Trading_Start'], trades['Trading_End'] = dates[0], dates[-1]
                if number < len(schedule):
                    trades.loc[trades.Exit_Reason == 'end_of_data', 'Exit_Reason'] = 'month_end'
                month_logs.append(trades)
        month_trades = (pd.concat(month_logs, ignore_index=True) if month_logs else
                        pd.DataFrame(columns=TRADE_COLUMNS + ['Portfolio_Weight', 'Variant', 'Window', 'Trading_Start', 'Trading_End']))
        final = float(curve.iloc[-1])
        if not np.isclose(month_trades.Net_PnL.sum(), final - capital, rtol=1e-10, atol=1e-7):
            raise AssertionError('Monthly allocated P&L must reconcile to account change')
        monthly_rows.append({'Variant': variant, 'Window': number, 'Formation_Start': window['formation_start'],
                             'Formation_End': window['formation_end'], 'Trading_Start': dates[0], 'Trading_End': dates[-1],
                             'Selected_Pairs': n, 'Valid_ECM_Models': int(selected.ECM_Valid.sum()) if n else 0,
                             'Weight_Per_Pair': weight, 'Starting_Capital': capital, 'Ending_Capital': final,
                             'Net_PnL': final - capital, 'Num_Trades': len(month_trades),
                             'Total_Costs': float(month_trades.Costs.sum()), 'Return': final / capital - 1})
        curves.append(curve)
        if not month_trades.empty:
            logs.append(month_trades)
        capital = final
    equity = pd.concat(curves).rename('Portfolio_Value')
    trades = (pd.concat(logs, ignore_index=True) if logs else
              pd.DataFrame(columns=TRADE_COLUMNS + ['Portfolio_Weight', 'Variant', 'Window', 'Trading_Start', 'Trading_End']))
    monthly = pd.DataFrame(monthly_rows)
    if not np.allclose(monthly.Starting_Capital.iloc[1:], monthly.Ending_Capital.iloc[:-1], rtol=1e-12, atol=1e-8):
        raise AssertionError('Monthly account capital must carry forward')
    if not np.isclose(trades.Net_PnL.sum(), equity.iloc[-1] - initial_capital, rtol=1e-10, atol=1e-7):
        raise AssertionError('All allocated trade profits must reconcile to total account change')
    records = []
    for pair, count in sorted(counts.items()):
        log = trades.loc[trades.Pair == pair]
        records.append({'Pair': pair, 'Net_PnL': float(log.Net_PnL.sum()), 'Num_Trades': len(log),
                        'Costs': float(log.Costs.sum()), 'Selected_Windows': count})
    return {'equity_curve': equity, 'trades': trades, 'monthly': monthly,
            'pair_profits': pd.DataFrame(records, columns=PAIR_PROFIT_COLUMNS),
            'selections': pd.DataFrame(selections),
            'forecast_diagnostics': pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame(),
            'metrics': performance_metrics(equity, trades, initial_capital),
            'settings': dict(ECM_RULES, variant=variant, initial_capital=initial_capital)}
