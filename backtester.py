"""Two-leg, daily-close pairs backtesting with explicit cash and share accounting.

Signals observed at close t execute at close t+1. Each entry allocates the
initial capital as gross notional across both legs; quantities stay fixed until
exit. Costs apply to the actual notional bought/sold on both entry and exit.
Adjusted daily prices are a research proxy, not executable historical quotes.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADE_COLUMNS = [
    'Pair', 'Ticker1', 'Ticker2', 'Entry_Date', 'Exit_Date', 'Signal',
    'Hedge_Ratio', 'Quantity1', 'Quantity2', 'Entry_Price1', 'Entry_Price2',
    'Exit_Price1', 'Exit_Price2', 'Entry_Price', 'Exit_Price',
    'Entry_Gross_Notional', 'Exit_Gross_Notional', 'Gross_PnL', 'Entry_Cost',
    'Exit_Cost', 'Borrow_Cost', 'Costs', 'Net_PnL', 'PnL', 'Trade_Return',
    'Bars_Held', 'Exit_Reason',
]
# These columns are dollar or share amounts and must scale with pair allocation.
SCALED_TRADE_COLUMNS = [
    'Quantity1', 'Quantity2', 'Entry_Gross_Notional', 'Exit_Gross_Notional',
    'Gross_PnL', 'Entry_Cost', 'Exit_Cost', 'Borrow_Cost', 'Costs', 'Net_PnL', 'PnL',
]


def _validate_index(index):
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError('Prices must have a DatetimeIndex')
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError('Dates must be finite, unique, and sorted chronologically')


def performance_metrics(equity_values, trades, initial_capital):
    """Metrics for one account; turnover is annual gross traded notional/capital.

    Sharpe uses daily arithmetic excess returns (zero risk-free rate), while
    annualized return is CAGR over actual calendar time. Trade returns are net
    P&L / entry gross notional, including entry, exit and borrowing costs.
    """
    equity_values = pd.Series(equity_values, dtype=float)
    trades = pd.DataFrame(trades)
    final = float(equity_values.iloc[-1]) if len(equity_values) else initial_capital
    total_return = final / initial_capital - 1
    if len(equity_values) > 1:
        elapsed_years = (equity_values.index[-1] - equity_values.index[0]).total_seconds() / (365.25 * 86400)
    else:
        elapsed_years = 0.0
    annual_return = ((final / initial_capital) ** (1 / elapsed_years) - 1
                     if elapsed_years > 0 and final > 0 else (-1.0 if final <= 0 else 0.0))
    values = equity_values.to_numpy()
    previous = np.r_[initial_capital, values[:-1]]
    valid = previous > 0
    daily_returns = (values[valid] / previous[valid] - 1) if len(values) else np.array([])
    # The first valuation is the capital baseline, not an additional trading day.
    if len(values) and values[0] == initial_capital:
        daily_returns = daily_returns[1:]
    std = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    vol = std * np.sqrt(252)
    sharpe = float(np.mean(daily_returns) / std * np.sqrt(252)) if std > 0 else 0.0
    peaks = np.maximum.accumulate(np.r_[initial_capital, values])[1:]
    drawdown = float(np.min(values / peaks - 1)) if len(values) else 0.0

    def total(column):
        return float(trades[column].sum()) if column in trades else 0.0

    count = len(trades)
    net = trades['Net_PnL'] if 'Net_PnL' in trades else pd.Series(dtype=float)
    fees = total('Entry_Cost') + total('Exit_Cost')
    borrow = total('Borrow_Cost')
    gross_traded = total('Entry_Gross_Notional') + total('Exit_Gross_Notional')
    return {
        'Total_Return': float(total_return),
        'Annualized_Return': float(annual_return),
        'Volatility': float(vol), 'Sharpe_Ratio': sharpe,
        'Max_Drawdown': drawdown,
        'Win_Rate': float((net > 0).mean()) if count else 0.0,
        'Num_Trades': count,
        'Avg_Trade_PnL': float(net.mean()) if count else 0.0,
        'Avg_Trade_Return': float(trades['Trade_Return'].mean()) if count else 0.0,
        'Total_Transaction_Costs': fees,
        'Total_Borrow_Costs': borrow,
        'Total_Costs': fees + borrow,
        'Gross_Traded_Notional': gross_traded,
        'Turnover': gross_traded / initial_capital / elapsed_years if elapsed_years > 0 else 0.0,
        'Final_Portfolio_Value': final,
        'Initial_Capital': float(initial_capital),
    }


class PairBacktester:
    """Simulate fixed shares in both legs, with next-close signal execution."""

    def __init__(self, signals, initial_capital=100000, transaction_cost=0.001,
                 max_holding_period=None, annual_borrow_rate=0.0):
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.transaction_cost = float(transaction_cost)
        self.max_holding_period = max_holding_period
        self.annual_borrow_rate = float(annual_borrow_rate)
        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError('initial_capital must be finite and positive')
        if not np.isfinite(self.transaction_cost) or self.transaction_cost < 0:
            raise ValueError('transaction_cost must be finite and nonnegative')
        if not np.isfinite(self.annual_borrow_rate) or self.annual_borrow_rate < 0:
            raise ValueError('annual_borrow_rate must be finite and nonnegative')
        if max_holding_period is not None and (isinstance(max_holding_period, bool) or
                not isinstance(max_holding_period, (int, np.integer)) or max_holding_period < 1):
            raise ValueError('max_holding_period must be a positive integer of trading bars')
        self.trades = []
        self.equity_curve = None

    def run_backtest(self):
        """Run/reset the simulation, including liquidation on the last close."""
        self.trades = []
        self.equity_curve = None
        s = self.signals
        _validate_index(s.index)
        required = ['Price1', 'Price2', 'Signal']
        if any(c not in s for c in required):
            raise ValueError('Signals must contain Price1, Price2, and Signal')
        p1 = s['Price1'].to_numpy(dtype=float)
        p2 = s['Price2'].to_numpy(dtype=float)
        targets = s['Signal'].to_numpy(dtype=float)
        if (not np.isfinite(p1).all() or not np.isfinite(p2).all() or
                (p1 <= 0).any() or (p2 <= 0).any()):
            raise ValueError('Both price series must be finite and positive; missing prices are not tradable')
        if not np.isfinite(targets).all() or not np.isin(targets, [-1, 0, 1]).all():
            raise ValueError('Signal must be finite and one of -1, 0, 1')
        if 'Hedge_Ratio' in s:
            betas = s['Hedge_Ratio'].to_numpy(dtype=float)
        elif 'hedge_ratio' in s.attrs:
            betas = np.full(len(s), float(s.attrs['hedge_ratio']))
        elif 'Spread' in s:
            # Compatibility with old signal frames: Spread = Price1 - beta*Price2.
            betas = (p1 - s['Spread'].to_numpy(dtype=float)) / p2
        else:
            raise ValueError('Signals need Hedge_Ratio (or hedge_ratio metadata or Spread)')
        if not np.isfinite(betas).all():
            raise ValueError('Hedge ratios must be finite')
        ticker1 = s.attrs.get('ticker1', 'Price1')
        ticker2 = s.attrs.get('ticker2', 'Price2')
        pair = f'{ticker1}-{ticker2}'
        n = len(s)
        cash = self.initial_capital
        position = 0
        q1 = q2 = 0.0
        opened = None
        blocked_direction = 0
        result = {k: np.zeros(n, dtype=float) for k in [
            'Cash', 'Position', 'Quantity1', 'Quantity2', 'Spread_PnL',
            'Portfolio_Value', 'Gross_Exposure', 'Net_Exposure',
            'Transaction_Costs', 'Borrow_Costs',
        ]}
        result['Entry_Price'] = np.full(n, np.nan)
        entry_dates = [pd.NaT] * n
        reasons = s['Signal_Reason'].to_numpy() if 'Signal_Reason' in s else None
        dates = s.index
        calendar_days = np.r_[0.0, (dates[1:] - dates[:-1]).total_seconds() / 86400] if n else np.array([])
        for i in range(n):
            target = int(targets[i - 1]) if i > 0 else 0
            fees_today = borrow_today = 0.0
            if position:
                # Borrow accrues across weekends using the preceding close notional.
                short_notional = max(-q1, 0) * p1[i - 1] + max(-q2, 0) * p2[i - 1]
                borrow_today = short_notional * self.annual_borrow_rate * calendar_days[i] / 365.25
                cash -= borrow_today
                opened['Borrow_Cost'] += borrow_today
            if target != blocked_direction:
                blocked_direction = 0
            timed_out = (position != 0 and self.max_holding_period is not None and
                         i - opened['entry_index'] >= self.max_holding_period)
            final_close = i == n - 1
            must_exit = position and (target != position or timed_out or final_close)
            if must_exit:
                exit_gross = abs(q1) * p1[i] + abs(q2) * p2[i]
                exit_cost = exit_gross * self.transaction_cost
                fees_today += exit_cost
                cash += q1 * p1[i] + q2 * p2[i] - exit_cost
                gross_pnl = q1 * (p1[i] - opened['Entry_Price1']) + q2 * (p2[i] - opened['Entry_Price2'])
                total_cost = opened['Entry_Cost'] + exit_cost + opened['Borrow_Cost']
                net_pnl = gross_pnl - total_cost
                if final_close:
                    reason = 'end_of_data'
                elif timed_out:
                    reason = 'max_holding_period'
                elif target != 0:
                    reason = 'signal_reversal'
                else:
                    reason = str(reasons[i - 1]) if reasons is not None else 'signal_exit'
                    if not reason:
                        reason = 'signal_exit'
                self.trades.append({k: opened[k] for k in [
                    'Pair', 'Ticker1', 'Ticker2', 'Entry_Date', 'Signal', 'Hedge_Ratio',
                    'Quantity1', 'Quantity2', 'Entry_Price1', 'Entry_Price2', 'Entry_Price',
                    'Entry_Gross_Notional', 'Entry_Cost', 'Borrow_Cost',
                ]} | {
                    'Exit_Date': dates[i], 'Exit_Price1': p1[i], 'Exit_Price2': p2[i],
                    'Exit_Price': p1[i] - opened['Hedge_Ratio'] * p2[i],
                    'Exit_Gross_Notional': exit_gross, 'Gross_PnL': gross_pnl,
                    'Exit_Cost': exit_cost, 'Costs': total_cost, 'Net_PnL': net_pnl,
                    'PnL': net_pnl, 'Trade_Return': net_pnl / opened['Entry_Gross_Notional'],
                    'Bars_Held': i - opened['entry_index'], 'Exit_Reason': reason,
                })
                if timed_out:
                    # Wait for a new signal cycle; do not reopen yesterday's stale target.
                    blocked_direction = position
                position = 0
                q1 = q2 = 0.0
                opened = None
            if position == 0 and target and not final_close and target != blocked_direction and cash > 0:
                # Hedge ratio was known at signal time; shares are fixed at execution.
                beta = betas[i - 1]
                units = self.initial_capital / (p1[i] + abs(beta) * p2[i])
                q1 = target * units
                q2 = -target * beta * units
                entry_gross = abs(q1) * p1[i] + abs(q2) * p2[i]
                entry_cost = entry_gross * self.transaction_cost
                fees_today += entry_cost
                cash -= q1 * p1[i] + q2 * p2[i] + entry_cost
                position = target
                opened = {
                    'Pair': pair, 'Ticker1': ticker1, 'Ticker2': ticker2,
                    'Entry_Date': dates[i], 'Signal': position, 'Hedge_Ratio': beta,
                    'Quantity1': q1, 'Quantity2': q2,
                    'Entry_Price1': p1[i], 'Entry_Price2': p2[i],
                    'Entry_Price': p1[i] - beta * p2[i],
                    'Entry_Gross_Notional': entry_gross, 'Entry_Cost': entry_cost,
                    'Borrow_Cost': 0.0, 'entry_index': i,
                }
            market_value = q1 * p1[i] + q2 * p2[i]
            result['Cash'][i] = cash
            result['Position'][i] = position
            result['Quantity1'][i] = q1
            result['Quantity2'][i] = q2
            result['Portfolio_Value'][i] = cash + market_value
            result['Gross_Exposure'][i] = abs(q1) * p1[i] + abs(q2) * p2[i]
            result['Net_Exposure'][i] = market_value
            result['Transaction_Costs'][i] = fees_today
            result['Borrow_Costs'][i] = borrow_today
            if opened is not None:
                result['Entry_Price'][i] = opened['Entry_Price']
                entry_dates[i] = opened['Entry_Date']
                result['Spread_PnL'][i] = q1 * (p1[i] - opened['Entry_Price1']) + q2 * (p2[i] - opened['Entry_Price2'])
        self.equity_curve = pd.DataFrame(result, index=s.index)
        self.equity_curve['Entry_Date'] = pd.to_datetime(entry_dates)
        self.equity_curve.attrs['initial_capital'] = self.initial_capital
        return self.equity_curve

    def get_performance_metrics(self):
        if self.equity_curve is None:
            raise ValueError('Must run backtest first')
        return performance_metrics(self.equity_curve['Portfolio_Value'], self.get_trades_dataframe(), self.initial_capital)

    def get_trades_dataframe(self):
        """One row per completed roundtrip; PnL is an alias for Net_PnL."""
        return pd.DataFrame(self.trades, columns=TRADE_COLUMNS)


def backtest_all_pairs(signal_data, initial_capital=100000, transaction_cost=0.001,
                       max_holding_period=None, annual_borrow_rate=0.0):
    """Simulate each pair on a standalone account; combine later for allocation."""
    results = {}
    for pair_id, signals in signal_data.items():
        try:
            engine = PairBacktester(signals, initial_capital, transaction_cost,
                                    max_holding_period, annual_borrow_rate)
            engine.run_backtest()
            trades = engine.get_trades_dataframe()
            trades['Pair'] = pair_id
            results[pair_id] = {
                'metrics': engine.get_performance_metrics(),
                'equity_curve': engine.equity_curve, 'trades': trades,
                'initial_capital': float(initial_capital),
            }
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning('Error backtesting %s: %s', pair_id, exc)
    logger.info('Completed backtests for %d pairs', len(results))
    return results


def combine_portfolio_results(backtest_results, equal_weight=True, weights=None,
                              initial_capital=None):
    """Allocate one account across pair sleeves and scale both equity and trades.

    Weights are fixed at the start (no daily rebalancing); untraded sleeves stay
    in cash. Explicit weights must be chosen without the evaluation returns.
    Ex-post Sharpe weighting is disallowed because it uses future information.
    """
    if not backtest_results:
        return None
    ids = list(backtest_results)
    if weights is None:
        if not equal_weight:
            raise ValueError('Ex-post Sharpe weighting leaks future returns; supply ex-ante weights explicitly')
        weights = {key: 1.0 / len(ids) for key in ids}
    else:
        weights = {key: float(value) for key, value in weights.items()}
        if set(weights) != set(ids) or not all(np.isfinite(v) and v >= 0 for v in weights.values()):
            raise ValueError('Weights must be finite, nonnegative, and cover exactly the input pairs')
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError('Weights must sum to one')
    capitals = {}
    for pair_id, result in backtest_results.items():
        capital = result.get('initial_capital', result['metrics'].get('Initial_Capital'))
        if capital is None:
            capital = result['equity_curve'].attrs.get('initial_capital')
        if capital is None:
            raise ValueError('Every pair result must identify its initial capital')
        capitals[pair_id] = float(capital)
    capital = float(initial_capital) if initial_capital is not None else capitals[ids[0]]
    if not np.isfinite(capital) or capital <= 0 or any(not np.isfinite(v) or v <= 0 for v in capitals.values()):
        raise ValueError('Account initial capital must be finite and positive')
    scaled = {}
    trade_frames = []
    for pair_id, result in backtest_results.items():
        factor = weights[pair_id] * capital / capitals[pair_id]
        scaled[pair_id] = result['equity_curve']['Portfolio_Value'] * factor
        if not result['trades'].empty and factor > 0:
            trades = result['trades'].copy()
            trades['Pair'] = pair_id
            trades['Portfolio_Weight'] = weights[pair_id]
            for column in SCALED_TRADE_COLUMNS:
                if column in trades:
                    trades[column] *= factor
            trade_frames.append(trades)
    combined = pd.concat(scaled, axis=1).sort_index()
    # Before a sleeve's first valuation it is cash; after its last it stays cash.
    combined = combined.ffill().fillna({p: weights[p] * capital for p in ids})
    curve = combined.sum(axis=1).rename('Portfolio_Value')
    curve.attrs['initial_capital'] = capital
    trades = (pd.concat(trade_frames, ignore_index=True).sort_values(['Exit_Date', 'Pair']).reset_index(drop=True)
              if trade_frames else pd.DataFrame(columns=TRADE_COLUMNS + ['Portfolio_Weight']))
    return {
        'weights': weights, 'initial_capital': capital,
        'metrics': performance_metrics(curve, trades, capital),
        'equity_curve': curve, 'trades': trades,
    }
