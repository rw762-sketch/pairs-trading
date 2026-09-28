"""Pairs signals computed from prior-window statistics, executed on the next close."""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PairSignalGenerator:
    """A positive signal owns stock 1 and shorts beta shares of stock 2."""

    def __init__(self, price_data, ticker1, ticker2, hedge_ratio=None, lookback=60):
        if not isinstance(lookback, (int, np.integer)) or isinstance(lookback, bool) or lookback < 2:
            raise ValueError('lookback must be an integer of at least two trading observations')
        if (not isinstance(price_data.index, pd.DatetimeIndex) or price_data.index.hasnans or
                not price_data.index.is_unique or not price_data.index.is_monotonic_increasing):
            raise ValueError('Price dates must be unique, finite, chronological DatetimeIndex values')
        self.price_data = price_data[[ticker1, ticker2]].copy()
        values = self.price_data.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('Prices must be finite and positive; clean missing observations first')
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.lookback = lookback
        self._estimated_beta = hedge_ratio is None
        self.hedge_ratio = self._estimate_hedge_ratio() if hedge_ratio is None else float(hedge_ratio)
        if not np.isfinite(self.hedge_ratio):
            raise ValueError('hedge_ratio must be finite')
        self.spread = self.z_score = self.moving_mean = self.moving_std = None

    def _estimate_hedge_ratio(self):
        """Fit on the FIRST lookback observations, before allowing any signal."""
        fit = self.price_data.iloc[:self.lookback]
        if len(fit) < 2:
            return 0.0  # An empty/one-row series will have no tradable z-scores.
        x = np.column_stack([np.ones(len(fit)), fit[self.ticker2].to_numpy()])
        return float(np.linalg.lstsq(x, fit[self.ticker1].to_numpy(), rcond=None)[0][1])

    def calculate_spread(self):
        self.spread = self.price_data[self.ticker1] - self.hedge_ratio * self.price_data[self.ticker2]
        return self.spread

    def calculate_zscore(self, lookback=None):
        """Today's spread relative to mean/std of the PRECEDING lookback closes."""
        window = self.lookback if lookback is None else lookback
        if not isinstance(window, (int, np.integer)) or isinstance(window, bool) or window < 2:
            raise ValueError('lookback must be an integer of at least two')
        if self.spread is None:
            self.calculate_spread()
        self.moving_mean = self.spread.rolling(window=window, min_periods=window).mean().shift(1)
        self.moving_std = self.spread.rolling(window=window, min_periods=window).std().shift(1)
        self.z_score = (self.spread - self.moving_mean) / self.moving_std.replace(0.0, np.nan)
        self.z_score = self.z_score.replace([np.inf, -np.inf], np.nan)
        if self._estimated_beta:
            self.z_score.iloc[:self.lookback] = np.nan
        return self.z_score

    def get_signals(self, entry_threshold=2.0, exit_threshold=0.0,
                    stop_zscore=None, max_holding_period=None):
        """Desired positions known at each close (the backtester applies one lag).

        Long exits at z >= -exit_threshold and short at z <= +exit_threshold.
        Optional stops are adverse z-score crossings. Optional holding limits
        count trading intervals; the next-close execution delay applies to exits
        as it does to entries.
        """
        if not np.isfinite(entry_threshold) or not np.isfinite(exit_threshold) or not 0 <= exit_threshold < entry_threshold:
            raise ValueError('Require 0 <= exit_threshold < entry_threshold')
        if stop_zscore is not None and (not np.isfinite(stop_zscore) or stop_zscore <= entry_threshold):
            raise ValueError('stop_zscore must exceed entry_threshold')
        if max_holding_period is not None and (not isinstance(max_holding_period, (int, np.integer)) or
                isinstance(max_holding_period, bool) or max_holding_period < 1):
            raise ValueError('max_holding_period must be a positive integer')
        if self.z_score is None:
            self.calculate_zscore()
        n = len(self.price_data)
        positions = np.zeros(n, dtype=int)
        reasons = [''] * n
        position = 0
        entry_index = None
        for i, z in enumerate(self.z_score.to_numpy()):
            if not np.isfinite(z):
                if position:
                    reasons[i] = 'invalid_zscore'
                position = 0
                entry_index = None
            elif position == 0:
                # Do not enter a position already beyond its own stop boundary.
                within_stop = stop_zscore is None or abs(z) < stop_zscore
                if within_stop and z < -entry_threshold:
                    position = 1
                elif within_stop and z > entry_threshold:
                    position = -1
                if position:
                    entry_index = i
                    reasons[i] = 'entry'
            else:
                reverted = (position == 1 and z >= -exit_threshold) or (position == -1 and z <= exit_threshold)
                stopped = stop_zscore is not None and ((position == 1 and z <= -stop_zscore) or
                                                      (position == -1 and z >= stop_zscore))
                timed_out = max_holding_period is not None and i - entry_index >= max_holding_period
                if stopped or timed_out or reverted:
                    reasons[i] = 'stop_zscore' if stopped else ('max_holding_period' if timed_out else 'mean_reversion')
                    position = 0
                    entry_index = None
            positions[i] = position
        signals = pd.DataFrame({
            'Price1': self.price_data[self.ticker1],
            'Price2': self.price_data[self.ticker2],
            'Hedge_Ratio': self.hedge_ratio, 'Spread': self.spread,
            'Rolling_Mean': self.moving_mean, 'Rolling_Std': self.moving_std,
            'ZScore': self.z_score, 'Signal': positions, 'Signal_Reason': reasons,
        }, index=self.price_data.index)
        signals['Signal_Change'] = signals['Signal'].diff().fillna(signals['Signal'])
        signals.attrs.update({
            'ticker1': self.ticker1, 'ticker2': self.ticker2,
            'hedge_ratio': self.hedge_ratio, 'lookback': self.lookback,
            'entry_threshold': entry_threshold, 'exit_threshold': exit_threshold,
            'stop_zscore': stop_zscore, 'max_holding_period': max_holding_period,
            'execution': 'next close; lag applied by backtester',
        })
        return signals

    def get_summary(self):
        if self.spread is None:
            self.calculate_spread()
        if self.z_score is None:
            self.calculate_zscore()
        return {
            'Ticker1': self.ticker1, 'Ticker2': self.ticker2, 'Hedge_Ratio': self.hedge_ratio,
            'Spread_Mean': self.spread.mean(), 'Spread_Std': self.spread.std(),
            'ZScore_Mean': self.z_score.mean(), 'ZScore_Std': self.z_score.std(),
            'Spread_Min': self.spread.min(), 'Spread_Max': self.spread.max(),
        }


def generate_all_signals(price_data, pair_results, entry_threshold=2.0,
                         exit_threshold=0.0, lookback=60, stop_zscore=None,
                         max_holding_period=None):
    """Generate each pair's desired positions without using subsequent prices."""
    signal_data = {}
    for result in pair_results:
        ticker1, ticker2 = result['ticker1'], result['ticker2']
        if ticker1 not in price_data or ticker2 not in price_data:
            logger.warning('Missing prices for %s-%s', ticker1, ticker2)
            continue
        try:
            generator = PairSignalGenerator(price_data, ticker1, ticker2,
                                            hedge_ratio=result.get('hedge_ratio'), lookback=lookback)
            signal_data[f'{ticker1}-{ticker2}'] = generator.get_signals(
                entry_threshold, exit_threshold, stop_zscore, max_holding_period)
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning('Error generating signals for %s-%s: %s', ticker1, ticker2, exc)
    logger.info('Generated signals for %d pairs', len(signal_data))
    return signal_data
