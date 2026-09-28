"""Rebuild the complete, dated student report from the saved screen and prices.

No new price download or full-universe cointegration sweep is required. Pair
selection uses training statistics only; later-period sensitivity is descriptive.
Run with the project's venv/bin/python, not a class Python installation.
"""

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from backtester import PairBacktester, combine_portfolio_results
from cointegration_review import build_cointegration_review, integration_diagnostics
from reporting import create_run_directory, report_path, REPORT_TIMEZONE
from signal_generation import PairSignalGenerator

ROOT = Path(__file__).resolve().parent
CAPITAL = 100_000.0
BASE = {'entry': 2.0, 'exit': .5, 'fee': .001, 'borrow': .02, 'max_hold': 20}
def prepare_signals(prices, pairs, evaluation_index, entry, exit):
    result = {}
    for row in pairs.itertuples():
        gen = PairSignalGenerator(prices, row.Ticker1, row.Ticker2,
                                  hedge_ratio=row.Trading_Beta, lookback=60)
        gen.calculate_zscore()
        # Compute rolling statistics with preceding training history, then reset
        # the state machine flat at the beginning of the evaluation window.
        for attr in ('price_data', 'spread', 'z_score', 'moving_mean', 'moving_std'):
            setattr(gen, attr, getattr(gen, attr).loc[evaluation_index])
        result[row.Pair] = gen.get_signals(entry_threshold=entry, exit_threshold=exit)
    return result


def simulate(signals, *, fee=.001, borrow=.02, max_hold=20):
    results = {}
    for pair, frame in signals.items():
        engine = PairBacktester(frame, CAPITAL, fee, max_hold, borrow)
        equity = engine.run_backtest()
        results[pair] = {'equity_curve': equity, 'trades': engine.get_trades_dataframe(),
                         'metrics': engine.get_performance_metrics(), 'initial_capital': CAPITAL}
    portfolio = combine_portfolio_results(results, initial_capital=CAPITAL)
    if portfolio is None:
        raise ValueError('No candidate pairs available for the specified strategy.')
    # Independent accounting identities: terminal liquidation must reconcile.
    trades = portfolio['trades']
    assert np.isclose(trades.Net_PnL.sum(), portfolio['equity_curve'].iloc[-1] - CAPITAL)
    assert np.allclose(trades.Gross_PnL - trades.Costs, trades.Net_PnL)
    assert (trades.Exit_Date > trades.Entry_Date).all()
    if max_hold is not None:
        assert (trades.Bars_Held <= max_hold).all()
    for pair, result in results.items():
        assert result['equity_curve'].iloc[-1]['Gross_Exposure'] == 0
        frame = signals[pair]
        for trade in result['trades'].itertuples():
            entry_index = frame.index.get_loc(trade.Entry_Date)
            assert entry_index > 0 and trade.Signal == frame.Signal.iloc[entry_index - 1]
    return portfolio, results


def save_fig(fig, directory, kind):
    path = report_path(directory, kind, 'png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def build_report(screen_run, price_cache, universe_file, output_dir, make_pdf=True):
    started = datetime.now(REPORT_TIMEZONE)
    screen_run = Path(screen_run)
    metadata_path = report_path(screen_run, 'run-metadata', 'json')
    source_metadata = json.loads(metadata_path.read_text())
    source_pairs = report_path(screen_run, 'cointegrated-pairs', 'csv')
    raw = pd.read_csv(source_pairs)
    universe = json.loads(Path(universe_file).read_text())
    cached = pd.read_pickle(price_cache)
    prices = cached.loc[:, source_metadata['stocks']].copy()
    n_train = source_metadata['training_data']['rows']
    train, later = prices.iloc[:n_train], prices.iloc[n_train:]
    assert len(prices) == source_metadata['all_data']['rows']
    assert str(train.index[-1].date()) == source_metadata['training_data']['last_date']
    assert str(later.index[0].date()) == source_metadata['test_data']['first_date']
    assert str(later.index[-1].date()) == source_metadata['test_data']['last_date']
    assert np.isfinite(prices.to_numpy()).all() and (prices > 0).all().all()
    n = prices.shape[1]
    total_tests = n * (n - 1) // 2
    directory = create_run_directory(output_dir, 'research-calculations', n, started_at=started)
    print(f'Output: {directory}', flush=True)
    print('Checking integration assumptions on training prices...', flush=True)
    review = build_cointegration_review(raw, universe, total_tests)
    integration = integration_diagnostics(train)
    integration.to_csv(report_path(directory, 'stock-integration-diagnostics', 'csv'), index=False)
    info = {s['Yahoo ticker']: s for s in universe}
    stocks = pd.DataFrame({'Ticker': prices.columns})
    stocks = stocks.merge(review['ranked_stocks'], on='Ticker', how='left').merge(integration, on='Ticker')
    stocks['Partner_Count'] = stocks['Partner_Count'].fillna(0).astype(int)
    stocks['Company'] = stocks.Ticker.map(lambda t: info[t]['Security'])
    stocks = stocks.sort_values(['Partner_Count', 'Ticker'], ascending=[False, True])
    pairs = review['lowest_p_pairs'].copy()
    compatible = integration.set_index('Ticker').I1_Compatible
    pairs['Both_I1_Compatible'] = pairs.Ticker1.map(compatible) & pairs.Ticker2.map(compatible)
    pairs['Different_Issuers'] = [str(info[a]['CIK']) != str(info[b]['CIK'])
                                 for a, b in zip(pairs.Ticker1, pairs.Ticker2)]
    pairs['Strategy_Eligible'] = (pairs.Both_I1_Compatible & pairs.Same_Industry &
                                 pairs.Positive_Beta & pairs.Different_Issuers)
    selected = pairs.loc[pairs.Strategy_Eligible].copy()
    print(f'{len(pairs)} raw candidates -> {len(selected)} strategy candidates', flush=True)
    details = []
    for row in selected.itertuples():
        a, b = row.Ticker1, row.Ticker2
        x = np.column_stack([np.ones(len(train)), train[b]])
        alpha, beta = np.linalg.lstsq(x, train[a], rcond=None)[0]
        later_spread = later[a] - beta * later[b]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='adfuller currently returns', category=FutureWarning)
            fixed_beta_adf_p = adfuller(later_spread, autolag='AIC')[1]
        details.append({
            'Pair': f'{a}-{b}', 'Trading_Alpha': alpha, 'Trading_Beta': beta,
            'Recomputed_Train_P': coint(train[a], train[b])[1],
            'Later_Coint_P': coint(later[a], later[b])[1],
            'Later_Fixed_Beta_ADF_P': fixed_beta_adf_p,
            'Training_Return_Correlation': np.log(train[a]).diff().corr(np.log(train[b]).diff()),
            'Later_Return_Correlation': np.log(later[a]).diff().corr(np.log(later[b]).diff()),
        })
    selected = pd.concat([selected.reset_index(drop=True), pd.DataFrame(details)], axis=1)
    # Verify that the separately cached price download still supports selection.
    if not ((selected.Recomputed_Train_P < .05) & (selected.Trading_Beta > 0)).all():
        raise ValueError('Cached price data changes candidate eligibility; re-run and review the original screen.')
    counts = pd.concat([selected.Ticker1, selected.Ticker2]).value_counts()
    stocks['Strategy_Partner_Count'] = stocks.Ticker.map(counts).fillna(0).astype(int)
    for frame, kind in [(stocks, 'all-stock-rankings'), (pairs, 'all-candidate-p-values'),
                        (selected, 'selected-pairs-statistics')]:
        frame.to_csv(report_path(directory, kind, 'csv'), index=False)

    signal_cache = {}
    periods = {'In sample': train.index, 'Later period': later.index}
    for entry in [1.5, 2.0, 2.5]:
        for exit in [0.0, .25, .5]:
            for label, dates in periods.items():
                history = train if label == 'In sample' else prices
                signal_cache[label, entry, exit] = prepare_signals(history, selected, dates, entry, exit)
    print('Running both periods, threshold sensitivity and cost scenarios...', flush=True)
    portfolios, pair_results, sensitivity_rows = {}, {}, []
    for (label, entry, exit), signals in signal_cache.items():
        portfolio, individual = simulate(signals)
        sensitivity_rows.append({'Period': label, 'Entry': entry, 'Exit': exit, **portfolio['metrics']})
        if entry == BASE['entry'] and exit == BASE['exit']:
            portfolios[label], pair_results[label] = portfolio, individual
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(report_path(directory, 'threshold-sensitivity', 'csv'), index=False)
    performance = pd.DataFrame([{'Period': label, **p['metrics']} for label, p in portfolios.items()])
    performance.to_csv(report_path(directory, 'performance-comparison', 'csv'), index=False)
    later_signals = signal_cache['Later period', BASE['entry'], BASE['exit']]
    cost_rows, cost_portfolios = [], {}
    for name, fee, borrow, hold in [
        ('No costs, 20-day cap', 0.0, 0.0, 20),
        ('5 bps + 2% borrow, 20-day cap', .0005, .02, 20),
        ('10 bps + 2% borrow, 20-day cap', .001, .02, 20),
        ('20 bps + 2% borrow, 20-day cap', .002, .02, 20),
        ('No costs, no holding cap', 0.0, 0.0, None),
    ]:
        p = portfolios['Later period'] if fee == .001 else simulate(later_signals, fee=fee, borrow=borrow, max_hold=hold)[0]
        cost_portfolios[name] = p
        cost_rows.append({'Scenario': name, 'Fee_Per_Dollar': fee, 'Annual_Borrow_Rate': borrow,
                          'Max_Holding_Bars': hold, **p['metrics']})
    costs = pd.DataFrame(cost_rows)
    costs.to_csv(report_path(directory, 'cost-and-constraint-sensitivity', 'csv'), index=False)
    all_metrics = []
    for period, results in pair_results.items():
        for pair, result in results.items():
            all_metrics.append({'Period': period, 'Pair': pair, 'Portfolio_Weight': 1 / len(selected),
                                'Portfolio_Net_PnL': result['trades'].Net_PnL.sum() / len(selected),
                                **result['metrics']})
    pd.DataFrame(all_metrics).to_csv(report_path(directory, 'pair-performance', 'csv'), index=False)
    for label, portfolio in portfolios.items():
        slug = 'training' if label == 'In sample' else 'later'
        portfolio['trades'].to_csv(report_path(directory, f'{slug}-trade-log', 'csv'), index=False)
        curve = portfolio['equity_curve']
        daily = pd.DataFrame({'Account_Value': curve, 'Cumulative_Return': curve / CAPITAL - 1})
        for field in ['Gross_Exposure', 'Net_Exposure', 'Transaction_Costs', 'Borrow_Costs']:
            daily[field] = sum(result['equity_curve'][field] / len(selected)
                               for result in pair_results[label].values())
        daily.to_csv(report_path(directory, f'{slug}-daily-portfolio', 'csv'), index_label='Date')
    trades = portfolios['Later period']['trades']
    positive = trades.loc[trades.Net_PnL > 0].sort_values('Net_PnL', ascending=False)
    top5_share = positive.head(5).Net_PnL.sum() / positive.Net_PnL.sum() if len(positive) else 0
    without_top5 = (trades.Net_PnL.sum() - positive.head(5).Net_PnL.sum()) / CAPITAL
    contributions = trades.groupby('Pair', as_index=False).agg(Net_PnL=('Net_PnL', 'sum'),
                                                              Closed_Trades=('Pair', 'size'))
    contributions = contributions.sort_values('Net_PnL', ascending=False)
    contributions.to_csv(report_path(directory, 'pair-profit-contributions', 'csv'), index=False)

    print('Building readable figures and report...', flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    values = sensitivity.Total_Return.to_numpy()
    limit = max(abs(values.min()), abs(values.max()), .001)
    for ax, period in zip(axes, periods):
        grid = sensitivity.loc[sensitivity.Period == period].pivot(index='Entry', columns='Exit', values='Total_Return')
        ax.imshow(grid, cmap='RdYlGn', vmin=-limit, vmax=limit, aspect='auto')
        ax.set_xticks(range(3), [str(v) for v in grid.columns])
        ax.set_yticks(range(3), [str(v) for v in grid.index])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{grid.iloc[i, j]:.2%}', ha='center', va='center', fontsize=12)
        ax.grid(False)
        ax.set_xlabel('Exit z-score band')
        ax.set_ylabel('Entry z-score threshold')
        ax.set_title(period)
    fig.suptitle('Threshold sensitivity | net total return, fixed pair selection', fontweight='bold')
    fig.tight_layout()
    save_fig(fig, directory, 'threshold-sensitivity')
    for row in selected.head(2).itertuples():
        later_signals[row.Pair].to_csv(
            report_path(directory, f'{row.Pair}-later-signals', 'csv'), index_label='Date')
    fig, ax = plt.subplots(figsize=(11, 5))
    shown = pd.concat([contributions.head(5), contributions.tail(5)]).drop_duplicates('Pair').sort_values('Net_PnL')
    ax.barh(shown.Pair, shown.Net_PnL, color=['#ae4345' if v < 0 else '#2d7d6d' for v in shown.Net_PnL])
    ax.xaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
    ax.set_title('Pair contributions | largest five gains and losses after costs', loc='left', fontweight='bold')
    ax.set_xlabel('Contribution to the one $100,000 portfolio')
    fig.tight_layout()
    save_fig(fig, directory, 'pair-profit-contributions')

    later_m = portfolios['Later period']['metrics']
    survived = int((selected.Later_Coint_P < .05).sum())
    fixed_survived = int((selected.Later_Fixed_Beta_ADF_P < .05).sum())
    days = f'{prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}'
    train_days = f'{train.index[0]:%Y-%m-%d} to {train.index[-1]:%Y-%m-%d}'
    later_days = f'{later.index[0]:%Y-%m-%d} to {later.index[-1]:%Y-%m-%d}'
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    metadata = {
        'generated_at': started.isoformat(), 'source_screen_run': str(screen_run.resolve()),
        'source_pairs_sha256': sha(source_pairs), 'price_cache': str(Path(price_cache).resolve()),
        'price_cache_sha256': sha(price_cache), 'universe_metadata': str(Path(universe_file).resolve()),
        'universe_metadata_sha256': sha(universe_file),
        'stocks': n, 'dates': days, 'train_dates': train_days, 'later_dates': later_days,
        'train_rows': len(train), 'later_rows': len(later), 'initial_capital_per_period': CAPITAL,
        'rules': BASE | {'lookback': 60, 'selection': 'training-only same-industry, positive-beta, different issuers, provisional I(1), raw p < .05'},
        'raw_pair_count': len(pairs), 'selected_pair_count': len(selected), 'screen_diagnostics': review['diagnostics'],
        'later_retest_p_below_05': survived, 'later_fixed_beta_adf_p_below_05': fixed_survived,
        'top5_winners_share_of_positive_profit': top5_share, 'return_without_top5_contributions': without_top5,
        'later_metrics': later_m,
        'validation': ['trade net P&L reconciles to portfolio equity', 'gross P&L minus costs equals net P&L',
                       'entries follow previous-close signals', '20-bar holding limit', 'all positions liquidated'],
    }
    report_path(directory, 'run-metadata', 'json').write_text(json.dumps(metadata, indent=2) + '\n')
    from clean_report import build_clean_report
    clean_directory = build_clean_report(directory, output_dir, make_pdf=make_pdf)
    print(json.dumps({'report': str(report_path(clean_directory, 'report', 'md')),
                      'calculation_data': str(directory), 'selected_pairs': len(selected),
                      'later_metrics': later_m}, indent=2), flush=True)
    return clean_directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen-run', type=Path,
                        default=ROOT / 'results/2026-09-18_17-10-39_pairs-trading-backtest_494-stocks')
    parser.add_argument('--price-cache', type=Path, default=ROOT / 'output/pair-research/adjusted-prices.pkl')
    parser.add_argument('--universe', type=Path, default=ROOT / 'output/stock-universe/sp500-source.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--no-pdf', action='store_true')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    build_report(args.screen_run, args.price_cache, args.universe, args.output, not args.no_pdf)


if __name__ == '__main__':
    main()
