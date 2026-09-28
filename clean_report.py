"""Create a concise assignment report from an existing, verified research run.

This changes presentation only. It never chooses new pairs or retunes trading
rules. The PDF contains required outputs; complete machine-readable records live
under data/ and chart assets under figures/.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np
import pandas as pd

from reporting import REPORT_TIMEZONE, create_run_directory, report_path, update_report_index
from visualization import plot_spread_and_zscore

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / 'results/2026-09-18_17-58-16_complete-research-report_494-stocks'


def table(rows, headers):
    def cell(value):
        return str(value).replace('|', '/').replace('\n', ' ')
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    lines.extend('| ' + ' | '.join(cell(row[h]) for h in headers) + ' |' for row in rows)
    return '\n'.join(lines)


def money(value, signed=False):
    sign = '-' if value < 0 else ('+' if signed and value > 0 else '')
    return f'{sign}${abs(value):,.2f}'


def clean_pair_table(selected, pair_metrics, training_trades, later_trades, capital):
    """Return one row per pair, with all dollar values on the portfolio basis."""
    if selected.Pair.duplicated().any():
        raise ValueError('Selected pair identifiers must be unique.')
    out = selected[['Pair', 'Ticker1', 'Company1', 'Ticker2', 'Company2', 'Industry1',
                    'Trading_Beta', 'Coint_PValue', 'Recomputed_Train_P', 'Later_Coint_P']].copy()
    out = out.rename(columns={
        'Industry1': 'Industry', 'Trading_Beta': 'Hedge_Ratio',
        'Coint_PValue': 'Training_Coint_PValue', 'Recomputed_Train_P': 'Cached_Training_Coint_PValue',
        'Later_Coint_P': 'Later_Coint_PValue',
    })
    for period, label, log in [('In sample', 'Training', training_trades),
                               ('Later period', 'Later', later_trades)]:
        metrics = pair_metrics.loc[pair_metrics.Period == period].set_index('Pair')
        if metrics.index.duplicated().any() or set(metrics.index) != set(out.Pair):
            raise ValueError(f'Missing or repeated {period} pair results.')
        by_pair = log.groupby('Pair')
        net = by_pair.Net_PnL.sum().reindex(out.Pair, fill_value=0)
        reported = metrics.loc[out.Pair, 'Portfolio_Net_PnL']
        if not np.allclose(net.to_numpy(), reported.to_numpy(), atol=1e-8):
            raise ValueError(f'{period} allocated profit does not match the trade log.')
        out[f'{label}_Net_Profit'] = net.to_numpy()
        out[f'{label}_Trades'] = by_pair.size().reindex(out.Pair, fill_value=0).to_numpy()
        out[f'{label}_Trading_Costs'] = (by_pair.Entry_Cost.sum() + by_pair.Exit_Cost.sum()).reindex(out.Pair, fill_value=0).to_numpy()
        out[f'{label}_Borrow_Costs'] = by_pair.Borrow_Cost.sum().reindex(out.Pair, fill_value=0).to_numpy()
        out[f'{label}_Win_Rate'] = out.Pair.map(metrics.Win_Rate)
        out[f'{label}_Return'] = out.Pair.map(metrics.Total_Return)
        allocations = out.Pair.map(metrics.Portfolio_Weight) * capital
        if 'Starting_Allocation' in out and not np.allclose(out.Starting_Allocation, allocations):
            raise ValueError('Pair allocations differ between periods.')
        out['Starting_Allocation'] = allocations
        if not np.allclose(out[f'{label}_Return'], out[f'{label}_Net_Profit'] / allocations):
            raise ValueError('Pair returns do not reconcile to allocated profit.')
    return out.sort_values(['Training_Coint_PValue', 'Pair']).reset_index(drop=True)


def build_clean_report(source_dir=DEFAULT_SOURCE, output_dir=ROOT / 'results', make_pdf=True):
    source = Path(source_dir).resolve()
    def src(kind, extension='csv'):
        return report_path(source, kind, extension)
    meta = json.loads(src('run-metadata', 'json').read_text())
    selected = pd.read_csv(src('selected-pairs-statistics'))
    metrics = pd.read_csv(src('performance-comparison')).set_index('Period')
    pair_metrics = pd.read_csv(src('pair-performance'))
    train_log = pd.read_csv(src('training-trade-log'))
    test_log = pd.read_csv(src('later-trade-log'))
    sensitivity = pd.read_csv(src('threshold-sensitivity'))
    costs = pd.read_csv(src('cost-and-constraint-sensitivity'))
    capital = float(meta['initial_capital_per_period'])
    pairs = clean_pair_table(selected, pair_metrics, train_log, test_log, capital)
    for period, label, log in [('In sample', 'Training', train_log), ('Later period', 'Later', test_log)]:
        expected = metrics.loc[period, 'Final_Portfolio_Value'] - capital
        if not np.isclose(pairs[f'{label}_Net_Profit'].sum(), expected, atol=1e-8):
            raise ValueError(f'{period} pair profits do not sum to the account change.')
        if int(pairs[f'{label}_Trades'].sum()) != len(log):
            raise ValueError('Trade counts do not reconcile.')
    started = datetime.now(REPORT_TIMEZONE)
    run = create_run_directory(output_dir, 'pairs-trading-clean-report', int(meta['stocks']), started_at=started)
    data_dir, figures_dir = run / 'data', run / 'figures'
    data_dir.mkdir()
    figures_dir.mkdir()
    def dest(kind, extension='csv', folder=data_dir):
        return folder / report_path(run, kind, extension).name
    def link(kind, label=None, extension='csv', folder=data_dir):
        path = dest(kind, extension, folder)
        return f'[{label or kind}]({path.relative_to(run).as_posix()})'
    def chart(kind, caption):
        path = dest(kind, 'png', figures_dir)
        return f'![{caption}]({path.relative_to(run).as_posix()})'
    def save_figure(fig, kind):
        fig.savefig(dest(kind, 'png', figures_dir), dpi=180, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    pairs.to_csv(dest('pair-profits-and-statistics'), index=False)
    for kind in ['performance-comparison', 'threshold-sensitivity', 'cost-and-constraint-sensitivity',
                 'training-trade-log', 'later-trade-log', 'training-daily-portfolio', 'later-daily-portfolio']:
        shutil.copy2(src(kind), dest(kind))
    # Keep the complete candidate fields for reproducibility, outside the PDF.
    selected.to_csv(dest('selected-pairs-details'), index=False)
    daily = pd.read_csv(src('later-daily-portfolio'), index_col='Date', parse_dates=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True)
    axes[0].plot(daily.index, daily.Account_Value, color='#245b83', linewidth=1.7)
    axes[0].axhline(capital, color='#64748b', linewidth=.8, linestyle='--')
    axes[0].yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
    axes[0].set_title('Backtest equity', loc='left', fontweight='bold')
    axes[0].set_ylabel('Account value')
    axes[1].plot(daily.index, daily.Cumulative_Return, color='#2d7d6d', linewidth=1.7)
    axes[1].axhline(0, color='#64748b', linewidth=.8, linestyle='--')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title('Cumulative portfolio return', loc='left', fontweight='bold')
    axes[1].set_ylabel('Net return')
    fig.tight_layout()
    save_figure(fig, 'portfolio-curves')
    for kind in ['threshold-sensitivity', 'pair-profit-contributions']:
        shutil.copy2(src(kind, 'png'), dest(kind, 'png', figures_dir))
    # Two examples selected by training p-value, independent of later profit.
    examples = selected.sort_values('Coint_PValue').head(2)
    for row in examples.itertuples():
        signals = pd.read_csv(src(f'{row.Pair}-later-signals'), index_col='Date', parse_dates=True)
        signals.attrs.update(entry_threshold=meta['rules']['entry'], exit_threshold=meta['rules']['exit'],
                             lookback=meta['rules']['lookback'])
        fig = plot_spread_and_zscore(signals, title=f'{row.Pair} | spread and z-score', figsize=(11, 4.8))
        save_figure(fig, f'{row.Pair}-spread-zscore')
        signals.to_csv(dest(f'{row.Pair}-signals'), index_label='Date')

    p = metrics.loc['Later period']
    train = metrics.loc['In sample']
    n_pairs = len(pairs)
    distinct_stocks = len(set(selected.Ticker1) | set(selected.Ticker2))
    win_pairs = int((pairs.Later_Net_Profit > 0).sum())
    lose_pairs = int((pairs.Later_Net_Profit < 0).sum())
    r = meta['rules']
    pages = []
    pages.append('\n\n'.join([
        '# Pairs trading: strategy and results',
        f'Report date: {started:%Y-%m-%d %H:%M %Z}. This is a clearer presentation of the existing simulation; no trading rules or results were changed.',
        f'**Later-period result: {p.Total_Return:.2%} | {money(p.Final_Portfolio_Value - capital, True)} net profit/loss | {money(p.Final_Portfolio_Value)} final capital.**',
        '## Data and objective',
        table([
            {'Period': 'Training / in sample', 'Dates': meta['train_dates'], 'Daily closes': meta['train_rows']},
            {'Period': 'Later / out of sample', 'Dates': meta['later_dates'], 'Daily closes': meta['later_rows']},
        ], ['Period', 'Dates', 'Daily closes']),
        f'{meta["stocks"]} complete stock histories produced {meta["raw_pair_count"]:,} raw candidate pairs; '
        f'{n_pairs} passed the additional strategy filters, involving {distinct_stocks} distinct stocks. '
        'The strategy trades a price spread and expects it to return toward its recent average. '
        'Cointegration is evidence of a stable linear combination under statistical assumptions, not a profit guarantee.',
        '## The tested strategy, step by step',
        table([
            {'Step': '1. Select pairs', 'Rule': 'Training cointegration p < 0.05; same sub-industry; different companies; positive beta; individual prices compatible with I(1).'},
            {'Step': '2. Fit the hedge', 'Rule': 'Regress A = alpha + beta x B on training adjusted closes. Freeze beta during testing. Spread = A - beta x B.'},
            {'Step': '3. Measure the gap', 'Rule': f'Z-score = (spread - preceding {r["lookback"]}-close mean) / preceding standard deviation. Today is excluded from those statistics.'},
            {'Step': '4. Enter', 'Rule': f'Below -{r["entry"]:g}: buy A, short beta shares of B per A share. Above +{r["entry"]:g}: short A, buy beta shares of B per A share.'},
            {'Step': '5. Exit', 'Rule': f'Close a long spread at z >= -{r["exit"]:g}; close a short at z <= +{r["exit"]:g}. Also close after {r["max_hold"]} trading intervals or at the period end. No additional stop-loss is used.'},
            {'Step': '6. Execute and size', 'Rule': f'Execute at the following close. Split one {money(capital)} account equally: {money(capital/n_pairs)} gross entry allocation per pair. Keep shares fixed until exit.'},
            {'Step': '7. Charge costs', 'Rule': f'{r["fee"]:.2%} combined fees/slippage per dollar traded, on both legs at entry and exit; {r["borrow"]:.0%} annual expense on short borrowing.'},
        ], ['Step', 'Rule']),
        '**Portfolio treatment:** all selected pairs are included; idle allocations stay in cash. Each period starts a separate account. After a timed exit, the pair waits for a new signal cycle. The pair list is fixed throughout the later period.',
    ]))

    metric_specs = [
        ('Total_Return', 'Total return', 'percent'), ('Annualized_Return', 'Annualized return (CAGR)', 'percent'),
        ('Volatility', 'Annualized volatility', 'percent'), ('Sharpe_Ratio', 'Sharpe ratio', 'number'),
        ('Max_Drawdown', 'Maximum drawdown', 'percent'), ('Win_Rate', 'Win rate, after costs', 'percent'),
        ('Avg_Trade_Return', 'Average net trade return', 'percent'), ('Turnover', 'Annual gross turnover', 'times'),
        ('Num_Trades', 'Completed pair trades', 'integer'), ('Total_Costs', 'Total fees + borrowing', 'money'),
        ('Final_Portfolio_Value', 'Final account value', 'money'),
    ]
    def format_metric(value, kind):
        return {'percent': lambda: f'{value:.2%}', 'number': lambda: f'{value:.2f}',
                'times': lambda: f'{value:.2f}x', 'integer': lambda: str(int(value)),
                'money': lambda: money(value)}[kind]()
    metric_rows = [{'Metric': label, 'Training': format_metric(train[key], kind),
                    'Later test': format_metric(p[key], kind)} for key, label, kind in metric_specs]
    pages.append('\n\n'.join([
        '## Portfolio performance',
        table(metric_rows, ['Metric', 'Training', 'Later test']),
        'Returns and trade profits include costs. Training performance uses pairs and regression fitted on that same period, so it is optimistic. The later period has already been examined during development; it is not an untouched final holdout.',
        chart('portfolio-curves', f'Later-period equity and cumulative return for all {n_pairs} selected pairs, after costs.'),
        'Sharpe uses daily returns and a zero risk-free rate. Volatility uses 252 trading days/year; CAGR uses elapsed calendar time. Turnover is both-leg entry/exit dollars divided by starting capital per year. Average trade return uses entry gross notional. Drawdown includes starting capital.',
    ]))

    # Full readable candidate and profit table replaces dense matrices and raw hubs.
    for start in range(0, n_pairs, 26):
        part = pairs.iloc[start:start+26]
        rows = [{'Pair': f'{row.Ticker1} / {row.Ticker2}', 'Train p': f'{row.Training_Coint_PValue:.5f}',
                 'Later p': f'{row.Later_Coint_PValue:.4f}', 'Train profit': money(row.Training_Net_Profit, True),
                 'Later profit': money(row.Later_Net_Profit, True), 'Trades*': int(row.Later_Trades)}
                for row in part.itertuples()]
        intro = (f'Every selected pair is listed, sorted by training p-value. Profit is **after fees and borrowing**, '
                 f'using {money(capital/n_pairs)} per pair from the same {money(capital)} account. '
                 'Negative amounts are losses. *Trades means later-period completed round trips.')
        notes = (f'**Totals across all {n_pairs} pairs:** training {money(pairs.Training_Net_Profit.sum(), True)}; '
                 f'later {money(pairs.Later_Net_Profit.sum(), True)}. Later: {win_pairs} profitable pairs and {lose_pairs} losing pairs. '
                 'Totals use unrounded amounts; displayed cents can differ slightly when added.')
        pages.append('\n\n'.join([
            f'## Candidate p-values and profit for every pair ({start+1}-{min(start+26,n_pairs)})',
            intro if start == 0 else 'Continued on the same allocation and after-cost basis. *Trade counts refer to the later period.',
            table(rows, ['Pair', 'Train p', 'Later p', 'Train profit', 'Later profit', 'Trades*']),
            ('Later p-values are diagnostic retests after the period; they did not select trades. These raw p-values are not adjusted for the original multiple-pair search.' if start == 0 else notes),
            link('pair-profits-and-statistics', 'Full pair table: company names, hedge ratios, returns, costs and profits'),
        ]))

    chart_blocks = ['## Selected spreads and trading thresholds',
                    'The two examples were selected by the smallest training p-values, not later profits. '
                    'The spread need not center on zero; its preceding rolling mean is the reference. '
                    'Markers show signal changes. Actual executions occur at the following close.']
    for row in examples.itertuples():
        chart_blocks.append(chart(f'{row.Pair}-spread-zscore',
                                  f'{row.Pair}: beta {row.Trading_Beta:.3f}; enter beyond +/-{r["entry"]:g}; exit toward the +/-{r["exit"]:g} band.'))
    pages.append('\n\n'.join(chart_blocks))

    later_grid = sensitivity.loc[sensitivity.Period == 'Later period']
    cost_names = ['No costs; 20-day cap', '5 bps + 2% borrow', '10 bps + 2% borrow (base)',
                  '20 bps + 2% borrow', 'No costs; no holding cap']
    cost_rows = [{'Scenario': cost_names[i] if len(costs) == 5 else row.Scenario,
                  'Net return': f'{row.Total_Return:.2%}', 'Sharpe': f'{row.Sharpe_Ratio:.2f}',
                  'Trading costs': money(row.Total_Transaction_Costs), 'Borrowing': money(row.Total_Borrow_Costs)}
                 for i, row in enumerate(costs.itertuples())]
    pages.append('\n\n'.join([
        '## Threshold and cost sensitivity',
        'The same pairs and training hedge ratios are used throughout. The baseline is entry 2 / exit 0.5. '
        'Each cell changes only entry/exit thresholds, with baseline costs and the 20-day cap. '
        'These comparisons describe the historical periods; the best later-period cell was not selected as a validated model.',
        chart('threshold-sensitivity', 'Nine entry/exit combinations in each period; cell values are net total return.'),
        f'All nine later-period variants lost money: {later_grid.Total_Return.min():.2%} to {later_grid.Total_Return.max():.2%}.',
        '### Costs and maximum holding period',
        table(cost_rows, ['Scenario', 'Net return', 'Sharpe', 'Trading costs', 'Borrowing']),
        'One basis point (bp) = 0.01% per traded dollar. Unless labeled otherwise, the holding cap is 20 trading intervals. '
        f'Even the no-cost version with that cap returned {costs.iloc[0].Total_Return:.2%}; costs alone do not explain the loss.',
        link('threshold-sensitivity', 'Full sensitivity metrics') + ' | ' + link('cost-and-constraint-sensitivity', 'Cost scenarios'),
    ]))

    preview = []
    for row in test_log.head(8).itertuples():
        preview.append({'Pair': row.Pair, 'Entry': row.Entry_Date[:10], 'Exit': row.Exit_Date[:10],
                        'Spread': 'Long' if row.Signal == 1 else 'Short', 'Net profit': money(row.Net_PnL, True),
                        'Bars': int(row.Bars_Held)})
    pages.append('\n\n'.join([
        '## Trade log and profit concentration',
        f'The later period contains {len(test_log)} completed pair trades; training contains {len(train_log)}. '
        'One row is a complete entry-and-exit round trip across both stocks. The first eight later trades appear below.',
        table(preview, ['Pair', 'Entry', 'Exit', 'Spread', 'Net profit', 'Bars']),
        '**Complete logs:** ' + link('later-trade-log', 'All later-period trades') + ' | ' + link('training-trade-log', 'All training trades') + '. '
        'The files include signed shares, both entry/exit prices, gross profit, fees, borrowing, net profit and exit reason.',
        chart('pair-profit-contributions', 'Five largest positive and negative pair contributions to the same portfolio.'),
        f'The five largest winning trades produced {meta["top5_winners_share_of_positive_profit"]:.1%} of all positive net trade profits. '
        f'Subtracting those five contributions leaves {meta["return_without_top5_contributions"]:.2%} total return. '
        'This is a contribution calculation, not a new strategy simulation.',
    ]))

    questions = [
        ('Does high correlation imply cointegration?', 'No. Correlation measures return co-movement; cointegration concerns a stationary combination of price levels.'),
        ('Do the relationships persist?', f'{meta["later_retest_p_below_05"]} of {n_pairs} later cointegration retests pass raw p < 0.05; {meta["later_fixed_beta_adf_p_below_05"]} frozen-beta spreads pass the separate ADF diagnostic. These are evidence, not certainty.'),
        ('Are thresholds important?', f'Later net returns range from {later_grid.Total_Return.min():.2%} to {later_grid.Total_Return.max():.2%}; none of the nine settings is profitable.'),
        ('Do costs destroy the profits?', f'The 20-day-cap version returns {costs.iloc[0].Total_Return:.2%} before costs and {p.Total_Return:.2%} with baseline costs.'),
        ('Are results concentrated?', f'{win_pairs} pairs make money and {lose_pairs} lose money. The top five winning trades supply {meta["top5_winners_share_of_positive_profit"]:.1%} of the sum of positive net trade profits.'),
        ('Does the strategy survive constraints?', 'The tested delayed-execution, cost and holding-cap version loses money. Live viability has not been demonstrated.'),
    ]
    brief_status = [
        {'PDF parameter requirement': 'Entry / exit thresholds', 'Status': 'Nine combinations tested in each period.'},
        {'PDF parameter requirement': 'Transaction costs / holding period', 'Status': 'Cost scenarios and 20-day versus no cap tested.'},
        {'PDF parameter requirement': 'Cointegration p-value cutoff', 'Status': 'Still to test; 0.05 used here.'},
        {'PDF parameter requirement': 'Hedge-ratio lookback / stop-loss', 'Status': 'Still to test; full training beta and no extra stop-loss used.'},
        {'PDF parameter requirement': 'Validation-based optimization', 'Status': 'Still to implement before claiming an optimized model.'},
    ]
    pages.append('\n\n'.join([
        '## Findings and remaining research plan',
        table([{'Research question': q, 'Answer from this run': a} for q,a in questions],
              ['Research question', 'Answer from this run']),
        '### Remaining parameter work from the original PDF',
        table(brief_status, ['PDF parameter requirement', 'Status']),
        '### Proposed next version (not used in these results)',
        'Retest pair stability and refit hedge ratios each month using only preceding prices; keep each open trade\'s hedge ratio fixed until exit. '
        'Choose settings on earlier validation periods, control multiple testing, and limit repeated exposure to the same stock. '
        'Freeze the rules before evaluation on a genuinely new period. Compare with this baseline using identical costs.',
        '**Statistical and execution limits:** raw pair p-values are exploratory; none passes the conservative correction for the original '
        f'{meta["screen_diagnostics"]["total_tests_in_original_screen"]:,}-pair search. Individual I(1) checks are provisional. '
        'Current index membership, full-period availability filtering, adjusted-close fills, fractional shares and assumed borrow availability simplify the experiment. '
        'Liquidity, market impact and margin calls are not modeled.',
        'Requirements source: *4. Pairs Trading Strategy.pdf*, pages 1-3. '
        'Statistical definitions: [Engle-Granger](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html) and '
        '[ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html). '
        'The required eleven output types are present in this PDF and its linked full data files; parameter optimization remains incomplete as shown above.',
    ]))
    markdown = report_path(run, 'report', 'md')
    markdown.write_text('\n\n<!-- pagebreak -->\n\n'.join(pages) + '\n', encoding='utf-8')
    readme = run / 'START_HERE.md'
    readme.write_text(
        '# Pairs trading results\n\n' +
        (f'- [Readable PDF]({report_path(run, "report", "pdf").name})\n' if make_pdf else '') +
        f'- [Readable Markdown]({markdown.name})\n'
        f'- {link("pair-profits-and-statistics", "Profit and statistics for all selected pairs")}\n'
        f'- {link("later-trade-log", "Complete later-period trade log")}\n'
        f'- {link("training-trade-log", "Complete training trade log")}\n\n'
        'The PDF includes the strategy, required statistical outputs, every pair\'s net profit, and remaining work. '
        'Detailed records live in `data/`; chart images live in `figures/`. '
        'Profits use actual allocations to one portfolio, not a separate $100,000 account for each pair.\n\n'
        'Regenerate this presentation from its saved source:\n\n'
        f'```bash\n./venv/bin/python clean_report.py --source "{source}"\n```\n', encoding='utf-8')
    source_files = [src(kind) for kind in ['pair-performance', 'selected-pairs-statistics',
                    'training-trade-log', 'later-trade-log', 'performance-comparison']]
    manifest = {
        'report_date': started.isoformat(), 'source_run': str(source), 'presentation_only': True,
        'source_data_dates': meta['dates'], 'selected_pairs': n_pairs, 'distinct_stocks': distinct_stocks,
        'starting_capital_per_period': capital, 'training_net_profit': float(pairs.Training_Net_Profit.sum()),
        'later_net_profit': float(pairs.Later_Net_Profit.sum()), 'later_profitable_pairs': win_pairs,
        'later_losing_pairs': lose_pairs, 'source_metadata': meta,
        'source_hashes': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        'validation': ['all pair profits match allocated logs', 'pair profit sums match account changes',
                       'pair returns use allocated capital', 'all selected pairs included'],
        'parameter_optimization_complete': False,
    }
    dest('provenance', 'json').write_text(json.dumps(manifest, indent=2) + '\n')
    if make_pdf:
        from research_report_pdf import render_report
        render_report(markdown, report_path(run, 'report', 'pdf'), profile='clean')
    update_report_index()
    print(json.dumps({'clean_report': str(run), 'pairs': n_pairs,
                      'net_profit_later': float(pairs.Later_Net_Profit.sum())}, indent=2))
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--no-pdf', action='store_true')
    args = parser.parse_args()
    build_clean_report(args.source, args.output, not args.no_pdf)


if __name__ == '__main__':
    main()
