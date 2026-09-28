"""Run a fixed, source-informed 50-stock rolling-selection experiment.

This comparison is separate from the established 52-pair baseline and does not
tune rules after seeing later-period returns. Output names retain the run date.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from reporting import REPORT_TIMEZONE, create_run_directory, report_path, update_report_index

ROOT = Path(__file__).resolve().parent
LABELS = {'fixed_selection': 'Fixed pairs + hedge ratios', 'monthly_raw': 'Monthly reselection + refit',
          'monthly_holm': 'Monthly refresh + Holm'}
SETTINGS = {'evaluation_start': '2025-10-23', 'formation_days': 252,
            'initial_capital': 100000.0, 'entry': 2.0, 'exit': .5, 'lookback': 60,
            'fee': .001, 'borrow': .02, 'max_hold': 20}
SOURCES = {
    'formation_and_trading': 'https://depot.som.yale.edu/icf/papers/fileuploads/2573/original/08-03.pdf',
    'dynamic_pairs_example': 'https://www.quantconnect.com/research/15347/intraday-dynamic-pairs-trading-using-correlation-and-cointegration-approach/',
    'rolling_regression': 'https://www.statsmodels.org/stable/examples/notebooks/generated/rolling_ls.html',
    'multiple_testing': 'https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html',
    'chronological_validation': 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html',
}


def markdown_table(rows, headers):
    def cell(v):
        return str(v).replace('|', '/').replace('\n', ' ')
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    lines.extend('| ' + ' | '.join(cell(row[h]) for h in headers) + ' |' for row in rows)
    return '\n'.join(lines)


def run_report(price_cache, universe_path, output_dir):
    from walk_forward import run_experiment
    prices = pd.read_pickle(price_cache)
    metadata = json.loads(Path(universe_path).read_text())
    tickers = [row['Yahoo ticker'] for row in metadata]
    prices = prices.loc[:, tickers]
    started = datetime.now(REPORT_TIMEZONE)
    directory = create_run_directory(output_dir, 'online-learning-walk-forward', len(tickers), started_at=started)
    def dest(kind, extension='csv'):
        return report_path(directory, kind, extension)
    def link(kind, label, extension='csv'):
        return f'[{label}]({dest(kind, extension).name})'
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    plan = {'specified_before_simulation': started.isoformat(), 'settings': SETTINGS,
            'variants': LABELS, 'price_cache': str(Path(price_cache).resolve()),
            'price_cache_sha256': sha(price_cache), 'universe_path': str(Path(universe_path).resolve()),
            'universe_sha256': sha(universe_path), 'tickers': tickers, 'sources': SOURCES,
            'evaluation_end': str(prices.index[-1].date()),
            'common_monthly_liquidation': True, 'status': 'retrospective comparison; period previously inspected'}
    dest('experiment-plan', 'json').write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Plan saved before simulation: {dest("experiment-plan", "json")}', flush=True)
    result = run_experiment(prices, metadata, **SETTINGS)
    result['screening'].to_csv(dest('all-monthly-screening'), index=False)
    variants = result['variants']
    if set(variants) != set(LABELS):
        raise ValueError('Unexpected experiment variants.')
    summary_rows, display_rows, pair_frames, month_frames = [], [], [], []
    expected_dates = prices.index[prices.index >= pd.Timestamp(SETTINGS['evaluation_start'])]
    for name, v in variants.items():
        curve, trades, metrics = v['equity_curve'], v['trades'], v['metrics']
        pd.testing.assert_index_equal(curve.index, expected_dates)
        if not np.isclose(trades.Net_PnL.sum(), curve.iloc[-1] - SETTINGS['initial_capital'], atol=1e-8):
            raise ValueError(f'{name} trades fail portfolio reconciliation.')
        monthly = v['monthly']
        pd.testing.assert_series_equal(monthly.Starting_Capital.iloc[1:].reset_index(drop=True),
                                       monthly.Ending_Capital.iloc[:-1].reset_index(drop=True), check_names=False)
        pair_profits = v['pair_profits']
        if not np.isclose(pair_profits.Net_PnL.sum(), trades.Net_PnL.sum(), atol=1e-8):
            raise ValueError('Per-pair profit does not reconcile.')
        active = int(metrics['Num_Trades']) > 0
        summary = {'Variant': name, 'Name': LABELS[name], **metrics,
                   'Pairs_Selected_Ever': len(pair_profits), 'Pairs_Traded': trades.Pair.nunique(),
                   'Months_Without_Eligible_Pairs': int((monthly.Selected_Pairs == 0).sum())}
        summary_rows.append(summary)
        display_rows.append({'Method': LABELS[name], 'Net return': f'{metrics["Total_Return"]:.2%}',
                             'Net profit': f'${curve.iloc[-1] - SETTINGS["initial_capital"]:,.2f}',
                             'Sharpe': f'{metrics["Sharpe_Ratio"]:.2f}' if active else 'n/a: no trades',
                             'Trades': int(metrics['Num_Trades']), 'Pairs traded': trades.Pair.nunique()})
        pd.DataFrame({'Account_Value': curve, 'Cumulative_Return': curve / SETTINGS['initial_capital'] - 1}).to_csv(
            dest(f'{name}-daily-account'), index_label='Date')
        trades.to_csv(dest(f'{name}-trade-log'), index=False)
        monthly.to_csv(dest(f'{name}-monthly-results'), index=False)
        pair_profits.to_csv(dest(f'{name}-pair-profits'), index=False)
        pair_frames.append(pair_profits.assign(Variant=name))
        month_frames.append(monthly.assign(Variant=name))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(dest('method-comparison'), index=False)
    all_pairs = pd.concat(pair_frames, ignore_index=True)
    all_pairs.to_csv(dest('all-pair-profits'), index=False)
    all_months = pd.concat(month_frames, ignore_index=True)
    all_months.to_csv(dest('all-monthly-results'), index=False)
    figure, ax = plt.subplots(figsize=(11, 5))
    colors = ['#64748b', '#24688f', '#32956e']
    for color, (name, v) in zip(colors, variants.items()):
        ax.plot(v['equity_curve'].index, v['equity_curve'] / SETTINGS['initial_capital'] - 1,
                label=LABELS[name], color=color, linewidth=1.9)
    ax.axhline(0, color='#94a3b8', linewidth=.7)
    ax.set_title(f'Rolling selection experiment | {len(tickers)} stocks', loc='left', fontweight='bold')
    ax.set_ylabel('Cumulative return on total account capital, after costs')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(dest('cumulative-returns', 'png'), dpi=160, bbox_inches='tight')
    plt.close(figure)
    counts = []
    first_months = variants['monthly_raw']['monthly']
    for i, row in first_months.iterrows():
        counts.append({'Trading window': f'{pd.Timestamp(row.Trading_Start):%Y-%m-%d} to {pd.Timestamp(row.Trading_End):%Y-%m-%d}',
                       'Formation ends': f'{pd.Timestamp(row.Formation_End):%Y-%m-%d}',
                       **{LABELS[name]: int(v['monthly'].iloc[i].Selected_Pairs) for name,v in variants.items()}})
    fixed_return = variants['fixed_selection']['metrics']['Total_Return']
    relative = {name: variants[name]['metrics']['Total_Return'] - fixed_return
                for name in ['monthly_raw', 'monthly_holm']}
    no_trade_names = [LABELS[name] for name,v in variants.items() if v['metrics']['Num_Trades'] == 0]
    sections = [
        '# Learning from online examples: rolling pair selection',
        f'Generated {started:%Y-%m-%d %H:%M %Z}. Evaluation: {expected_dates[0]:%Y-%m-%d} to {expected_dates[-1]:%Y-%m-%d}. '
        f'Universe: {len(tickers)} stocks; {result["config"]["family_size"]} predefined same-industry pair hypotheses per monthly screen.',
        '**Purpose:** measure whether refreshing pair selection improves this particular historical experiment. '
        'The three methods and settings were recorded before simulation. No threshold was changed to improve these results. '
        'This period was already inspected during project development, so it is not a fresh final holdout.',
        '## What was learned and implemented',
        markdown_table([
            {'Source': f'[Gatev et al.]({SOURCES["formation_and_trading"]})', 'Lesson': 'Separate formation from subsequent trading.', 'Applied here': 'Use preceding 252 daily closes; evaluate subsequent monthly windows. Their original strategy uses distance selection and different trading windows.'},
            {'Source': f'[QuantConnect dynamic example]({SOURCES["dynamic_pairs_example"]})', 'Lesson': 'Refresh eligibility and models, with an explicit open-position policy.', 'Applied here': 'Monthly cointegration screens and beta estimates; shares and beta fixed within trades; close at each monthly boundary.'},
            {'Source': f'[Statsmodels multiple tests]({SOURCES["multiple_testing"]})', 'Lesson': 'Account for testing many pairs.', 'Applied here': 'Compare a raw 5% threshold with a Holm-adjusted 5% threshold over each prescribed monthly family.'},
        ], ['Source', 'Lesson', 'Applied here']),
        '## Comparison after transaction and borrowing costs',
        markdown_table(display_rows, ['Method', 'Net return', 'Net profit', 'Sharpe', 'Trades', 'Pairs traded']),
        f'Monthly selection differs from the matched fixed-selection control by **{relative["monthly_raw"]*100:+.2f} percentage points** in total return. '
        f'The Holm variant differs by **{relative["monthly_holm"]*100:+.2f} percentage points**. '
        'These are sample outcomes, not evidence that an optimized strategy has been found. '
        'The monthly version changes both eligibility and beta estimates, so their individual effects are not isolated.',
        ('**Cash-only result:** ' + ', '.join(no_trade_names) + ' made no trades. Avoiding trading can avoid losses, but it does not establish a profitable trading signal.' if no_trade_names else
         'Trade counts and the monthly selection table show how much each method was active. A quieter strategy can change risk as well as return.'),
        f'![Net cumulative returns]({dest("cumulative-returns", "png").name})',
        '## Shared rules and fair-comparison boundaries',
        '- Select only same-sub-industry pairs from different issuers, with positive beta and provisional I(1) compatibility. Test price levels using Engle-Granger; regress with an intercept.',
        '- Entry z-score +/-2, exit band +/-0.5; normalization uses the preceding 60 closes. Signals execute at the following close. Maximum holding is 20 trading intervals.',
        '- Initial account $100,000 for each method. At each monthly boundary, close positions and divide that method\'s actual remaining capital equally across its eligible pairs. Capital carries forward, including losses and costs.',
        '- Fees/slippage: 10 basis points per traded dollar on entry and exit of both legs; annual short borrowing: 2%. No eligible pairs means idle cash with no interest.',
        '- All methods share monthly liquidation and resizing. The fixed-selection comparator keeps its first formation pairs and betas, while the other two refit each month. The first and last trading windows are partial.',
        '**Different study from the main report:** this is the pre-existing 50-stock pilot and a matched monthly-liquidation control. '
        'The main report uses 494 stocks, selects 52 pairs from a longer formation window, and does not force every monthly close. '
        'Do not attribute differences versus that main report solely to monthly selection.',
        '## How pair eligibility changed',
        markdown_table(counts, ['Trading window', 'Formation ends', *LABELS.values()]),
        'Every formation window ends before its trading window begins. The fixed method keeps its initial formation dates; '
        'the displayed formation-end column describes the two monthly methods. No later retest decides an earlier trade.',
        'The complete screen retains observed raw p-values and adjusted p-values. Invalid or I(1)-incompatible series are '
        'explicitly ineligible, with a conservative correction input of one rather than an invented observed p-value. '
        'Holm corrections apply to each monthly family, not automatically to the entire repeated research process.',
        '## Profit for every selected pair and full records',
        link('all-pair-profits', 'All methods: per-pair net profits') + ' | ' + link('all-monthly-results', 'All monthly results') + ' | ' + link('all-monthly-screening', 'Complete screening history'),
    ]
    for name, v in variants.items():
        rows = [{'Pair': r.Pair, 'Net profit': f'${r.Net_PnL:,.2f}', 'Costs': f'${r.Costs:,.2f}',
                 'Trades': int(r.Num_Trades)} for r in v['pair_profits'].sort_values('Pair').itertuples()]
        sections.extend(['### ' + LABELS[name],
                         markdown_table(rows, ['Pair', 'Net profit', 'Costs', 'Trades']) if rows else 'No pairs selected; the account stayed in cash.',
                         link(f'{name}-trade-log', 'Complete trade log') + ' | ' + link(f'{name}-monthly-results', 'Monthly capital and selection')])
    sections.extend([
        '## Limits and next learning steps',
        'The fixed sample was drawn with seed 42 from current industry metadata. Current constituents and industry labels '
        'introduce historical membership/selection limitations. Prices are adjusted closes; fractional shares and borrow '
        'availability are assumed. Liquidity, impact, recalls, margin and taxes are not modeled.',
        'The test period has been examined before. A monthly refresh and a multiple-testing correction can improve '
        'research discipline without improving returns. Do not keep changing the rules until this period becomes profitable. '
        'Use chronological development folds for further settings, and reserve genuinely new observations for final validation.',
        'Still to investigate: stability across several earlier windows, validation of hedge lookback/stop-loss/holding limits, '
        'and limits on repeated stock exposure. This pilot does not complete every parameter-optimization item from the assignment.',
        'Experiment inputs and sources: ' + link('experiment-plan', 'Recorded experiment plan', 'json') + '. '
        'Configuration, numerical checks and results: ' + link('run-metadata', 'Run metadata', 'json') + '.',
    ])
    dest('report', 'md').write_text('\n\n'.join(sections) + '\n')
    output_meta = {'generated_at': started.isoformat(), 'config': result['config'], 'source_plan': plan,
                   'summary': summary_rows, 'relative_returns': relative,
                   'validation': ['complete common date indexes', 'net trades reconcile to account profit',
                                  'capital carries forward across months', 'per-pair profit sums reconcile'],
                   'fresh_holdout': False, 'no_parameter_optimization': True}
    dest('run-metadata', 'json').write_text(json.dumps(output_meta, indent=2, default=str) + '\n')
    update_report_index()
    print(summary[['Name','Total_Return','Sharpe_Ratio','Num_Trades','Pairs_Traded']].to_string(index=False), flush=True)
    print(f'Report: {dest("report", "md")}', flush=True)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices', type=Path, default=ROOT / 'output/pair-research-50/adjusted-prices.pkl')
    parser.add_argument('--universe', type=Path, default=ROOT / 'output/pair-research-50/universe.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    run_report(args.prices, args.universe, args.output)


if __name__ == '__main__':
    main()
