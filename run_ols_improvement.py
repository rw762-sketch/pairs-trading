"""Compare five fixed OLS strategy changes, selecting only on earlier folds.

The later period was already inspected in prior work. Freezing a rule before
this new simulation does not make that period an untouched final holdout.
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
LABELS = {
    'baseline': 'Monthly OLS baseline',
    'quality': 'Stable beta + recovery speed',
    'entry_filter': 'Entry confirmation + cost hurdle',
    'risk_limits': 'Stops + allocation limits',
    'combined': 'All three changes',
}
PERIODS = {
    'validation_1': ('2024-09-19', '2025-03-31'),
    'validation_2': ('2025-04-01', '2025-10-22'),
    'later': ('2025-10-23', '2026-09-17'),
}
CAPITAL = 100000.0
SOURCES = {
    'OLS': 'https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html',
    'cointegration': 'https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html',
    'chronological_selection': 'https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html',
    'timing_costs_and_deadlines': 'https://arxiv.org/abs/1707.03498',
}


def money(value):
    return f'{"-" if value < 0 else ""}${abs(value):,.2f}'


def table(rows, headers):
    clean = lambda v: str(v).replace('|', '/').replace('\n', ' ')
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    lines += ['| ' + ' | '.join(clean(row[h]) for h in headers) + ' |' for row in rows]
    return '\n'.join(lines)


def select_validation_rule(validation):
    """Rank ONLY the two earlier folds; never accept later-period metrics.

    Eligibility is a practical research gate, not a significance test. Require
    positive net profit in both folds, >=3 trades each and >=10 overall. Rank by
    the worse fold's return, then mean fold return, then stable variant name.
    """
    if set(validation) != {'validation_1', 'validation_2'}:
        raise ValueError('Selection accepts exactly the two earlier folds.')
    if set(validation['validation_1']) != set(validation['validation_2']):
        raise ValueError('Both folds must compare the same strategies.')
    rows = []
    for name in sorted(validation['validation_1']):
        metrics = [validation[fold][name]['metrics'] for fold in ('validation_1', 'validation_2')]
        returns = [float(m['Total_Return']) for m in metrics]
        trades = [int(m['Num_Trades']) for m in metrics]
        if not np.isfinite(returns).all() or min(trades) < 0:
            raise ValueError('Validation metrics must be finite with nonnegative trade counts.')
        eligible = min(returns) > 0 and min(trades) >= 3 and sum(trades) >= 10
        rows.append({'Variant': name, 'Fold_1_Return': returns[0], 'Fold_2_Return': returns[1],
                     'Fold_1_Trades': trades[0], 'Fold_2_Trades': trades[1],
                     'Worst_Fold_Return': min(returns), 'Mean_Fold_Return': np.mean(returns),
                     'Eligible': bool(eligible)})
    ranked = sorted((r for r in rows if r['Eligible']),
                    key=lambda r: (-r['Worst_Fold_Return'], -r['Mean_Fold_Return'], r['Variant']))
    return {'selected_variant': ranked[0]['Variant'] if ranked else None,
            'status': 'eligible research candidate' if ranked else 'no strategy met the earlier-fold gate; cash fallback',
            'selection_rows': rows,
            'rule': 'positive net return in each fold; >=3 trades each; >=10 total; maximize worst-fold return, then mean return, then variant name'}


def run(output_dir):
    from ols_improvement import RULES, VARIANTS, build_schedule, run_variant
    prices_path = ROOT / 'output/pair-research-50/adjusted-prices.pkl'
    universe_path = ROOT / 'output/pair-research-50/universe.json'
    prices = pd.read_pickle(prices_path)
    metadata = json.loads(universe_path.read_text())
    started = datetime.now(REPORT_TIMEZONE)
    directory = create_run_directory(output_dir, 'ols-strategy-improvement', len(metadata), started_at=started)
    path = lambda kind, ext='csv': report_path(directory, kind, ext)
    link = lambda kind, label, ext='csv': f'[{label}]({path(kind, ext).name})'
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if set(VARIANTS) != set(LABELS):
        raise ValueError('The engine must match the five prescribed strategy variants.')
    plan = {'recorded_before_simulation': started.isoformat(), 'periods': PERIODS,
            'variants': VARIANTS, 'rules': RULES, 'labels': LABELS, 'initial_capital_per_fold': CAPITAL,
            'selection_rule': 'both fold returns >0; >=3 trades per fold and >=10 total; highest worst-fold return, mean return, then lexical variant tie-break',
            'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [prices_path, universe_path, ROOT / 'ols_improvement.py', Path(__file__)]},
            'sources': SOURCES, 'fresh_holdout': False,
            'limitations': 'Current constituents and prior sample availability filters introduce selection bias; later period already inspected and informed the research ideas.'}
    path('experiment-plan', 'json').write_text(json.dumps(plan, indent=2, default=str) + '\n')
    print(f'Prespecified plan: {path("experiment-plan", "json")}', flush=True)
    results, metric_rows, screening_frames = {}, [], []

    def evaluate(period):
        start, end = PERIODS[period]
        # Truncate the supplied data at this fold's end as an additional boundary.
        data = prices.loc[:end]
        schedule = build_schedule(data, metadata, start, end)
        panels = []
        for window in schedule:
            panel = window['panel'].copy()
            panel['Period'] = period
            panel['Formation_Start'] = window['formation_start']
            panel['Formation_End'] = window['formation_end']
            panel['Trading_Start'] = window['dates'][0]
            panel['Trading_End'] = window['dates'][-1]
            panels.append(panel)
        screening_frames.extend(panels)
        results[period] = {}
        for name in LABELS:
            result = run_variant(data, schedule, name, initial_capital=CAPITAL)
            results[period][name] = result
            curve, trades, pair_profits = result['equity_curve'], result['trades'], result['pair_profits']
            expected = data.index[(data.index >= start) & (data.index <= end)]
            pd.testing.assert_index_equal(curve.index, expected)
            assert np.isclose(trades.Net_PnL.sum(), curve.iloc[-1] - CAPITAL, atol=1e-7)
            assert np.isclose(pair_profits.Net_PnL.sum(), trades.Net_PnL.sum(), atol=1e-7)
            trades.to_csv(path(f'{period}-{name}-trades'), index=False)
            pair_profits.to_csv(path(f'{period}-{name}-pair-profits'), index=False)
            result['monthly'].to_csv(path(f'{period}-{name}-monthly'), index=False)
            result['selections'].to_csv(path(f'{period}-{name}-selections'), index=False)
            pd.DataFrame({'Account_Value': curve, 'Cumulative_Return': curve / CAPITAL - 1}).to_csv(
                path(f'{period}-{name}-account'), index_label='Date')
            metric_rows.append({'Period': period, 'Variant': name, 'Name': LABELS[name], **result['metrics']})
        print(f'Completed {period}: {start} to {end}', flush=True)

    evaluate('validation_1')
    evaluate('validation_2')
    decision = select_validation_rule(results)
    decision['frozen_before_later_simulation'] = datetime.now(REPORT_TIMEZONE).isoformat()
    path('frozen-selection', 'json').write_text(json.dumps(decision, indent=2) + '\n')
    print(f'Frozen earlier-fold decision: {decision["selected_variant"] or "cash fallback"}', flush=True)
    # The later-period results cannot enter select_validation_rule or change its decision.
    evaluate('later')
    pd.DataFrame(metric_rows).to_csv(path('all-performance'), index=False)
    pd.DataFrame(decision['selection_rows']).to_csv(path('validation-selection'), index=False)
    pd.concat(screening_frames, ignore_index=True).to_csv(path('all-screening'), index=False)
    pair_frames = [r['pair_profits'].assign(Variant=name, Period=period)
                   for period, variants in results.items() for name, r in variants.items()]
    pd.concat(pair_frames, ignore_index=True).to_csv(path('all-pair-profits'), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for color, (name, label) in zip(['#64748b', '#26857b', '#286aab', '#a5692a', '#8052a2'], LABELS.items()):
        curve = results['later'][name]['equity_curve']
        axes[0].plot(curve.index, curve / CAPITAL - 1, label=label, linewidth=1.8, color=color)
        axes[1].plot(curve.index, curve / curve.cummax().clip(lower=CAPITAL) - 1, color=color, linewidth=1.8)
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.axhline(0, color='#94a3b8', linewidth=.7)
        ax.grid(alpha=.2)
        ax.tick_params(axis='x', labelsize=8, rotation=25)
    axes[0].set_title('Later return after costs', loc='left')
    axes[1].set_title('Later drawdown', loc='left')
    axes[0].legend(fontsize=7)
    fig.suptitle('Keep OLS: five prespecified strategy variants | 50 stocks', fontweight='bold')
    fig.tight_layout()
    fig.savefig(path('later-performance', 'png'), dpi=160, bbox_inches='tight')
    plt.close(fig)

    comparison = []
    for name, label in LABELS.items():
        later = results['later'][name]['metrics']
        comparison.append({'Strategy': label,
                           'Earlier fold 1': f'{results["validation_1"][name]["metrics"]["Total_Return"]:.2%}',
                           'Earlier fold 2': f'{results["validation_2"][name]["metrics"]["Total_Return"]:.2%}',
                           'Later return': f'{later["Total_Return"]:.2%}',
                           'Later net profit': money(later['Final_Portfolio_Value'] - CAPITAL),
                           'Later trades': later['Num_Trades'],
                           'Later drawdown': f'{later["Max_Drawdown"]:.2%}'})
    selected = decision['selected_variant']
    if selected:
        outcome = (f'**Earlier-fold candidate: {LABELS[selected]}.** It met the predefined activity and positive-return gate '
                   f'in both earlier folds. Its later return was {results["later"][selected]["metrics"]["Total_Return"]:.2%}. '
                   'This gate is not a statistical proof of profitability.')
    else:
        outcome = ('**No strategy met the earlier-fold gate.** The frozen decision is to stay in cash, with a simulated '
                   '0% return and no interest. Any positive later-period result below remains a research observation; '
                   'it does not override the earlier-fold decision.')
    sections = [
        '# Improving the strategy while keeping linear regression',
        f'Generated {started:%Y-%m-%d %H:%M %Z}. A fixed 50-stock research pilot, not a full-universe result.',
        outcome,
        '## The regression stays',
        'Each month, fit **price A = alpha + beta x price B + residual** by ordinary least squares on the preceding '
        '252 closes. Beta determines hedge shares. Trade the deviation of the spread from its preceding 60-close '
        'average. Prices, selection diagnostics and normalization use no subsequent observations. Beta and shares '
        'remain fixed within each trade; this experiment uses no neural network or tree model.',
        '## Five rules specified before the run',
        table([
            {'Strategy': LABELS['baseline'], 'Change from baseline': 'None: monthly raw cointegration selection, entry +/-2, exit +/-0.5, holding cap 20 trading intervals.'},
            {'Strategy': LABELS['quality'], 'Change from baseline': 'Require residual AR(1) half-life 2-20 closes; both half-window betas positive and within 50% of full-window beta.'},
            {'Strategy': LABELS['entry_filter'], 'Change from baseline': 'Enter only while the z-score starts moving toward zero, with potential convergence distance at least twice the estimated round-trip cost.'},
            {'Strategy': LABELS['risk_limits'], 'Change from baseline': 'Adverse z-score stop 3.5; holding cap 10; disallow overlapping tickers; allocate at most 20% of monthly capital per pair and retain unused cash.'},
            {'Strategy': LABELS['combined'], 'Change from baseline': 'Apply all three changes together.'},
        ], ['Strategy', 'Change from baseline']),
        'All variants require same-sub-industry, different-issuer pairs with provisional I(1) compatibility, positive beta and raw '
        'Engle-Granger p < 0.05. These are exploratory raw-p screens, not multiple-testing-adjusted discoveries. '
        'The existing Holm experiment remains available and selected no trades; these filters do not replace a formal correction.',
        '## Dates and selection procedure',
        '- Cached history: 2023-09-19 to 2026-09-17. Formation uses the preceding 252 closes each month.\n'
        '- Earlier fold 1: 2024-09-19 to 2025-03-31. Earlier fold 2: 2025-04-01 to 2025-10-22.\n'
        '- Later comparison: 2025-10-23 to 2026-09-17.\n'
        '- Each fold and strategy starts with $100,000; gains/losses carry between months within that fold. Positions '
        'close at monthly and fold boundaries. The folds are separate accounts, not one continuous return series.\n'
        '- Eligible candidates need positive net return in BOTH earlier folds, at least three trades in each and ten '
        'total. Rank by the worse fold return, then mean return, then name. Save this decision before simulating the later period.',
        '**The later period was already inspected in prior work and informed these research ideas.** Freezing the decision '
        'now prevents this runner from choosing by later profits, but cannot restore untouched holdout status. '
        'The two earlier folds are development/validation data and the gate is a heuristic, not a significance test.',
        '## Measured results',
        table(comparison, ['Strategy', 'Earlier fold 1', 'Earlier fold 2', 'Later return', 'Later net profit', 'Later trades', 'Later drawdown']),
        f'![Later performance]({path("later-performance", "png").name})',
        'Smaller losses can result from smaller allocations and more idle cash. They do not by themselves show a stronger '
        'trading edge. All five later outcomes are disclosed; no later winner is automatically promoted.',
        '## Exact entry, cost and risk assumptions',
        'The entry filter requires current and previous z-scores on the same side of zero, absolute z decreasing, '
        'and current |z| > 2. The gross convergence proxy is (|z| - 0.5) x prior spread standard deviation / '
        '(price A + beta x price B). Compare it with twice [2 x 0.001 + 0.02 x 28/365.25 x short-leg fraction]. '
        'This distance is not an expected return or a probability-weighted forecast. It uses a 28-calendar-day cost allowance.',
        'Every signal executes at the following close. Actual fees/slippage are 0.1% of traded dollars for both legs '
        'at entry and exit. Short borrowing is 2% annualized over actual calendar days. The stop also executes at '
        'the following close and does not guarantee a maximum dollar loss. Limits apply to entry allocation; market '
        'moves can change exposure. Cash earns zero interest.',
        'Risk variants choose non-overlapping pairs greedily by formation p-value, then pair name. Each receives '
        'min(1 / selected pair count, 20%) of monthly starting account capital. Other variants divide that capital equally. '
        'Unused allocations remain cash. Fractional shares and short availability are assumed.',
        '## Pair profits and full records',
        link('all-pair-profits', 'Every selected pair, every method and period') + ' | ' +
        link('all-performance', 'All performance statistics') + ' | ' + link('all-screening', 'All monthly diagnostics'),
    ]
    for name, label in LABELS.items():
        pair_profit = results['later'][name]['pair_profits']
        pair_rows = [{'Pair': r.Pair, 'Net profit': money(r.Net_PnL), 'Trades': r.Num_Trades,
                      'Costs': money(r.Costs)} for r in pair_profit.sort_values('Pair').itertuples()]
        sections += ['### ' + label,
                     table(pair_rows, ['Pair', 'Net profit', 'Trades', 'Costs']) if pair_rows else 'No selected pairs.',
                     link(f'later-{name}-trades', 'Complete later trade log') + ' | ' +
                     link(f'later-{name}-monthly', 'Monthly allocation and account results') + ' | ' +
                     link(f'later-{name}-selections', 'Selected pairs, hedge ratios and allocations')]
    sections += [
        '## Sources, limitations and next step',
        f'[OLS documentation]({SOURCES["OLS"]}) explains the retained regression model. '
        f'[Engle-Granger documentation]({SOURCES["cointegration"]}) states the I(1) assumption and no-cointegration null. '
        f'[Chronological validation]({SOURCES["chronological_selection"]}) motivates earlier training and later checking. '
        f'[Kitapbayev and Leung]({SOURCES["timing_costs_and_deadlines"]}) study mean-reversion timing with costs and deadlines. '
        'Our exact five rules are research hypotheses; they are not an implementation of that paper\'s optimal stopping solution.',
        'Current index membership, industry labels and previous sample-availability filters introduce historical selection bias. '
        'Adjusted closes are proxies for executable prices. Liquidity, margin calls, recalls, market impact and taxes '
        'are not modeled. Only two relatively short earlier folds and one already-examined later period are available. '
        'A positive result is not a profit guarantee. No parameter was changed after viewing these new results.',
        'Next research step: obtain longer history and point-in-time membership; repeat a frozen rule across several '
        'chronological windows, and reserve genuinely new data or forward paper trading for final checking. '
        'Do not keep changing thresholds until this same later period turns positive.',
        link('experiment-plan', 'Plan recorded before simulation', 'json') + ' | ' +
        link('frozen-selection', 'Decision saved before later simulation', 'json'),
    ]
    path('report', 'md').write_text('\n\n'.join(sections) + '\n')
    update_report_index()
    print(table(comparison, ['Strategy', 'Earlier fold 1', 'Earlier fold 2', 'Later return', 'Later net profit', 'Later trades', 'Later drawdown']), flush=True)
    print(f'Report: {path("report", "md")}', flush=True)
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    run(parser.parse_args().output)
