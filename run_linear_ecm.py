"""Run one prespecified spread-ECM entry gate against the unchanged OLS baseline.

The existing periods have already been inspected. This is a transparent research
comparison, not a new untouched holdout and not an automatic trading deployment.
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

from linear_ecm import ECM_RULES, SOURCES, build_ecm_schedule, run_ecm_variant
from ols_improvement import RULES as BASE_RULES
from reporting import REPORT_TIMEZONE, create_run_directory, report_path
from run_ols_improvement import select_validation_rule

ROOT = Path(__file__).resolve().parent
PERIODS = {
    'validation_1': ('2024-09-19', '2025-03-31'),
    'validation_2': ('2025-04-01', '2025-10-22'),
    'later': ('2025-10-23', '2026-09-17'),
}
LABELS = {'baseline': 'Monthly OLS baseline', 'ecm_gate': 'OLS + spread ECM forecast gate'}
CAPITAL = 100000.


def table(rows, headings):
    def cell(value):
        return str(value).replace('|', '/').replace('\n', ' ')
    lines = ['| ' + ' | '.join(headings) + ' |', '| ' + ' | '.join(['---'] * len(headings)) + ' |']
    lines += ['| ' + ' | '.join(cell(row[key]) for key in headings) + ' |' for row in rows]
    return '\n'.join(lines)


def money(value):
    return f'{"-" if value < 0 else ""}${abs(value):,.2f}'


def run(output_dir=ROOT / 'results'):
    prices_path = ROOT / 'output/pair-research-50/adjusted-prices.pkl'
    metadata_path = ROOT / 'output/pair-research-50/universe.json'
    prices = pd.read_pickle(prices_path)
    metadata = json.loads(metadata_path.read_text())
    started = datetime.now(REPORT_TIMEZONE)
    directory = create_run_directory(output_dir, 'linear-ecm-forecast-gate', len(metadata), started_at=started)
    path = lambda kind, ext='csv': report_path(directory, kind, ext)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source_files = [prices_path, metadata_path, ROOT / 'linear_ecm.py', Path(__file__),
                    ROOT / 'ols_improvement.py', ROOT / 'walk_forward.py', ROOT / 'backtester.py',
                    ROOT / 'signal_generation.py', ROOT / 'run_ols_improvement.py']
    plan = {
        'recorded_before_any_simulation': started.isoformat(), 'periods': PERIODS, 'variants': LABELS,
        'ecm_rules': ECM_RULES, 'baseline_rules': BASE_RULES, 'initial_capital_per_period': CAPITAL,
        'selection_rule': 'Only earlier folds: positive net return in both, >=3 trades each and >=10 total; rank worst-fold return, then mean return, then name; otherwise cash fallback',
        'formula': 'delta_s[t] = a + k*s[t-1] + g*delta_s[t-1]; s = P1-alpha-beta*P2',
        'stability': 'Full OLS rank 3; k<0; companion eigenvalues of AR2 (1+k+g,-g) have modulus <1',
        'forecast_gate': 'direction*(forecast_s[t+6]-forecast_s[t+1])/(P1+beta*P2) > 2*(2*.001+.02*7/365.25*short_leg_fraction)',
        'isolated_change': 'Entry gate only; normal exits, allocations, costs, execution and monthly closures unchanged',
        'invalid_models': 'No new entries; their baseline pair allocations remain idle cash rather than being reassigned',
        'sources': SOURCES, 'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in source_files},
        'fresh_holdout': False,
        'limitations': 'Earlier folds and later period have already been inspected and informed research ideas; recording a new plan does not restore independent validation. Raw pair screening, current constituents and prior availability filters remain exploratory.',
    }
    path('experiment-plan', 'json').write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Plan recorded: {path("experiment-plan", "json")}', flush=True)
    results, metrics_rows, panels = {}, [], []

    def evaluate(period):
        start, end = PERIODS[period]
        data = prices.loc[:end]
        schedule = build_ecm_schedule(data, metadata, start, end)
        results[period] = {}
        for window in schedule:
            panels.append(window['panel'].assign(Period=period))
        for variant in LABELS:
            result = run_ecm_variant(data, schedule, variant, CAPITAL)
            expected = data.index[(data.index >= start) & (data.index <= end)]
            pd.testing.assert_index_equal(result['equity_curve'].index, expected)
            assert np.isclose(result['trades'].Net_PnL.sum(), result['equity_curve'].iloc[-1] - CAPITAL, atol=1e-7)
            assert np.isclose(result['pair_profits'].Net_PnL.sum(), result['trades'].Net_PnL.sum(), atol=1e-7)
            results[period][variant] = result
            for key, suffix in [('trades', 'trades'), ('monthly', 'monthly'), ('pair_profits', 'pair-profits'),
                                ('selections', 'selections')]:
                result[key].to_csv(path(f'{period}-{variant}-{suffix}'), index=False)
            if variant == 'ecm_gate':
                result['forecast_diagnostics'].to_csv(path(f'{period}-{variant}-forecast-diagnostics'), index=False)
            curve = result['equity_curve']
            pd.DataFrame({'Account_Value': curve, 'Cumulative_Return': curve / CAPITAL - 1}).to_csv(
                path(f'{period}-{variant}-account'), index_label='Date')
            metrics_rows.append({'Period': period, 'Variant': variant, 'Name': LABELS[variant], **result['metrics']})
        print(f'Completed {period}: {start} to {end}', flush=True)

    evaluate('validation_1')
    evaluate('validation_2')
    decision = select_validation_rule(results)
    decision['frozen_before_later_simulation'] = datetime.now(REPORT_TIMEZONE).isoformat()
    path('frozen-selection', 'json').write_text(json.dumps(decision, indent=2) + '\n')
    print(f'Frozen earlier-fold decision: {decision["selected_variant"] or "cash fallback"}', flush=True)
    evaluate('later')
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(path('all-performance'), index=False)
    pd.concat(panels, ignore_index=True).to_csv(path('all-model-diagnostics'), index=False)
    pd.DataFrame(decision['selection_rows']).to_csv(path('validation-selection'), index=False)
    all_profits = pd.concat([result['pair_profits'].assign(Period=period, Variant=variant)
                            for period, variants in results.items() for variant, result in variants.items()], ignore_index=True)
    all_profits.to_csv(path('all-pair-profits'), index=False)

    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for variant, color in [('baseline', '#385f7b'), ('ecm_gate', '#a5672a')]:
        curve = results['later'][variant]['equity_curve']
        axes[0].plot(curve.index, curve / CAPITAL - 1, color=color, linewidth=1.8, label=LABELS[variant])
        peaks = curve.cummax().clip(lower=CAPITAL)
        axes[1].plot(curve.index, curve / peaks - 1, color=color, linewidth=1.6)
    for axis in axes:
        axis.axhline(0, color='#94a3b8', linewidth=.7)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=.2)
    axes[0].set_title('Later-period cumulative return after costs', loc='left', fontweight='bold')
    axes[0].legend(loc='best')
    axes[1].set_title('Drawdown from account peak', loc='left', fontweight='bold')
    fig.suptitle('One additional linear model: an ECM forecast entry gate', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path('later-comparison', 'png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

    comparison = []
    for variant, name in LABELS.items():
        m = results['later'][variant]['metrics']
        comparison.append({'Strategy': name,
                           'Earlier fold 1': f'{results["validation_1"][variant]["metrics"]["Total_Return"]:.2%}',
                           'Earlier fold 2': f'{results["validation_2"][variant]["metrics"]["Total_Return"]:.2%}',
                           'Later return': f'{m["Total_Return"]:.2%}', 'Later profit': money(m['Final_Portfolio_Value'] - CAPITAL),
                           'Later trades': int(m['Num_Trades']), 'Later costs': money(m['Total_Costs'])})
    later_ecm = results['later']['ecm_gate']
    later_base = results['later']['baseline']
    selection = decision['selected_variant']
    outcome = (f'The earlier-fold research gate selected **{LABELS[selection]}**.' if selection else
               'Neither strategy met the earlier-fold research gate. The recorded decision is **cash fallback**, with 0% modeled return and no interest.')
    gate_d = later_ecm['forecast_diagnostics']
    entry_setup_days = int((gate_d.ZScore.abs() > 2).sum()) if len(gate_d) else 0
    pass_days = int(gate_d.Entry_Gate_Passed.sum()) if len(gate_d) else 0
    profit_rows = []
    for row in later_ecm['pair_profits'].sort_values('Pair').itertuples():
        profit_rows.append({'Pair': row.Pair, 'Net profit': money(row.Net_PnL), 'Trades': row.Num_Trades,
                            'Costs': money(row.Costs), 'Selected months': row.Selected_Windows})
    def link(kind, label, ext='csv'):
        return f'[{label}]({path(kind, ext).name})'
    report = [
        '# Linear ECM: an additional OLS strategy',
        f'Recorded {started:%Y-%m-%d %H:%M %Z}. Fixed 50-stock pilot; no thresholds were tuned after this run.',
        outcome,
        f'The added gate returned **{later_ecm["metrics"]["Total_Return"]:.2%}** in the later comparison, versus '
        f'**{later_base["metrics"]["Total_Return"]:.2%}** for the unchanged monthly OLS baseline. '
        'This comparison alone does not establish a dependable trading edge.',
        '## What changed',
        'The original OLS hedge remains: price A = alpha + beta × price B + residual. The additional linear regression '
        'models how that residual changes: **Δs(t) = a + k × s(t−1) + g × Δs(t−1)**. All coefficients are fitted '
        'using only the preceding 252 closes and then frozen during the month. This is one error-correction equation '
        'for a spread, not a fitted multivariate VECM and not a machine-learning classifier.',
        'A new entry needs the usual z-score beyond ±2 and a valid forecast model: three independent OLS columns, '
        'k < 0, and stable implied AR(2) dynamics. We forecast the residual at the next close (t+1) and six closes '
        'ahead (t+6). The signed difference between those two forecasts is the potential movement during five '
        'intervals after execution. The earlier t-to-t+1 movement is deliberately excluded because the order has '
        'not executed yet.',
        'The signed forecast movement divided by current gross notional must exceed **twice** '
        '[0.002 + 0.02 × 7/365.25 × short-leg fraction]. For a long spread, the short leg is beta × price B; '
        'for a short spread, it is price A. The seven-calendar-day borrowing allowance is a fixed entry heuristic, '
        'not the realized borrowing bill or a confidence interval. The forecast can be wrong.',
        '## What stayed unchanged',
        '- Same monthly peer selection: different issuers, same sub-industry, provisional I(1) diagnostics, positive OLS beta and raw cointegration p < 0.05.\n'
        '- Same preceding 60-close z-score, entry ±2 and exit toward ±0.5. Normal exits do not depend on the new entry gate.\n'
        '- Same next-close execution, 20-trading-interval cap, monthly liquidation, fixed shares within a trade and equal allocation across the baseline-selected pairs.\n'
        '- Invalid models cannot open new positions; their allocated capital remains idle. It is not reallocated to the surviving models.\n'
        '- Same 0.10% fees/slippage on actual traded dollars on entry and exit, 2% annual short borrow over actual calendar days, and zero cash interest.',
        'The five-interval forecast is a filter, not an exit deadline. Actual trades can end earlier from a signal or '
        'monthly boundary, or later up to the existing 20-interval limit.',
        '## Dates and evaluation',
        table([{'Period': name, 'Start': dates[0], 'End': dates[1]} for name, dates in PERIODS.items()], ['Period', 'Start', 'End']),
        'Each strategy and period begins with a separate $100,000 account; capital carries across months within a '
        'period. The decision rule, saved before the later simulation, requires positive net return in both earlier '
        'folds, at least three trades each and ten total. Eligible variants are ranked by the worse fold return, '
        'then the mean return, then name. This is a research activity gate, not a statistical significance test.',
        '**All these dates have already been inspected in prior project work.** Earlier folds are reused development '
        'data; the later window is not an untouched holdout. A newly recorded plan prevents outcome-based changes '
        'within this run but cannot restore independent validation.',
        '## Measured results',
        table(comparison, ['Strategy', 'Earlier fold 1', 'Earlier fold 2', 'Later return', 'Later profit', 'Later trades', 'Later costs']),
        f'![Later account returns and drawdowns]({path("later-comparison", "png").name})',
        'Fewer trades and more idle cash can reduce both gains and losses. Compare costs, activity and drawdown '
        'alongside return before describing the gate as an improvement.',
        '## Net profit by pair: later ECM strategy',
        'Dollar amounts use the actual changing monthly allocations from one account. A pair with no executed '
        'trades remains in this list if it was selected for a monthly allocation.',
        table(profit_rows, ['Pair', 'Net profit', 'Trades', 'Costs', 'Selected months']),
        f'Allocated profits sum to **{money(later_ecm["trades"].Net_PnL.sum())}** and reconcile to ending capital minus $100,000. '
        f'Among selected-pair daily observations, {entry_setup_days} had |z| > 2 and {pass_days} passed the full '
        'forecast gate. These are signal-day counts, not executed trade counts.',
        '## Audit files',
        '- ' + link('all-performance', 'All financial metrics') + '\n- ' + link('all-model-diagnostics', 'Every monthly screening and fitted ECM diagnostic') +
        '\n- ' + link('later-ecm_gate-forecast-diagnostics', 'Daily forecast inputs, t+1/t+6 projections, cost estimates and gate decisions') +
        '\n- ' + link('later-ecm_gate-trades', 'Actual allocated ECM trade log') + '\n- ' + link('later-ecm_gate-pair-profits', 'Every ECM pair contribution') +
        '\n- ' + link('experiment-plan', 'Protocol recorded before simulation', 'json') + '\n- ' + link('frozen-selection', 'Earlier-fold decision frozen before later simulation', 'json'),
        '## Sources and interpretation',
        f'[Statsmodels OLS]({SOURCES["ols"]}) documents ordinary least squares. '
        f'[Statsmodels VECM]({SOURCES["vecm_context"]}) provides the multivariate error-correction context; our '
        'single-spread equation is a deliberately smaller adaptation, not that full model. The choice of five '
        'forecast intervals and a two-times-cost hurdle is a prespecified project hypothesis, not a rule endorsed by these sources.',
        f'[AR roots documentation]({SOURCES["ar_root_convention"]}) uses roots of the AR lag polynomial, which must '
        'lie outside the unit circle. We check their reciprocal companion eigenvalues, which must lie inside. '
        'Here AR(2) coefficients are 1+k+g and −g. The exported diagnostics state that convention explicitly.',
        'Current index constituents and past complete-history filtering introduce selection bias. Raw multiple-pair '
        'testing remains exploratory. Adjusted closes, fractional shares, available stock borrow and no market-impact '
        'or margin-call constraints simplify execution. Stable fitted coefficients do not guarantee future convergence. '
        'No live trades were placed.',
    ]
    path('report', 'md').write_text('\n\n'.join(report) + '\n')
    (directory / 'START_HERE.md').write_text('# Linear ECM experiment\n\n' +
        link('report', 'Read the strategy and measured results', 'md') + '\n\n' +
        link('all-performance', 'Open all metrics') + '\n\n' + link('all-pair-profits', 'Open pair profits') + '\n')
    summary = {'directory': str(directory), 'report': str(path('report', 'md')),
               'decision': decision['selected_variant'], 'later_metrics': {k:v['metrics'] for k,v in results['later'].items()},
               'sources': SOURCES}
    path('run-summary', 'json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    run(parser.parse_args().output)
