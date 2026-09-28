"""Run the fixed multiple-regression experiment on the saved 50-stock pilot."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from multiple_regression import FEATURES, RULES, RegressionGate
from ols_improvement import build_schedule, run_variant
from reporting import REPORT_TIMEZONE, create_run_directory, report_path
from run_ols_improvement import PERIODS, select_validation_rule, table

ROOT = Path(__file__).resolve().parent


def run(output_dir=ROOT / 'results'):
    started = datetime.now(REPORT_TIMEZONE)
    directory = create_run_directory(output_dir, 'multiple-regression', 50, started_at=started)
    path = lambda kind, ext='csv': report_path(directory, kind, ext)
    price_file = ROOT / 'output/pair-research-50/adjusted-prices.pkl'
    metadata_file = ROOT / 'output/pair-research-50/universe.json'
    prices = pd.read_pickle(price_file)
    metadata = json.loads(metadata_file.read_text())
    files = [price_file, metadata_file, Path(__file__), ROOT / 'multiple_regression.py',
             ROOT / 'ols_improvement.py', ROOT / 'backtester.py', ROOT / 'signal_generation.py',
             ROOT / 'walk_forward.py', ROOT / 'run_ols_improvement.py']
    plan = {'recorded_before_simulation': started.isoformat(), 'features': FEATURES, 'rules': RULES,
            'periods': PERIODS, 'fresh_holdout': False,
            'target': '(spread[t+6]-spread[t+1])/(P1[t]+beta*P2[t])',
            'fit': 'Intercept plus five standardized predictors, OLS, preceding 252 closes; final six feature rows excluded; >=120 complete rows; full rank required.',
            'selection': 'Existing earlier-fold rule: positive return and >=3 trades in each fold, >=10 total; rank worst-fold then mean return; otherwise cash.',
            'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    path('experiment-plan', 'json').write_text(json.dumps(plan, indent=2) + '\n')
    results, rows = {}, []
    for period, (start, end) in PERIODS.items():
        if period == 'later':
            decision = select_validation_rule(results)
            decision['recorded_before_later_simulation'] = datetime.now(REPORT_TIMEZONE).isoformat()
            path('frozen-selection', 'json').write_text(json.dumps(decision, indent=2) + '\n')
        data = prices.loc[:end]
        schedule = build_schedule(data, metadata, start, end)
        results[period] = {}
        for variant in ['baseline', 'multiple_regression']:
            gate = RegressionGate() if variant == 'multiple_regression' else None
            result = run_variant(data, schedule, 'baseline', signal_transform=gate)
            results[period][variant] = result
            expected = data.loc[start:end].index
            pd.testing.assert_index_equal(result['equity_curve'].index, expected)
            assert np.isclose(result['trades'].Net_PnL.sum(), result['equity_curve'].iloc[-1] - 100000, atol=1e-7)
            assert np.isclose(result['pair_profits'].Net_PnL.sum(), result['trades'].Net_PnL.sum(), atol=1e-7)
            for key in ['trades', 'monthly', 'pair_profits', 'selections']:
                frame = result[key].copy()
                frame['Variant'] = variant
                frame.to_csv(path(f'{period}-{variant}-{key}'), index=False)
            result['equity_curve'].to_csv(path(f'{period}-{variant}-account'), index_label='Date')
            if gate is not None:
                path(f'{period}-models', 'json').write_text(json.dumps(gate.models, indent=2) + '\n')
                audit = pd.concat(gate.diagnostics, ignore_index=True) if gate.diagnostics else pd.DataFrame()
                audit.to_csv(path(f'{period}-predictions'), index=False)
                # Every actual fill must follow a passing gate on the preceding close.
                for trade in result['trades'].itertuples():
                    previous = data.index[data.index.get_loc(trade.Entry_Date) - 1]
                    matched = audit.loc[(audit.Pair == trade.Pair) & (audit.Date == previous)]
                    assert len(matched) == 1 and bool(matched.Entry_Gate_Passed.iloc[0])
            rows.append({'Period': period, 'Variant': variant, **result['metrics']})
        print(f'Completed {period}', flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(path('performance'), index=False)
    display = [{'Period': r['Period'], 'Model': r['Variant'], 'Net return': f"{r['Total_Return']:.2%}",
                'Trades': int(r['Num_Trades']), 'Costs': f"${r['Total_Costs']:,.2f}",
                'Max drawdown': f"{r['Max_Drawdown']:.2%}"} for r in rows]
    report = f'''# Multiple-regression experiment

Recorded {started:%Y-%m-%d %H:%M %Z}. Saved 50-stock pilot. Each period starts with $100,000.

{table(display, ['Period', 'Model', 'Net return', 'Trades', 'Costs', 'Max drawdown'])}

Earlier-fold decision: **{decision['selected_variant'] or 'cash fallback; neither model qualified'}**.

## Model

Ordinary least squares with an intercept and five predictors: spread deviation from its preceding 60-close mean, one-day spread momentum, five-day spread momentum, 20-day volatility of daily spread movement, and the peer stock's five-day return. Spread deviation and momentum are divided by current gross notional. Daily spread movements for volatility use preceding-close gross notional. Each feature is standardized using training means and standard deviations only.

The target is (s[t+6] - s[t+1]) / (P1[t] + beta*P2[t]), where s = P1 - alpha - beta*P2. This excludes the move before next-close entry. The existing pair hedge remains a two-stock hedge; multiple regression predicts the spread move, rather than adding tradable stocks to the hedge.

Fit each selected pair on the preceding 252 closes; discard missing warm-up rows and the last six feature rows, whose targets are not yet complete. Require at least 120 rows, full rank, and standardized design condition number <= 1e6. Freeze coefficients for the trading month. Monthly model JSON files include coefficients, scaling, rank, training fit and dates. Training R-squared is descriptive, not a performance claim. Overlapping five-day labels are dependent; no independent-observation significance claim is made.

Entry requires the existing z-score beyond ±2 and a signed predicted move greater than twice estimated round-trip costs: 2 × [0.002 + 0.02 × 7/365.25 × short-leg fraction]. This is a fixed hypothesis, not a confidence interval. Invalid fits leave their allocations in cash. The five-day horizon is an entry filter, not a promised exit date.

Selection, allocation, ordinary ±0.5 z-score exits, next-close execution, 20-interval holding cap, monthly liquidation, 0.10% fees on traded dollars and actual-calendar-day 2% annual short borrowing use the existing engine. Cash earns zero interest.

## Evaluation and limits

{table([{'Period': key, 'Start': dates[0], 'End': dates[1]} for key, dates in PERIODS.items()], ['Period', 'Start', 'End'])}

The fixed feature set and thresholds were recorded before simulation. The existing selection rule uses only the two earlier folds and was saved before the later-period run. **All these dates have already been used in project research; the later period is not a fresh holdout.** Current constituents, prior history filtering, raw multiple-pair screening, adjusted-close execution and simplified borrowing remain limitations. Fewer entries also mean more idle cash, so reduced losses alone do not establish better predictions.

All monthly and total allocated trade profits reconcile to account changes. Every actual regression entry was checked against the preceding close's forecast gate. Metrics, trade logs, account curves, monthly allocations, pair profits, predictions, coefficients, and input/source hashes are saved alongside this report.

Method reference: [Statsmodels OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html). The feature choices and cost hurdle are project hypotheses.

Reproduce: `./venv/bin/python run_multiple_regression.py`
'''
    path('report', 'md').write_text(report)
    (directory / 'START_HERE.md').write_text(f'# Multiple regression\n\n[Results and method]({path("report", "md").name})\n')
    print(metrics[['Period', 'Variant', 'Total_Return', 'Num_Trades', 'Total_Costs', 'Max_Drawdown']].to_string(index=False))
    print(f'Report: {path("report", "md")}', flush=True)
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    run(parser.parse_args().output)
