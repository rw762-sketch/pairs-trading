"""Screen the project's stock universe; no trading or backtest P&L.

Run with: ./venv/bin/python screen_pairs.py
Ranks daily log-return correlations using training observations only.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
from reporting import REPORT_TIMEZONE, create_run_directory, report_path, update_report_index


def pair_statistics(a, b, train, check, training_returns, check_returns, metadata):
    first, second = metadata[a], metadata[b]
    history = train[[a, b]].dropna()
    later = check[[a, b]].dropna()
    returns = training_returns[[a, b]].dropna()
    later_returns = check_returns[[a, b]].dropna()
    fit = np.linalg.lstsq(
        np.column_stack([np.ones(len(history)), history[b]]), history[a], rcond=None
    )[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        p_train = float(coint(history[a], history[b], trend='c', autolag='aic')[1])
        p_check = (float(coint(later[a], later[b], trend='c', autolag='aic')[1])
                   if len(later) >= 120 else None)
    rolling = returns[a].rolling(60).corr(returns[b]).dropna()
    return {
        'ticker1': a, 'ticker2': b,
        'company1': first['Security'], 'company2': second['Security'],
        'industry1': first['GICS Sub-Industry'],
        'industry2': second['GICS Sub-Industry'],
        'same_issuer': first['CIK'] == second['CIK'],
        'same_industry': first['GICS Sub-Industry'] == second['GICS Sub-Industry'],
        'train_return_correlation': float(returns[a].corr(returns[b])),
        'train_last126_correlation': float(returns[a].tail(126).corr(returns[b].tail(126))),
        'train_min60_correlation': float(rolling.min()),
        'check_return_correlation': (float(later_returns[a].corr(later_returns[b]))
                                     if len(later_returns) >= 120 else None),
        'train_return_observations': len(returns),
        'check_return_observations': len(later_returns),
        'train_coint_pvalue': p_train,
        'check_coint_pvalue': p_check,
        'train_intercept': float(fit[0]), 'train_hedge_ratio': float(fit[1]),
        'warnings': list(dict.fromkeys(str(w.message) for w in caught)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2023-09-19')
    parser.add_argument('--end', default='2026-09-18', help='Exclusive end date')
    parser.add_argument('--universe', default='output/stock-universe/sp500-source.json')
    parser.add_argument('--output', default='output/pair-research', help='Base directory for caches and dated report folders')
    parser.add_argument('--name', default='correlation-screen', help='Descriptive name included in report filenames')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    started_at = datetime.now(REPORT_TIMEZONE)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = json.loads(Path(args.universe).read_text())
    metadata = {r['Yahoo ticker']: r for r in records}
    tickers = sorted(metadata)
    # Cache only this script's downloaded data, with source/window verification.
    cache = output / 'adjusted-prices.pkl'
    cache_info = output / 'price-cache-info.json'
    signature = {'tickers': tickers, 'start': args.start, 'end_exclusive': args.end}
    if (not args.refresh and cache.exists() and cache_info.exists()
            and json.loads(cache_info.read_text())['request'] == signature):
        prices = pd.read_pickle(cache)
        fetched_at = json.loads(cache_info.read_text())['fetched_at_utc']
    else:
        data = yf.download(tickers, start=args.start, end=args.end,
                           auto_adjust=False, progress=False, threads=8)
        if not isinstance(data.columns, pd.MultiIndex) or 'Adj Close' not in data.columns.levels[0]:
            raise ValueError('Expected adjusted close prices; refusing unadjusted fallback.')
        prices = data['Adj Close'].sort_index().reindex(columns=tickers)
        if prices.empty:
            raise ValueError('No downloaded prices.')
        prices.to_pickle(cache)
        fetched_at = datetime.now(timezone.utc).isoformat()
        cache_info.write_text(json.dumps({'request': signature, 'fetched_at_utc': fetched_at}, indent=2))
    boundary = int(len(prices) * .7)
    train, check = prices.iloc[:boundary], prices.iloc[boundary:]
    if len(train) < 252 or len(check) < 120:
        raise ValueError('Insufficient training or later-check observations.')
    # Only training history determines eligibility. Never forward-fill missing prices.
    eligible = train.columns[train.notna().all() & (train > 0).all()]
    train = train[eligible]
    check = check[eligible].where(check[eligible] > 0)
    training_returns = np.log(train / train.shift(1)).iloc[1:]
    check_returns = np.log(check / check.shift(1)).iloc[1:]
    correlation = training_returns.corr()
    upper = np.triu_indices(len(eligible), 1)
    ranking = pd.DataFrame({
        'a': eligible.to_numpy()[upper[0]],
        'b': eligible.to_numpy()[upper[1]],
        'correlation': correlation.to_numpy()[upper],
    }).dropna().sort_values(['correlation', 'a', 'b'], ascending=[False, True, True])
    strong = ranking[ranking['correlation'] >= .8].copy()
    independent = strong[
        strong.apply(lambda r: metadata[r.a]['CIK'] != metadata[r.b]['CIK'] and
                     metadata[r.a]['GICS Sub-Industry'] == metadata[r.b]['GICS Sub-Industry'], axis=1)
    ]
    keys = list(dict.fromkeys(
        list(ranking.head(20)[['a', 'b']].itertuples(index=False, name=None)) +
        list(strong[['a', 'b']].itertuples(index=False, name=None))
    ))
    print(f'{len(eligible)} training-eligible stocks; {len(ranking)} correlations; '
          f'{len(strong)} >= 0.80; {len(independent)} different-company same-industry pairs.', flush=True)
    results = {}
    for a, b in keys:
        results[(a, b)] = pair_statistics(a, b, train, check, training_returns, check_returns, metadata)
    top = [results[(r.a, r.b)] for r in ranking.head(20).itertuples()]
    all_strong = [results[(r.a, r.b)] for r in strong.itertuples()]
    peers = [results[(r.a, r.b)] for r in independent.itertuples()]
    # Exploratory research shortlist. Later-period metrics never affect selection/order.
    candidates = [r for r in peers if r['train_coint_pvalue'] < .05
                  and r['train_hedge_ratio'] > 0 and r['train_last126_correlation'] >= .75]
    run_dir = create_run_directory(output, args.name, len(eligible), started_at=started_at)
    results_file = report_path(run_dir, 'results', 'json')
    report_file = report_path(run_dir, 'report', 'md')
    universe_file = report_path(run_dir, 'universe', 'json')
    universe_file.write_text(json.dumps([metadata[t] for t in eligible], indent=2))
    summary = {
        'run_name': args.name, 'run_id': run_dir.name,
        'run_started_at': started_at.isoformat(), 'report_timezone': str(REPORT_TIMEZONE),
        'used_tickers': eligible.tolist(),
        'fetched_at_utc': fetched_at,
        'requested_start': args.start, 'requested_end_exclusive': args.end,
        'training_start': str(train.index[0].date()),
        'training_end': str(train.index[-1].date()),
        'check_start': str(check.index[0].date()), 'check_end': str(check.index[-1].date()),
        'requested_tickers': len(tickers), 'training_eligible_tickers': len(eligible),
        'excluded_training_tickers': sorted(set(tickers) - set(eligible)),
        'training_price_rows': len(train), 'check_price_rows': len(check),
        'pair_correlations_computed': len(ranking), 'strong_correlation_pairs': len(strong),
        'same_industry_independent_strong_pairs': len(peers),
        'coint_tests_training': len(keys), 'research_candidates': len(candidates),
        'selection': 'Same GICS sub-industry, different CIK, training daily log-return correlation >= 0.8, '
                     'last 126 training returns correlation >= 0.75, training raw Engle-Granger p < 0.05, positive training beta.',
        'top_correlated': top, 'all_strong_correlations': all_strong,
        'same_industry_strong': peers, 'candidate_shortlist': candidates,
    }
    results_file.write_text(json.dumps(summary, indent=2, allow_nan=False))
    lines = [f'# {args.name} — {len(eligible)} stocks — {started_at:%Y-%m-%d}', '',
             f'Run started: {started_at.isoformat()} ({REPORT_TIMEZONE}).', '',
             f"Training: {summary['training_start']} to {summary['training_end']}; "
             f"later diagnostic period: {summary['check_start']} to {summary['check_end']}.", '',
             f"{len(tickers)} requested stocks; {len(eligible)} with complete positive training prices; "
             f"{len(ranking):,} pair correlations; {len(strong)} with training correlation >= 0.80.", '',
             'Correlation is Pearson correlation of daily log returns from adjusted closing prices, '
             'not correlation of price levels. 1 means perfectly matching linear daily-return moves; '
             'it does not measure expected profit. No missing prices were filled.', '',
             '## Method and limits', '',
             summary['selection'], '',
             'The later period is a diagnostic only and is not used for ranking. It has already been '
             'examined in this project, so it is not an untouched final test. Any refinements need '
             'a new chronological validation/test plan or future paper-trading data.', '',
             'Engle-Granger is run on price levels, with an intercept, AIC lag selection and alphabetical '
             'ticker ordering (first ticker is dependent). Later-period cointegration is separately '
             'refitted; it is not a fixed-hedge profitability test. P-values are raw/unadjusted, following '
             'a data-based correlation screen. Multiple testing, changing membership and the I(1) '
             'assumption still need assessment; passing this screen is not proof of a valid strategy.', '',
             'This uses the project\'s fetched present-day constituent snapshot, not historical membership. '
             'Eligibility uses training prices only; later missing rows can reduce diagnostic observations.', '',
             'Sources: [Yahoo Finance](https://finance.yahoo.com/) via '
             '[yfinance](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html); '
             '[constituents](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies); '
             '[Engle-Granger documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html).', '']
    def table(title, rows):
        lines.extend([f'## {title}', '',
                      '| Pair | Companies | Industry | Train r | Later r | Train coint p | Later coint p | Beta |',
                      '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |'])
        for r in rows:
            later_r = 'N/A' if r['check_return_correlation'] is None else f"{r['check_return_correlation']:.3f}"
            later_p = 'N/A' if r['check_coint_pvalue'] is None else f"{r['check_coint_pvalue']:.4g}"
            companies = (r['company1'] + ' / ' + r['company2']).replace('|', '\\|')
            industry = r['industry1'] if r['same_industry'] else 'Different industries'
            if r['same_issuer']:
                industry += ' (same issuer)'
            lines.append(f"| {r['ticker1']}/{r['ticker2']} | {companies} | {industry} | "
                         f"{r['train_return_correlation']:.3f} | {later_r} | "
                         f"{r['train_coint_pvalue']:.4g} | {later_p} | {r['train_hedge_ratio']:.3f} |")
        lines.append('')
    table('20 strongest training correlations across all industries', top)
    table('Exploratory mean-reversion candidates, selected on training data', candidates)
    table('All pairs with training daily-return correlation at least 0.80', all_strong)
    warnings_found = [(a, b, r['warnings']) for (a, b), r in results.items() if r['warnings']]
    if warnings_found:
        lines.extend(['## Statistical warnings', ''])
        lines.extend(f'- {a}/{b}: {"; ".join(messages)}' for a, b, messages in warnings_found)
    report_file.write_text('\n'.join(lines) + '\n')
    update_report_index()
    print(f'{len(candidates)} exploratory mean-reversion candidates. Report: {report_file}', flush=True)
    for r in candidates[:15]:
        print(r['ticker1'], r['ticker2'], f"train r={r['train_return_correlation']:.3f}",
              f"later r={r['check_return_correlation']:.3f}",
              f"train p={r['train_coint_pvalue']:.4g}", f"later p={r['check_coint_pvalue']:.4g}", flush=True)


if __name__ == '__main__':
    main()
