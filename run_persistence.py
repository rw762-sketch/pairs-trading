"""Six-formation persistence screen followed by a reused historical test block."""
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from threadpoolctl import threadpool_limits
from config import Config
from data_fetcher import reserve_testing_period
from walk_forward import schedule_windows
from cointegration import screen_pairs
from stock_clustering import cluster_stocks, within_cluster_pairs
from selection import apply_filters
from evaluation import evaluate_pairs


def persistent_mask(pvalues, threshold=0.05, required=5, require_latest=False):
    """Count passes; optionally require the latest formation to pass."""
    passed = pvalues.lt(threshold)
    eligible = passed.sum(axis=1).ge(required)
    return eligible & passed.iloc[:, -1] if require_latest else eligible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--required', type=int, default=5, choices=range(1, 7))
    parser.add_argument('--screen-results', type=Path)
    parser.add_argument('--require-latest', action='store_true', help='Restore the optional latest-of-six requirement.')
    parser.add_argument('--entry', type=float, default=Config().entry)
    parser.add_argument('--exit', type=float, default=Config().exit)
    args = parser.parse_args()
    config = replace(Config(), entry=args.entry, exit=args.exit)
    if not 0 <= config.exit < config.entry < config.stop:
        parser.error("Require 0 <= exit < entry < stop")
    prices = pd.read_pickle('data/cache/adjusted-prices.pkl')
    development, reservation = reserve_testing_period(prices)
    schedule = schedule_windows(development.index)
    if len(schedule) != 6:
        raise ValueError('This experiment requires six development formations.')
    output = Path('results') / datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d_%H-%M-%S')
    output.mkdir()
    plan = dict(config=asdict(config), reservation=reservation,
                persistence=f'Raw p < 0.05 in at least {args.required}/6 formation windows; latest-of-six required: {args.require_latest}; final pre-test formation must also pass.',
                price_sha256=hashlib.sha256(Path('data/cache/adjusted-prices.pkl').read_bytes()).hexdigest(),
                hold_limits=[config.max_hold], fresh_holdout=False,
                note='Historical test period previously inspected. No variant selected using test returns. Persistence replaces previous-loss exclusion for this experiment; existing recovery and exposure rules retained.')
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    # Final candidates and fits use only the 252 observations before testing.
    train=development.tail(config.formation)
    valid=np.isfinite(train).all() & (train>0).all()
    train=train.loc[:,valid]
    if args.screen_results:
        previous=json.loads((args.screen_results/'plan.json').read_text())
        if previous['reservation'] != reservation:
            raise ValueError('Cached reservation differs.')
        for key,value in asdict(config).items():
            if key not in {'max_hold', 'entry', 'exit', 'half_life_min', 'half_life_max', 'min_crossings', 'max_pair_weight'} and previous['config'][key] != value:
                raise ValueError(f'Cached screening configuration differs: {key}')
        panel=pd.read_csv(args.screen_results/'final-formation-tests.csv')
        saved=pd.read_csv(args.screen_results/'persistence-tests.csv',index_col='Pair')
        values=saved[[f'P_Window_{n}' for n in range(1,7)]].copy()
        # Validate all final fits against current formation prices before reuse.
        for row in panel.itertuples():
            if row.Status == 'tested':
                a,b=np.linalg.lstsq(np.column_stack([np.ones(len(train)),train[row.Ticker2]]),train[row.Ticker1],rcond=None)[0]
                np.testing.assert_allclose([a,b],[row.Alpha,row.Beta],rtol=1e-9,atol=1e-9)
        plan['screen_results']=str(args.screen_results.resolve())
        plan['cached_screens_sha256']={name:hashlib.sha256((args.screen_results/name).read_bytes()).hexdigest() for name in ['final-formation-tests.csv','persistence-tests.csv']}
        (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    else:
        labels,_=cluster_stocks(train,config.clusters,config.seed)
        family=within_cluster_pairs(labels)
        print(f'Final formation: {len(family)} within-cluster tests',flush=True)
        with threadpool_limits(limits=1):
            panel,_=screen_pairs(train,family,progress=True)
        panel.to_csv(output/'final-formation-tests.csv',index=False)
        candidates=panel.loc[panel.Selected].copy()
        values=pd.DataFrame(index=candidates.Pair,columns=range(1,7),dtype=float)
        cache=Path('results/2026-10-05_20-11-00/windows')
        with threadpool_limits(limits=1):
            for number,(formation,_) in enumerate(schedule,1):
                cached=pd.read_csv(cache/f'{number:02d}/cointegration-tests.csv').set_index('Pair')
                training=prices.loc[formation]
                missing=0
                for r in candidates.itertuples():
                    if r.Pair in cached.index:
                        value=cached.loc[r.Pair,'Raw_P']
                    else:
                        a=training[r.Ticker1].to_numpy();b=training[r.Ticker2].to_numpy()
                        value=np.nan
                        if np.isfinite(a).all() and np.isfinite(b).all() and (a>0).all() and (b>0).all():
                            value=float(coint(a,b,trend='c',autolag='aic')[1]);missing+=1
                    values.loc[r.Pair,number]=value
                print(f'Formation {number}: {missing} additional tests for pairs outside earlier clusters',flush=True)
    keep=persistent_mask(values,required=args.required,require_latest=args.require_latest)
    values.columns=[f'P_Window_{n}' for n in range(1,7)]
    values['Pass_Count']=values.lt(.05).sum(axis=1)
    values['Persistent']=keep
    values.to_csv(output/'persistence-tests.csv',index_label='Pair')
    panel.to_csv(output/'final-formation-tests.csv',index=False)
    panel.loc[~panel.Pair.isin(values.index[keep]),'Selected']=False
    selected_panel=apply_filters(panel,train,config)
    selected=selected_panel.loc[selected_panel.Selected].copy()
    selected.to_csv(output/'selected-pairs.csv',index=False)
    print(f'{keep.sum()} persistent candidates; {len(selected)} after recovery/exposure filters',flush=True)
    dates=prices.index[-252:]
    rows=[]
    for limit in [config.max_hold]:
        label=f'{limit}-day' if limit else 'unlimited'
        portfolio,_,pair_metrics=evaluate_pairs(prices,dates,selected,replace(config,max_hold=limit))
        portfolio['equity_curve'].to_csv(output/f'{label}-account.csv')
        portfolio['trades'].to_csv(output/f'{label}-trades.csv',index=False)
        pair_metrics.to_csv(output/f'{label}-pairs.csv',index=False)
        rows.append({'Holding_Limit':limit if limit else 'none','Selected_Pairs':len(selected),**portfolio['metrics']})
        print(f'{label}: {portfolio["metrics"]["Total_Return"]:.2%}',flush=True)
    summary=pd.DataFrame(rows)
    summary.to_csv(output/'comparison.csv',index=False)
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Persistence experiment</title><h1>'+str(args.required)+'-of-6 persistence experiment</h1><p>Historical evaluation, not untouched final validation. Reserved 252 closes: '+reservation['reserved_start']+' through '+reservation['reserved_end']+'. No per-trade timeout with the current default; convergence, stops and test-end liquidation remain active. Recovery half-life ceiling: '+str(config.half_life_max)+' days.</p>'+summary[['Holding_Limit','Selected_Pairs','Total_Return','Sharpe_Ratio','Max_Drawdown','Num_Trades','Total_Costs']].to_html(index=False)+'<p>See plan.json, persistence-tests.csv, final-formation-tests.csv and per-variant trade records for full evidence.</p>')
    print(f'Results: {output.resolve()}',flush=True)

if __name__=='__main__':
    main()
