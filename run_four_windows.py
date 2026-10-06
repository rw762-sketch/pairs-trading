"""Four non-overlapping annual development blocks and separate final-year trades."""
from dataclasses import asdict,replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import hashlib,json,warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from threadpoolctl import threadpool_limits
from config import Config
from data_fetcher import reserve_testing_period
from selection import apply_filters
from evaluation import evaluate_pairs
from run_persistence import persistent_mask
from run_optimization import plot_account


def annual_blocks(index):
    """Use every development close once, in four near-equal annual blocks."""
    if not index.is_unique or not index.is_monotonic_increasing or len(index)<4*60:
        raise ValueError('Need chronological dates and sufficient development history.')
    return [index.take(part) for part in np.array_split(np.arange(len(index)),4)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--required', type=int, choices=range(1, 5), default=2)
    parser.add_argument('--half-life-max', type=float, default=Config().half_life_max)
    parser.add_argument('--entry', type=float, default=Config().entry)
    parser.add_argument('--exit', type=float, default=0.25)
    args = parser.parse_args()
    if not np.isfinite(args.half_life_max) or args.half_life_max < Config().half_life_min:
        parser.error('Half-life ceiling must be finite and at least the minimum.')
    config=replace(Config(),entry=args.entry,exit=args.exit,half_life_max=args.half_life_max)
    if not 0 <= config.exit < config.entry < config.stop:
        parser.error('Require 0 <= exit < entry < stop.')
    prices=pd.read_pickle('data/cache/adjusted-prices.pkl')
    development,reservation=reserve_testing_period(prices)
    blocks=annual_blocks(development.index)
    train=development.tail(252)
    output=Path('results')/datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d_%H-%M-%S');output.mkdir()
    source=Path('results/2026-10-05_20-41-12/final-formation-tests.csv')
    panel=pd.read_csv(source)
    # Confirm cached final formation regressions match today's development inputs.
    for row in panel.itertuples():
        if row.Status=='tested':
            coefficients=np.linalg.lstsq(np.column_stack([np.ones(len(train)),train[row.Ticker2]]),train[row.Ticker1],rcond=None)[0]
            np.testing.assert_allclose(coefficients,[row.Alpha,row.Beta],rtol=1e-9,atol=1e-9)
    candidates=panel.loc[panel.Selected].copy()
    plan={'recorded_before_simulation':datetime.now(ZoneInfo('America/New_York')).isoformat(),'config':asdict(config),'reservation':reservation,'windows':[{'Window':n,'Start':str(block[0].date()),'End':str(block[-1].date()),'Closes':len(block)} for n,block in enumerate(blocks,1)],'rule':f'p < 0.05 in at least {args.required}/4 non-overlapping annual development blocks; no latest-of-four requirement. Existing final pre-trading cointegration, positive beta, integration, recovery and stock-exposure filters retained.','candidate_universe':'387 final pre-trading within-cluster qualifying candidates, not all stock combinations. Same pre-trading K-means screen as earlier comparisons.','fresh_holdout':False,'price_sha256':hashlib.sha256(Path('data/cache/adjusted-prices.pkl').read_bytes()).hexdigest(),'cached_screen_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    values=pd.DataFrame(index=candidates.Pair,columns=[f'P_Window_{n}' for n in range(1,5)],dtype=float)
    errors=[]
    with threadpool_limits(limits=1):
        for n,block in enumerate(blocks,1):
            data=development.loc[block]
            for row in candidates.itertuples():
                a=data[row.Ticker1].to_numpy();b=data[row.Ticker2].to_numpy();value=np.nan
                if np.isfinite(a).all() and np.isfinite(b).all() and (a>0).all() and (b>0).all():
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            value=float(coint(a,b,trend='c',autolag='aic')[1])
                    except (ValueError,np.linalg.LinAlgError) as exc:
                        errors.append({'Window':n,'Pair':row.Pair,'Error':str(exc)})
                else:errors.append({'Window':n,'Pair':row.Pair,'Error':'Unavailable complete positive prices'})
                values.loc[row.Pair,f'P_Window_{n}']=value
            print(f'Window {n}: {block[0].date()} to {block[-1].date()}, {len(block)} closes, {len(candidates)} pair tests',flush=True)
    keep=persistent_mask(values,required=args.required,require_latest=False)
    values['Pass_Count']=values.lt(.05).sum(axis=1);values['Persistent']=keep
    values.to_csv(output/'persistence-tests.csv',index_label='Pair')
    pd.DataFrame(errors,columns=['Window','Pair','Error']).to_csv(output/'failed-tests.csv',index=False)
    panel.to_csv(output/'final-formation-tests.csv',index=False)
    panel.loc[~panel.Pair.isin(keep.index[keep]),'Selected']=False
    filtered=apply_filters(panel,train,config);filtered.to_csv(output/'selection-audit.csv',index=False)
    selected=filtered.loc[filtered.Selected].copy();selected.to_csv(output/'selected-pairs.csv',index=False)
    print(f'{keep.sum()} passed {args.required}/4; {len(selected)} after recovery/exposure filters',flush=True)
    dates=prices.index[-252:]
    portfolio,signals,pairs=evaluate_pairs(prices,dates,selected,config,output)
    portfolio['equity_curve'].to_csv(output/'equity.csv',index_label='Date');portfolio['trades'].to_csv(output/'trades.csv',index=False)
    pairs.to_csv(output/'pair-metrics.csv',index=False)
    pd.DataFrame([portfolio['metrics']]).to_csv(output/'performance.csv',index=False)
    plot_account(portfolio,output/'cumulative-return.png',f'{args.required}-of-4 annual windows: designated final validation')
    m=portfolio['metrics'];assert np.isclose(portfolio['trades'].Net_PnL.sum(),m['Final_Portfolio_Value']-config.capital)
    trades=portfolio['trades']
    if len(trades):
        assert pd.to_datetime(trades.Entry_Date).min()>=dates[0]
        assert pd.to_datetime(trades.Exit_Date).max()<=dates[-1]
    html='<html><meta charset="utf-8"><title>'+str(args.required)+'-of-4 annual windows</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>'+str(args.required)+'-of-4 annual development windows</h1><p>Four disjoint development blocks; no trades during those blocks. Candidates passing at least '+str(args.required)+' cointegration tests are traded only in '+reservation['reserved_start']+' to '+reservation['reserved_end']+'. Entry +/-'+str(config.entry)+', exit +/-'+str(config.exit)+', stop +/-'+str(config.stop)+', no holding timeout. Half-life ceiling '+str(config.half_life_max)+' trading days. Final period previously inspected.</p>'+pd.DataFrame(plan['windows']).to_html(index=False)+'<h2>Final performance</h2>'+pd.DataFrame.from_dict(m,orient='index',columns=['Value']).to_html()+'<h2>Selected pairs</h2>'+selected[['Pair','Raw_P','Beta','Half_Life','Weight']].to_html(index=False)+'<img src="cumulative-return.png" style="width:100%;max-width:900px"><h2>Records</h2><ul>'+''.join(f'<li><a href="{x}">{x}</a></li>' for x in ['plan.json','persistence-tests.csv','selected-pairs.csv','selection-audit.csv','trades.csv','pair-metrics.csv','performance.csv'])+'</ul><p>Persistence counts are among final pre-trading candidates; raw p-values do not adjust for multiple testing. Near-equal calendar spans contain slightly different close counts. Exposure and pair selection can differ from the six-window version.</p></html>'
    (output/'report.html').write_text(html)
    print(f"Net return: {m['Total_Return']:.2%}; {m['Num_Trades']} trades; results: {output.resolve()}",flush=True)

if __name__=='__main__':main()
