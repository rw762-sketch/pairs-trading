"""Reproduce the explicitly selected historical strategy, with pinned settings."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib,json
import numpy as np
import pandas as pd
from config import Config
from data_fetcher import reserve_testing_period
from run_persistence import persistent_mask
from selection import apply_filters
from evaluation import evaluate_pairs
from allocation import persistence_weights
from run_optimization import plot_account

ROOT=Path(__file__).resolve().parent
REPLAY_FILES = (
    'adjusted-prices.pkl', 'final-formation-tests.csv', 'persistence-tests.csv',
    'selected-pairs.csv', 'unlimited-trades.csv', 'plan.json',
)

def load_selected_strategy():
    return json.loads((ROOT/'selected_strategy.json').read_text())


def resolve_replay_inputs(chosen, cache):
    """Use committed frozen inputs when a fresh clone lacks the local snapshot."""
    cache = Path(cache)
    price_file = cache / 'adjusted-prices.pkl'
    reference = ROOT / chosen['reference_run']
    bundle = ROOT / 'reproducibility/selected'
    default_cache = (ROOT / 'data/cache').resolve()
    bundled_prices = not price_file.exists() and cache.resolve() == default_cache
    if not price_file.exists() and not bundled_prices:
        raise FileNotFoundError(f'Selected replay price snapshot is missing: {price_file}')
    bundled_screens = not all((reference / name).exists() for name in REPLAY_FILES[1:])
    if bundled_prices or bundled_screens:
        manifest_file = bundle / 'manifest.json'
        if not manifest_file.exists():
            raise FileNotFoundError(
                'Frozen replay inputs are missing. Restore reproducibility/selected '
                'from the repository, or provide the original local snapshot and reference run.'
            )
        manifest = json.loads(manifest_file.read_text())
        strategy_hash = hashlib.sha256((ROOT / 'selected_strategy.json').read_bytes()).hexdigest()
        if manifest.get('selected_strategy_sha256') != strategy_hash:
            raise ValueError('Frozen replay bundle does not match selected_strategy.json.')
        expected = manifest.get('file_sha256', {})
        if set(expected) != set(REPLAY_FILES):
            raise ValueError('Frozen replay manifest has an invalid input-file list.')
        for name in REPLAY_FILES:
            path = bundle / name
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
                raise ValueError(f'Frozen replay input failed SHA256 verification: {name}')
        if expected['adjusted-prices.pkl'] != chosen['price_sha256']:
            raise ValueError('Frozen replay bundle has a different price snapshot.')
    if bundled_prices:
        price_file = bundle / 'adjusted-prices.pkl'
    if bundled_screens:
        reference = bundle
    return price_file, reference


def run_selected_strategy(cache=ROOT/'data/cache',output_root=ROOT/'results'):
    chosen=load_selected_strategy()
    config=Config(**chosen['config'])
    cache=Path(cache);output_root=Path(output_root)
    price_file,reference=resolve_replay_inputs(chosen,cache)
    if hashlib.sha256(price_file.read_bytes()).hexdigest()!=chosen['price_sha256']:
        raise ValueError('Selected replay requires the original price snapshot. Recompute screens for changed data.')
    prices=pd.read_pickle(price_file)
    development,reservation=reserve_testing_period(prices,chosen['reserved_closes'],config.formation)
    panel=pd.read_csv(reference/'final-formation-tests.csv')
    evidence=pd.read_csv(reference/'persistence-tests.csv',index_col='Pair')
    pvalues=evidence[[f'P_Window_{n}' for n in range(1,7)]]
    keep=persistent_mask(pvalues,required=chosen['required_passes'],require_latest=chosen['require_latest'])
    panel.loc[~panel.Pair.isin(keep.index[keep]),'Selected']=False
    filtered=apply_filters(panel,development.tail(config.formation),config)
    selected=filtered.loc[filtered.Selected].copy()
    original=pd.read_csv(reference/'selected-pairs.csv')
    pd.testing.assert_series_equal(selected.Pair.reset_index(drop=True),original.Pair.reset_index(drop=True))
    weights=persistence_weights(pvalues.loc[selected.Pair],config.pvalue,chosen['allocation_rule']['pvalue_floor'])
    selected['Weight']=selected.Pair.map(weights.Weight)
    for column in ['Pass_Count','Mean_P','Allocation_Score']:
        selected[column]=selected.Pair.map(weights[column])
    filtered['Weight']=filtered.Pair.map(weights.Weight).fillna(0.0)
    np.testing.assert_allclose(selected.Weight.sum(),1.0)
    output_root.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d_%H-%M-%S')
    output=output_root/stamp
    if output.exists():raise ValueError('Timestamp already exists; retry in another second.')
    output.mkdir()
    (output/'plan.json').write_text(json.dumps({'recorded_before_simulation':datetime.now(ZoneInfo('America/New_York')).isoformat(),**chosen,'reservation':reservation},indent=2)+'\n')
    dates=prices.index[-chosen['reserved_closes']:]
    portfolio,_,pairs=evaluate_pairs(prices,dates,selected,config,output)
    metrics=portfolio['metrics']
    original_trades=pd.read_csv(reference/'unlimited-trades.csv')
    original_net=original_trades.groupby('Pair').Net_PnL.sum()
    original_weights=original.set_index('Pair').Weight
    expected_pnl=sum(original_net[pair]*weights.loc[pair,'Weight']/original_weights[pair] for pair in weights.index)
    np.testing.assert_allclose(metrics['Final_Portfolio_Value']-config.capital,expected_pnl,rtol=1e-9,atol=1e-8)
    weights.to_csv(output/'allocation.csv',index_label='Pair')
    selected.to_csv(output/'selected-pairs.csv',index=False)
    evidence.to_csv(output/'persistence-tests.csv',index_label='Pair')
    filtered.to_csv(output/'selection-audit.csv',index=False)
    portfolio['equity_curve'].to_csv(output/'equity.csv',index_label='Date')
    portfolio['trades'].to_csv(output/'trades.csv',index=False)
    pairs.to_csv(output/'pair-metrics.csv',index=False)
    pd.DataFrame([metrics]).to_csv(output/'performance.csv',index=False)
    plot_account(portfolio,output/'cumulative-return.png','Selected 3-of-6 strategy: historical final-year return')
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Selected pairs strategy</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>Selected 3-of-6 strategy</h1><p>Six overlapping one-year development formations. At least three p-values below 0.05 including the latest; final pre-trading cointegration screen also required. Entry +/-2, exit +/-0.5, adverse stop +/-3.5, no holding timeout. Recovery half-life 1–20 closes, at least 12 annual mean crossings and no shared stocks. Weights proportional to periods passed divided by average p-value across all six periods, without a pair allocation cap; '+f'{max(0,1-selected.Weight.sum()):.2%}'+' unassigned capital. Inactive pair allocations remain cash.</p><p>Trading '+reservation['reserved_start']+' through '+reservation['reserved_end']+'. Signal/selection version retained from the highest-return comparison; capital allocation subsequently increased at user request. Historical evaluation previously inspected; not untouched validation.</p><h2>Performance</h2>'+pd.DataFrame.from_dict(metrics,orient='index',columns=['Value']).to_html()+'<h2>Pairs</h2>'+selected[['Pair','Pass_Count','Mean_P','Weight','Half_Life']].to_html(index=False)+'<img src="cumulative-return.png" style="width:100%"><h2>Records</h2><ul>'+''.join(f'<li><a href="{x}">{x}</a></li>' for x in ['plan.json','allocation.csv','selected-pairs.csv','persistence-tests.csv','selection-audit.csv','performance.csv','pair-metrics.csv','trades.csv','equity.csv'])+'</ul>')
    (output_root/'LATEST.txt').write_text(str(output.resolve())+'\n')
    (output_root/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={output.name}/report.html"><a href="{output.name}/report.html">Selected strategy</a>')
    archived=[p for p in sorted(output_root.iterdir()) if p.is_dir() and p.name not in {output.name,'archive'}]
    (output_root/'archive.html').write_text('<!doctype html><meta charset="utf-8"><h1>Archived experiment records</h1><p>Historical runs retained for reproducibility. The active selected strategy is linked from index.html.</p><ul>'+''.join(f'<li><a href="{p.name}/report.html">{p.name}</a></li>' for p in archived if (p/'report.html').exists())+'</ul>')
    print(f"Selected strategy: {metrics['Total_Return']:.2%}; {metrics['Num_Trades']} trades; results: {output.resolve()}",flush=True)
    return {'portfolio':portfolio,'selected':selected,'output':output}

if __name__=='__main__':run_selected_strategy()
