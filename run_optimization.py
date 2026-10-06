"""Development-only threshold search, frozen historical final validation, artifacts."""
from dataclasses import asdict, replace
from datetime import datetime
from itertools import product
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from threadpoolctl import threadpool_limits
from config import Config
from data_fetcher import reserve_testing_period
from walk_forward import schedule_windows
from selection import apply_filters
from evaluation import evaluate_pairs

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'results/2026-10-05_20-11-00/windows'
FINAL_CACHE=ROOT/'results/2026-10-05_20-41-12'

def select(panel, pvalues, training, config):
    out=panel.copy()
    persistence=pvalues.lt(config.pvalue).sum(axis=1).ge(3)
    out['Selected']=(out.Raw_P.lt(config.pvalue) & out.Status.eq('tested') & out.I1_Compatible & out.Beta.gt(0) & np.isfinite(out.Half_Life) & out.Pair.isin(persistence.index[persistence]))
    return apply_filters(out,training,config)

def formation_evidence(prices, formation, panel, histories, destination):
    train=prices.loc[formation]
    eligible=panel.loc[panel.Raw_P.lt(.10) & panel.I1_Compatible & panel.Beta.gt(0) & np.isfinite(panel.Half_Life)]
    values=pd.DataFrame(index=eligible.Pair,columns=[f'P_Window_{i}' for i in range(1,7)],dtype=float)
    with threadpool_limits(limits=1):
        for n,history in enumerate(histories,1):
            cached=pd.read_csv(CACHE/f'{n:02d}/cointegration-tests.csv').set_index('Pair')
            data=prices.loc[history]
            additional=0
            for r in eligible.itertuples():
                if r.Pair in cached.index:
                    value=cached.loc[r.Pair,'Raw_P']
                else:
                    a=data[r.Ticker1].to_numpy();b=data[r.Ticker2].to_numpy();value=np.nan
                    if np.isfinite(a).all() and np.isfinite(b).all() and (a>0).all() and (b>0).all():
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                value=float(coint(a,b,trend='c',autolag='aic')[1])
                        except (ValueError,np.linalg.LinAlgError):
                            pass
                        additional+=1
                values.loc[r.Pair,f'P_Window_{n}']=value
            print(f'{destination.name}: window {n}, {additional} additional tests',flush=True)
    destination.mkdir(exist_ok=True)
    panel.to_csv(destination/'cointegration-tests.csv',index=False)
    values.to_csv(destination/'persistence-pvalues.csv',index_label='Pair')
    return train,values

def fit_hedges(selected, prices, cutoff, length):
    out=selected.copy()
    for index,row in out.iterrows():
        data=prices.loc[:cutoff,[row.Ticker1,row.Ticker2]].tail(length)
        if len(data)!=length or not np.isfinite(data.to_numpy()).all():
            out.loc[index,'Weight']=0
            continue
        alpha,beta=np.linalg.lstsq(np.column_stack([np.ones(length),data[row.Ticker2]]),data[row.Ticker1],rcond=None)[0]
        out.loc[index,['Alpha','Beta']]=[alpha,beta]
        if beta<=0:
            out.loc[index,'Weight']=0
    return out.loc[out.Weight.gt(0)].copy()

def plot_account(portfolio,path,title):
    curve=portfolio['equity_curve'];fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(curve.index,100*(curve/portfolio['metrics']['Initial_Capital']-1));ax.set(title=title,ylabel='Cumulative return (%)');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)

def save_account(portfolio,pairs,destination):
    destination.mkdir(exist_ok=True)
    portfolio['equity_curve'].to_csv(destination/'equity.csv',index_label='Date')
    portfolio['trades'].to_csv(destination/'trades.csv',index=False)
    pairs.to_csv(destination/'pair-metrics.csv',index=False)

def main():
    config=Config()
    prices=pd.read_pickle(ROOT/'data/cache/adjusted-prices.pkl')
    development,reservation=reserve_testing_period(prices)
    schedule=schedule_windows(development.index)
    histories=[f for f,_ in schedule]
    if len(histories)!=6: raise ValueError('Expected six training periods.')
    dates=schedule[-1][1]
    assert histories[-1][-1]<dates[0] and dates[-1]<prices.index[-252]
    output=ROOT/'results'/datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d_%H-%M-%S');output.mkdir()
    plan={'recorded_before_search':datetime.now(ZoneInfo('America/New_York')).isoformat(),'config':asdict(config),'reservation':reservation,'persistence':'At least 3/6, no latest-of-six requirement; final pre-trading screen required.','development_validation_start':str(dates[0].date()),'development_validation_end':str(dates[-1].date()),'grid':{'pvalue':[.01,.05,.10],'entry':[1.5,2.,2.5],'exit':[0.,.25,.5]},'selection':'Highest development Sharpe among configurations with >=5 trades; tie: net return then grid order. Sensitivities do not change the chosen configuration.','fresh_holdout':False,'price_sha256':hashlib.sha256((ROOT/'data/cache/adjusted-prices.pkl').read_bytes()).hexdigest(),'limitation':'One short development trading validation block after six original formations. Final period previously inspected.'}
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    panel=pd.read_csv(CACHE/'06/cointegration-tests.csv')
    train,pvalues=formation_evidence(prices,histories[-1],panel,histories,output/'development')
    results=[];portfolios={}
    for i,(p,entry,exit) in enumerate(product([.01,.05,.10],[1.5,2.,2.5],[0.,.25,.5])):
        settings=replace(config,pvalue=p,entry=entry,exit=exit)
        filtered=select(panel,pvalues,train,settings);selected=filtered.loc[filtered.Selected].copy()
        portfolio,_,pairs=evaluate_pairs(development,dates,selected,settings)
        results.append({'ID':i,'P_Cutoff':p,'Entry_Z':entry,'Exit_Z':exit,'Selected_Pairs':len(selected),**portfolio['metrics']})
        portfolios[i]=(portfolio,selected,pairs)
    grid=pd.DataFrame(results);grid.to_csv(output/'parameter-grid.csv',index=False)
    eligible=grid.loc[grid.Num_Trades.ge(5)].sort_values(['Sharpe_Ratio','Total_Return','ID'],ascending=[False,False,True])
    if eligible.empty: raise ValueError('No configuration meets the predeclared minimum of five development trades.')
    winner=eligible.iloc[0];settings=replace(config,pvalue=float(winner.P_Cutoff),entry=float(winner.Entry_Z),exit=float(winner.Exit_Z))
    chosen={'frozen_before_final_evaluation':datetime.now(ZoneInfo('America/New_York')).isoformat(),'config':asdict(settings),'selected_grid_id':int(winner.ID),'development_metrics':winner.to_dict(),'selection_rule':plan['selection']}
    (output/'frozen-settings.json').write_text(json.dumps(chosen,indent=2)+'\n')
    print('FROZEN:',chosen['config'],flush=True)
    dev_portfolio,dev_selected,dev_pairs=portfolios[int(winner.ID)]
    save_account(dev_portfolio,dev_pairs,output/'development');dev_selected.to_csv(output/'development/selected-pairs.csv',index=False)
    sensitivity=[]
    cases=[('hedge_lookback',n) for n in [126,252,504]]+[('stop_z',n) for n in [3.,3.5,4.]]+[('holding_limit',n) for n in [0,20,40,60]]+[('transaction_fee',n) for n in [0.,.0005,.001,.002]]
    for kind,value in cases:
        local=settings;selected=dev_selected.copy()
        if kind=='hedge_lookback': selected=fit_hedges(selected,development,train.index[-1],value)
        elif kind=='stop_z':local=replace(settings,stop=value)
        elif kind=='holding_limit':local=replace(settings,max_hold=value)
        else:local=replace(settings,fee=value)
        portfolio,_,_=evaluate_pairs(development,dates,selected,local)
        sensitivity.append({'Parameter':kind,'Value':value,'Selected_Pairs':len(selected),**portfolio['metrics']})
    pd.DataFrame(sensitivity).to_csv(output/'robustness.csv',index=False)
    # Only now build final selections and evaluate final prices with frozen settings.
    final_panel=pd.read_csv(FINAL_CACHE/'final-formation-tests.csv')
    final_train,final_values=formation_evidence(prices,development.index[-252:],final_panel,histories,output/'final-validation')
    final_filtered=select(final_panel,final_values,final_train,settings)
    final_selected=final_filtered.loc[final_filtered.Selected].copy()
    final_selected.to_csv(output/'final-validation/selected-pairs.csv',index=False)
    final_portfolio,signals,final_pairs=evaluate_pairs(prices,prices.index[-252:],final_selected,settings,output/'final-validation')
    save_account(final_portfolio,final_pairs,output/'final-validation')
    metrics=pd.DataFrame([{'Period':'development validation',**dev_portfolio['metrics']},{'Period':'designated final validation',**final_portfolio['metrics']}]);metrics.to_csv(output/'performance.csv',index=False)
    figures=output/'figures';figures.mkdir()
    plot_account(dev_portfolio,figures/'development-equity.png','Development validation')
    plot_account(final_portfolio,figures/'final-equity.png','Designated final validation')
    for p in [.01,.05,.10]:
        data=grid.loc[grid.P_Cutoff.eq(p)].pivot(index='Entry_Z',columns='Exit_Z',values='Total_Return')*100
        fig,ax=plt.subplots(figsize=(6,4));im=ax.imshow(data,cmap='RdYlGn');ax.set(xticks=range(3),xticklabels=data.columns,yticks=range(3),yticklabels=data.index,xlabel='Exit z',ylabel='Entry z',title=f'Development net return (%), p < {p}')
        for (i,j),v in np.ndenumerate(data.to_numpy()):ax.text(j,i,f'{v:.2f}',ha='center',va='center')
        fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(figures/f'sensitivity-{p}.png',dpi=140);plt.close(fig)
    for pair,sig in signals.items():
        fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
        axes[0].plot(sig.index,sig.Spread);axes[0].set(title=pair,ylabel='Spread')
        # Use saved z column from signal generator.
        zcol='Z_Score' if 'Z_Score' in sig else 'ZScore'
        axes[1].plot(sig.index,sig[zcol]);axes[1].set(ylabel='Z-score')
        for value in [settings.entry,-settings.entry]:axes[1].axhline(value,color='red',ls='--')
        for value in [settings.exit,-settings.exit]:axes[1].axhline(value,color='green',ls=':')
        fig.tight_layout();fig.savefig(figures/f'{pair}.png',dpi=140);plt.close(fig)
    # Final cointegration diagnostics are reported only; they cannot alter trading.
    post=[]
    for r in final_selected.itertuples():
        a=prices.loc[prices.index[-252:],r.Ticker1];b=prices.loc[prices.index[-252:],r.Ticker2]
        if np.isfinite(a).all() and np.isfinite(b).all():
            with threadpool_limits(limits=1):pv=float(coint(a,b,trend='c',autolag='aic')[1])
        else:pv=np.nan
        post.append({'Pair':r.Pair,'Final_Period_P':pv,'Diagnostic_Only':True})
    pd.DataFrame(post,columns=['Pair','Final_Period_P','Diagnostic_Only']).to_csv(output/'final-validation/post-test-cointegration.csv',index=False)
    trade=final_portfolio['trades'];trade.groupby('Exit_Reason').Net_PnL.agg(['count','sum','mean']).to_csv(output/'exit-reasons.csv')
    for frame in [dev_portfolio,final_portfolio]:
        assert np.isclose(frame['trades'].Net_PnL.sum(),frame['metrics']['Final_Portfolio_Value']-config.capital)
    imgs=''.join(f'<img src="figures/{p.name}" style="max-width:900px;width:100%">' for p in figures.glob('*.png'))
    html='<html><meta charset="utf-8"><title>Pairs trading parameter study</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}img{display:block;margin:25px 0}</style><h1>Pairs trading parameter study</h1><p>Development validation: '+plan['development_validation_start']+' to '+plan['development_validation_end']+'. Final validation: '+reservation['reserved_start']+' to '+reservation['reserved_end']+'. Final dates were previously inspected. Settings selected only from development; a single short development validation block limits confidence. Raw p-values and overlapping training windows do not control multiple-testing bias.</p><h2>Frozen settings</h2><pre>'+json.dumps(asdict(settings),indent=2)+'</pre><h2>Performance</h2>'+metrics.to_html(index=False)+'<h2>Final selected pairs</h2>'+final_selected.to_html(index=False)+'<h2>Parameter search (development only)</h2>'+grid.to_html(index=False)+'<h2>Robustness (development only)</h2>'+pd.DataFrame(sensitivity).to_html(index=False)+'<h2>Charts</h2>'+imgs+'<p>CSV records: parameter-grid.csv, robustness.csv, performance.csv, exit-reasons.csv; development/ and final-validation/ contain candidates, p-values, allocations, equity, signals and trade logs. frozen-settings.json records the choice before final simulation. Equity curves include idle cash. All results are historical simulations.</p></html>'
    (output/'report.html').write_text(html)
    print(metrics[['Period','Total_Return','Sharpe_Ratio','Num_Trades']].to_string(index=False),flush=True)
    print('RESULTS',output,flush=True)

if __name__=='__main__':main()
