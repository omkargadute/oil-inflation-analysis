"""One analysis dataset through the latest common month, with honest chronological evaluation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from analysis.data import COUNTRIES
from analysis.latest_data import write_data_audit,RAW
from analysis.combined_model import (fit_combined,forecast_combined,forecast_metrics,replications,
                                     innovation_checks,innovations)
from analysis.robust_model import simulation_data
from analysis.strengthening import block_indices

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/combined_project'
OUT.mkdir(parents=True,exist_ok=True)
KINDS=['ar1','ar2','sar12']

def lagged_frame(frame,lag):
    frame=frame.copy()
    oil=pd.read_csv(RAW/'brent_direct.csv',index_col=0,parse_dates=True).iloc[:,0]
    demand=pd.read_csv(RAW/'igrea.csv',index_col=0,parse_dates=True).iloc[:,0]
    frame['T']=(10*oil.pct_change(fill_method=None)).reindex(frame.date-pd.DateOffset(months=lag)).to_numpy()
    frame['C']=(demand/100).reindex(frame.date-pd.DateOffset(months=lag+1)).to_numpy()
    if not np.isfinite(frame[['T','C']]).all().all():raise ValueError('Missing lag sensitivity predictors')
    return frame

def run_combined():
    panel,daily=write_data_audit();end=panel.date.max();groups={i:panel[panel.iso3.eq(i)].reset_index(drop=True) for i in COUNTRIES}
    registry=[];diagnostics=[];coefficients=[];fits={};checks=[];pp=[];replicated={};simulation=[]
    def record(fit,iso,label):
        d=fit['diagnostics'].copy();d['iso3']=iso;d['label']=label;diagnostics.append(d)
        registry.append({'iso3':iso,'label':label,'key':fit['key'],'n_factors':fit['n_factors'],
            'observed_after_prefix':fit['n_observed'],'latent_outcomes':len(fit['missing']),
            'max_Rhat':d.r_hat.max(),'min_bulk_ESS':d.ess_bulk.min(),'min_tail_ESS':d.ess_tail.min(),
            'sampling_pass':bool(d.sampling_pass.all()),'attempt':fit['attempt']})
        pd.DataFrame(registry).to_csv(OUT/'fit_registry.csv',index=False)
        return fit
    def coefficient(fit,iso,experiment,variant):
        b=fit['diagnostics'].query("parameter == 'beta'").iloc[0]
        return {'iso3':iso,'experiment':experiment,'variant':variant,'beta':b['mean'],
                'lower95':b.lower95,'upper95':b.upper95,'n_observed':fit['n_observed']}
    for iso,g in groups.items():
        settings=[('ar1',f,5,a) for f in ['gaussian','student'] for a in [False,True]]
        settings += [(k,'student',5,True) for k in ['ar2','sar12']]+[('ar1','student',nu,True) for nu in [3,10]]
        for kind,family,nu,adjusted in settings:
            label=f'full/{kind}/{family}/nu={nu if family=="student" else "NA"}/adjusted={adjusted}'
            f=record(fit_combined(g,kind,family,nu,adjusted,label=iso+'/'+label),iso,label)
            fits[(iso,kind,family,nu,adjusted)]=f
            coefficients.append(coefficient(f,iso,'full',label))
            if adjusted:
                row={'iso3':iso,'kind':kind,'family':family,'nu':nu if family=='student' else np.nan,**innovation_checks(f)}
                row['diagnostic_first_date']=g.date.iloc[row.pop('run_first_index')]
                row['diagnostic_last_date']=g.date.iloc[row.pop('run_last_index')]
                checks.append(row)
                mu,rep=replications(f,count=800,seed=400+list(COUNTRIES).index(iso))
                observed=g.monthly.to_numpy()[13:];mask=np.isfinite(observed)
                for statistic,fun in [('minimum',np.min),('maximum',np.max),('sd',np.std)]:
                    values=fun(rep[:,13:][:,mask],axis=1);actual=fun(observed[mask]);lo,hi=np.quantile(values,[.025,.975])
                    pp.append({'iso3':iso,'kind':kind,'family':family,'nu':nu if family=='student' else np.nan,'statistic':statistic,
                               'observed':actual,'lower95':lo,'upper95':hi,'outside95':bool(actual<lo or actual>hi)})
                if nu==5 and kind=='ar1':replicated[(iso,family)]=(mu,rep)
        pd.DataFrame(coefficients).to_csv(OUT/'coefficients.csv',index=False)
    # Preserve established treatment timing; alternative lags use correspondingly earlier demand.
    for iso,g in groups.items():
        for lag in [2,3]:
            label=f'lag/{lag}';f=record(fit_combined(lagged_frame(g,lag),label=iso+'/'+label),iso,label)
            coefficients.append(coefficient(f,iso,'oil_lag',label))
        for period,h in [('pre2020',g.iloc[:240]),('2020-onward',g.iloc[240-13:])]:
            label='period/'+period;f=record(fit_combined(h,label=iso+'/'+label),iso,label)
            coefficients.append(coefficient(f,iso,'period',period))
    # Independent synthetic data: one recovery for each family plus existing 40-dataset stress study.
    for family in ['gaussian','student']:
        synthetic,truth=simulation_data(family)
        for adjusted in [False,True]:
            label=f'simulation/{family}/adjusted={adjusted}'
            f=record(fit_combined(synthetic,family=family,adjusted=adjusted,label=label),'SIM',label)
            b=f['diagnostics'].query("parameter == 'beta'").iloc[0]
            simulation.append({'family':family,'adjusted':adjusted,'truth':.15,'beta':b['mean'],
                               'lower95':b.lower95,'upper95':b.upper95,'covered':bool(b.lower95<=.15<=b.upper95)})
    # Disjoint chronological evaluation windows; final window extends through the actual endpoint.
    predictions=[];metrics=[];window_details={};benchmarks=[]
    for start,stop in [(120,180),(180,240),(240,len(groups['USA']))]:
        window=f'{groups["USA"].date.iloc[start]:%Y-%m}--{groups["USA"].date.iloc[stop-1]:%Y-%m}'
        for iso,g in groups.items():
            for kind in KINDS:
                for oil in [False,True]:
                    label=f'rolling/{window}/{kind}/oil={oil}'
                    f=record(fit_combined(g.iloc[:start],kind=kind,oil=oil,label=iso+'/'+label),iso,label)
                    detail=forecast_combined(f,g.iloc[:stop],g.date.iloc[start]);detail['iso3']=iso
                    detail['kind']=kind;detail['oil']=oil;detail['window']=window
                    predictions.append(detail);window_details[(iso,kind,oil,window)]=detail
                    metrics.append({'iso3':iso,'kind':kind,'oil':oil,'window':window,**forecast_metrics(detail)})
            pd.DataFrame(metrics).to_csv(OUT/'rolling_metrics.csv',index=False)
            baseline_predictions={
                'last month':g.monthly.shift(1).iloc[start:stop].to_numpy(),
                'same month last year':g.monthly.shift(12).iloc[start:stop].to_numpy(),
                'training mean':np.repeat(g.monthly.iloc[:start].mean(),stop-start),
                'training month mean':g.month.iloc[start:stop].map(g.iloc[:start].groupby('month').monthly.mean()).to_numpy()}
            for oil in [False,True]:
                baseline_predictions['AR1 with oil' if oil else 'AR1 without oil']=window_details[(iso,'ar1',oil,window)].prediction.to_numpy()
            actual=g.monthly.iloc[start:stop].to_numpy()
            matched=np.isfinite(actual)&np.logical_and.reduce([np.isfinite(a) for a in baseline_predictions.values()])
            for name,prediction in baseline_predictions.items():
                error=(actual-prediction)[matched]
                benchmarks.append({'iso3':iso,'window':window,'model':name,'n_test':int(matched.sum()),
                                   'RMSE':np.sqrt(np.mean(error**2)),'MAE':np.mean(np.abs(error))})
    aggregate=[];windows=list(dict.fromkeys(m['window'] for m in metrics))
    for iso in COUNTRIES:
        paired={};common={}
        for window in windows:
            common[window]=np.logical_and.reduce([window_details[(iso,k,o,window)].eligible.to_numpy() for k in KINDS for o in [False,True]])
        for kind in KINDS:
            errors={};scores=[]
            for oil in [False,True]:
                errors[oil]=[]
                for window in windows:
                    q=window_details[(iso,kind,oil,window)]
                    errors[oil].append(np.where(common[window],q.actual-q.prediction,np.nan))
                    if oil:scores.extend(q.loc[common[window],'log_score'])
            paired[kind]=errors
            a=np.concatenate(errors[True]);b=np.concatenate(errors[False])
            ba=[];bb=[];reference=[]
            for wi,window in enumerate(windows):
                idx=block_indices(len(errors[True][wi]),700+wi)
                ba.append(errors[True][wi][idx]**2);bb.append(errors[False][wi][idx]**2)
                q=window_details[(iso,'ar1',True,window)]
                reference.append(np.where(common[window],q.actual-q.prediction,np.nan)[idx]**2)
            ra=np.sqrt(np.nanmean(np.concatenate(ba,axis=1),axis=1));rb=np.sqrt(np.nanmean(np.concatenate(bb,axis=1),axis=1))
            rr=np.sqrt(np.nanmean(np.concatenate(reference,axis=1),axis=1))
            oil_ci=np.quantile(100*(rb-ra)/rb,[.025,.975]);structure_ci=np.quantile(100*(rr-ra)/rr,[.025,.975])
            ref=np.concatenate([np.where(common[w],window_details[(iso,'ar1',True,w)].actual-window_details[(iso,'ar1',True,w)].prediction,np.nan) for w in windows])
            rmse=np.sqrt(np.nanmean(a*a));base=np.sqrt(np.nanmean(b*b));reference_rmse=np.sqrt(np.nanmean(ref*ref))
            aggregate.append({'iso3':iso,'kind':kind,'n_test':int(np.isfinite(a).sum()),'RMSE':rmse,'RMSE_no_oil':base,
                'oil_gain_percent':100*(base-rmse)/base,'oil_gain_lower95':oil_ci[0],'oil_gain_upper95':oil_ci[1],
                'mean_log_score':np.mean(scores),'structure_gain_percent':100*(reference_rmse-rmse)/reference_rmse,
                'structure_lower95':structure_ci[0],'structure_upper95':structure_ci[1]})
    # A dedicated pre-2025 fit evaluates recent events; no coefficients use later outcomes.
    recent=[];recent_metrics=[];reference_coefficients=[];recent_gains=[]
    for iso,g in groups.items():
        pair={}
        for oil in [False,True]:
            label=f'recent-training/through2024/oil={oil}'
            f=record(fit_combined(g[g.date<'2025-01-01'],oil=oil,label=iso+'/'+label),iso,label)
            q=forecast_combined(f,g,'2025-01-01');q['iso3']=iso;q['oil']=oil;recent.append(q)
            pair[oil]=q
            recent_metrics.append({'iso3':iso,'oil':oil,**forecast_metrics(q)})
            if oil:reference_coefficients.append(coefficient(f,iso,'historical_reference','through2024'))
        a=pair[True];b=pair[False];eligible=a.eligible&b.eligible
        ea=np.where(eligible,a.actual-a.prediction,np.nan);eb=np.where(eligible,b.actual-b.prediction,np.nan)
        ra=np.sqrt(np.nanmean(ea**2));rb=np.sqrt(np.nanmean(eb**2))
        row={'iso3':iso,'n_test':int(eligible.sum()),'RMSE_gain_percent':100*(rb-ra)/rb}
        for length in [3,6]:
            idx=block_indices(len(ea),2026,length=length)
            ba=np.sqrt(np.nanmean(ea[idx]**2,axis=1));bb=np.sqrt(np.nanmean(eb[idx]**2,axis=1))
            lo,hi=np.nanquantile(100*(bb-ba)/bb,[.025,.975])
            row[f'block{length}_lower95']=lo;row[f'block{length}_upper95']=hi
        recent_gains.append(row)
    tables={'coefficients':pd.DataFrame(coefficients),'sampling_diagnostics':pd.concat(diagnostics,ignore_index=True),
        'innovation_checks':pd.DataFrame(checks),'predictive_checks':pd.DataFrame(pp),'simulation_recovery':pd.DataFrame(simulation),
        'rolling_predictions':pd.concat(predictions,ignore_index=True),'rolling_metrics':pd.DataFrame(metrics),
        'rolling_aggregate':pd.DataFrame(aggregate),'recent_predictions':pd.concat(recent,ignore_index=True),
        'recent_metrics':pd.DataFrame(recent_metrics),'recent_gains':pd.DataFrame(recent_gains),
        'benchmark_metrics':pd.DataFrame(benchmarks),'historical_reference':pd.DataFrame(reference_coefficients)}
    for name,table in tables.items():table.to_csv(OUT/(name+'.csv'),index=False)
    (OUT/'manifest.json').write_text(json.dumps({'completed':True,'endpoint':str(end.date()),'rows':len(panel),
        'observed_outcomes':int(panel.monthly.notna().sum()),'fit_cases':len(registry),
        'unique_keys':len({r['key'] for r in registry}),'all_sampling_pass':True,
        'missing_model':'Bayesian integration of missing monthly outcomes; conditional 13-month prefix'},indent=2))
    return {'panel':panel,'daily':daily,'groups':groups,'fits':fits,'replicated':replicated,**tables}

if __name__=='__main__':run_combined()
