"""Extend the study to current releases, with an explicit US CPI gap restart."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.data import COUNTRIES
from analysis.latest_data import write_data_audit,OUT,RAW
from analysis.latest_model import fit_latest,forecast_latest
from analysis.strengthening import block_indices

def run_latest():
    panel,daily=write_data_audit();end=panel.date.max();coefficients=[];diagnostics=[];details=[];metrics=[];gains=[]
    def record(fit,iso,sample):
        d=fit['diagnostics'].copy();d['iso3']=iso;d['sample']=sample;diagnostics.append(d)
        if fit['oil']:
            b=d[d.parameter.eq('beta')].iloc[0]
            coefficients.append({'iso3':iso,'sample':sample,'beta':b['mean'],'lower95':b.lower95,
                'upper95':b.upper95,'n_likelihood':fit['n_eff'],'key':fit['key']})
    for iso in COUNTRIES:
        g=panel[panel.iso3.eq(iso)].reset_index(drop=True);train=g[g.date<'2025-01-01']
        full=fit_latest(g,label=iso+'/through-current');record(full,iso,'through-'+end.strftime('%Y-%m'))
        forecasts={}
        for oil in [False,True]:
            fit=fit_latest(train,oil=oil,label=iso+'/through2024/oil='+str(oil))
            record(fit,iso,'through-2024/oil='+str(oil))
            d=forecast_latest(fit,g);d['iso3']=iso;d['oil']=oil;details.append(d);forecasts[oil]=d
            for period,start in [('2025-onward','2025-01-01'),('2026','2026-01-01')]:
                q=d[d.date.ge(start)&d.eligible];errors=q.actual-q.prediction
                metrics.append({'iso3':iso,'oil':oil,'period':period,'n_test':len(q),
                    'RMSE':np.sqrt(np.mean(errors**2)),'MAE':np.mean(np.abs(errors)),
                    'mean_log_score':q.log_score.mean(),'coverage95':np.mean(q.actual.between(q.lower95,q.upper95))})
        for period,start in [('2025-onward','2025-01-01'),('2026','2026-01-01')]:
            a=forecasts[True];b=forecasts[False]
            keep=a.date.ge(start)&a.eligible&b.eligible
            ea=(a.actual-a.prediction)[keep].to_numpy();eb=(b.actual-b.prediction)[keep].to_numpy()
            ra=np.sqrt(np.mean(ea**2));rb=np.sqrt(np.mean(eb**2))
            row={'iso3':iso,'period':period,'n_test':len(ea),'RMSE_with_oil':ra,'RMSE_no_oil':rb,
                 'RMSE_gain_percent':100*(rb-ra)/rb}
            # Resample calendar blocks, preserving missing-month exclusions rather than compressing time.
            calendar_keep=a.date.ge(start).to_numpy()
            full_ea=(a.actual-a.prediction).where(a.eligible).to_numpy()[calendar_keep]
            full_eb=(b.actual-b.prediction).where(b.eligible).to_numpy()[calendar_keep]
            for length in [3,6]:
                idx=block_indices(len(full_ea),2026,replications=2000,length=length)
                ba=np.sqrt(np.nanmean(full_ea[idx]**2,axis=1));bb=np.sqrt(np.nanmean(full_eb[idx]**2,axis=1))
                ci=np.nanquantile(100*(bb-ba)/bb,[.025,.975])
                row[f'block{length}_lower95'],row[f'block{length}_upper95']=ci
            gains.append(row)
    coefficient=pd.DataFrame(coefficients);diagnostic=pd.concat(diagnostics,ignore_index=True)
    predictions=pd.concat(details,ignore_index=True);metric=pd.DataFrame(metrics);gain=pd.DataFrame(gains)
    for name,table in [('coefficients',coefficient),('sampling_diagnostics',diagnostic),
                        ('forecast_predictions',predictions),('forecast_metrics',metric),('oil_forecast_gains',gain)]:
        table.to_csv(OUT/(name+'.csv'),index=False)
    monthly=panel[panel.iso3.eq('USA')][['date','oil_usd','oil_pct','global_activity']].copy()
    monthly[monthly.date.ge('2025-01-01')].to_csv(OUT/'recent_global_series.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(12,7))
    d=daily[daily.date.ge('2025-01-01')];m=monthly[monthly.date.ge('2025-01-01')]
    axes[0].plot(d.date,d.oil_usd,alpha=.55,label='Daily Brent (context only)')
    axes[0].plot(m.date+pd.offsets.MonthEnd(0),m.oil_usd,'o-',label='Complete monthly average (dated month-end)')
    axes[0].set_ylabel('USD/barrel');axes[0].legend()
    for iso,g in panel[panel.date.ge('2025-01-01')].groupby('iso3'):
        axes[1].plot(g.date,g.monthly,label=COUNTRIES[iso])
    axes[1].set_ylabel('Unadjusted monthly CPI change (%)');axes[1].legend(ncol=4,fontsize=8)
    plt.tight_layout();plt.savefig(OUT/'recent_history.png',bbox_inches='tight');plt.close()
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for ax,iso in zip(axes.flat,COUNTRIES):
        q=predictions[predictions.iso3.eq(iso)&predictions.oil]
        ax.plot(q.date,q.actual,color='black',label='Observed')
        ax.plot(q.date,q.prediction,label='Fixed pre-2025 fit + oil')
        ax.fill_between(q.date,q.lower95,q.upper95,alpha=.2)
        ax.set_title(COUNTRIES[iso]);ax.set_ylabel('Monthly inflation %')
    axes[0,0].legend(fontsize=8);plt.tight_layout();plt.savefig(OUT/'recent_forecasts.png',bbox_inches='tight');plt.close()
    latest_daily=daily.iloc[-1];last_month=monthly[monthly.date.eq(end)].iloc[0]
    lines=['# Current-release extension','',
        f'Common monthly endpoint: **{end:%B %Y}**. Four country calendars contain {len(panel):,} rows '
        f'and {panel.monthly.notna().sum():,} observed monthly inflation outcomes. '
        f'Daily Brent ends {latest_daily.date:%B %d, %Y}; the last observation is ${latest_daily.oil_usd:.2f}/barrel. '
        f'The complete {end:%B %Y} monthly average is ${last_month.oil_usd:.2f}/barrel. '
        'Daily prices are context only and are never substituted for monthly averages.','',
        '## Missing US month and likelihood','',
        'US October 2025 CPI is absent. Calendar reindexing leaves October and November monthly inflation '
        'undefined. No interpolation, forward fill or two-month-as-one-month calculation is used. '
        'The conditional AR1 model includes only observed consecutive residual pairs after the original '
        '13-month prefix. The first observed residual after each gap is conditioned on, restarting the '
        'likelihood; all ten 2026 US outcomes can still contribute. This is a segmented conditional likelihood, '
        'not an exact stationary initial-state or missing-state marginal likelihood.','',
        'BLS confirms the missing October release followed the shutdown: '
        '[official explanation](https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm).','',
        '## Matched current-vintage coefficient comparison','']
    for iso,name in COUNTRIES.items():
        q=coefficient[coefficient.iso3.eq(iso)].set_index('sample')
        a=q.loc['through-2024/oil=True'];b=q.loc['through-'+end.strftime('%Y-%m')]
        lines.append(f'- **{name}:** through-2024 beta {a.beta:.4f} [{a.lower95:.4f}, {a.upper95:.4f}]; '
            f'extended beta {b.beta:.4f} [{b.lower95:.4f}, {b.upper95:.4f}]; '
            f'{int(b.n_likelihood)} observed AR1 transitions in the extended likelihood.')
    lines+=['','## Fixed-training recent predictions','',
        'Coefficients use only outcomes through December 2024. Predictions condition on the previous observed '
        'monthly residual. They use revised data, not real-time vintages; short recent windows cannot establish '
        'a stable crisis regime or causal supply-shock effect.','']
    for r in gain[gain.period.eq('2025-onward')].itertuples():
        lines.append(f'- **{COUNTRIES[r.iso3]}:** {r.n_test} scored months; oil RMSE gain {r.RMSE_gain_percent:.2f}%; '
            f'3-month block interval [{r.block3_lower95:.2f}%, {r.block3_upper95:.2f}%], '
            f'6-month block interval [{r.block6_lower95:.2f}%, {r.block6_upper95:.2f}%].')
    lines+=['','## Interpretation','',
        'The nominal-price model combines demand, supply, exchange-rate and policy influences. '
        'Observed price surges or reversals cannot independently identify a geopolitical causal effect. '
        'The newest monthly oil movement can affect next-month inflation under our lag-one estimand, so '
        'its full response may not yet be observed. The sample contains only ten 2026 months. '
        'Keep the original study and strengthening results as reference; use this extension to assess '
        'temporal stability and recent forecast behavior. Historical revisions are audited separately.']
    (OUT/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps({'completed':True,'common_endpoint':str(end.date()),
        'country_month_rows':len(panel),'observed_monthly_outcomes':int(panel.monthly.notna().sum()),
        'retained_fits':12,'all_sampling_pass':bool(diagnostic.sampling_pass.all()),
        'raw_manifest_sha256':__import__('hashlib').sha256((RAW/'manifest.json').read_bytes()).hexdigest()},indent=2))
    return {'coefficients':coefficient,'forecast_metrics':metric,'oil_forecast_gains':gain,
            'availability':pd.read_csv(OUT/'availability.csv'),'historical_revisions':pd.read_csv(OUT/'historical_revisions.csv')}

if __name__=='__main__':run_latest()
