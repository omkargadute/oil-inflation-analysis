# %% [markdown]
# # Oil-price changes and inflation: US, China, Japan and Canada
# Monthly 2000-2024, separate country models with the same three substantive variables: prior Brent
# change T, monthly CPI inflation Y, prior global activity C. Candidate DAG: C -> T -> Y and C -> Y.
# This is an assumption-dependent observational analysis, not identification of oil supply shocks.
#
# Every country uses **unadjusted national all-items CPI** with country-specific calendar-month effects.
# We do not mix officially adjusted US CPI with unadjusted Asian CPI. Fixed seasonal patterns do not
# capture all moving holidays (including Lunar New Year), changing seasonality, taxes or administered prices.
# Seasonal parameters are statistical structure, not additional claimed causal confounders; explain this
# richer temporal model under the course's three-variable requirement. No pooled independence or country
# ranking test is assumed. Sources: [OECD](https://data-explorer.oecd.org/),
# [Brent](https://fred.stlouisfed.org/series/MCOILBRENTEU), [IGREA](https://www.dallasfed.org/research/igrea).
# %%
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT=Path.cwd()
if not (ROOT/'data/raw').exists(): ROOT=ROOT.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from analysis.model import SEASON, design, log_target, sample_model, summarize
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
COUNTRIES={'USA':'United States','CHN':'China','JPN':'Japan','CAN':'Canada'}
FLOWS={'USA':'DSD_PRICES@DF_PRICES_ALL','CHN':'DSD_PRICES@DF_PRICES_ALL',
       'JPN':'DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL','CAN':'DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL'}
plt.rcParams.update({'figure.figsize':(10,4),'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False})
def plot_save(name):
    plt.tight_layout();plt.savefig(OUT/(name+'.png'),bbox_inches='tight');plt.show();plt.close()

frames=[]
for manifest_name in ['manifest.json','country_manifest.json']:
    for item in json.loads((ROOT/'data/raw'/manifest_name).read_text()):
        path=ROOT/'data/raw'/item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],item['file']
        if manifest_name=='country_manifest.json':
            f=pd.read_csv(path);f['flow']=item['flow'];frames.append(f)
cpi=pd.concat(frames,ignore_index=True)
assert not cpi.duplicated(['REF_AREA','flow','ADJUSTMENT','TIME_PERIOD']).any()
oil=pd.read_csv(ROOT/'data/raw/brent_direct.csv',parse_dates=[0]);oil.columns=['date','oil_usd']
dem=pd.read_csv(ROOT/'data/raw/igrea.csv',parse_dates=[0]);dem.columns=['date','global_activity']
assert not oil.date.duplicated().any() and not dem.date.duplicated().any()
base=pd.DataFrame({'date':pd.date_range('1999-01-01','2024-12-01',freq='MS')})
base=base.merge(oil,on='date',how='left',validate='one_to_one').merge(dem,on='date',how='left',validate='one_to_one')
base['oil_pct']=base.oil_usd.pct_change(fill_method=None)*100
base['T']=base.oil_pct.shift(1)/10;base['C']=base.global_activity.shift(2)/100
groups={};panels=[]
for iso,name in COUNTRIES.items():
    f=cpi[(cpi.REF_AREA==iso)&(cpi.flow==FLOWS[iso])&(cpi.ADJUSTMENT=='N')].copy()
    assert f.FREQ.eq('M').all() and f.UNIT_MEASURE.eq('IX').all() and f.MEASURE.eq('CPI').all()
    f['date']=pd.to_datetime(f.TIME_PERIOD)
    d=base.merge(f[['date','OBS_VALUE']].rename(columns={'OBS_VALUE':'cpi'}),on='date',how='left',validate='one_to_one')
    d['monthly']=d.cpi.pct_change(fill_method=None)*100;d['yearly']=d.cpi.pct_change(12,fill_method=None)*100
    d['last_monthly']=d.monthly.shift(1);d['last_year_same_month']=d.monthly.shift(12)
    d['month']=d.date.dt.month;d['country']=name;d['iso3']=iso;d['adjustment']='N';d['flow']=FLOWS[iso]
    d=d[d.date.ge('2000-01-01')].reset_index(drop=True)
    assert len(d)==300 and d[['monthly','yearly','T','C','last_monthly']].notna().all().all()
    groups[iso]=d;panels.append(d)
panel=pd.concat(panels,ignore_index=True)
assert len(panel)==1200 and not panel.duplicated(['iso3','date']).any()
panel.to_csv(ROOT/'data/processed/multicountry_monthly.csv',index=False)
description=panel.groupby('country')[['monthly','yearly','T','C']].describe()
description.to_csv(OUT/'descriptive_statistics.csv')
print('Four complete calendars, 300 months each; no imputation or series splicing.')
print(description.round(3).to_string())
print('Shared treatment-demand correlation:',round(groups['USA']['T'].corr(groups['USA'].C),3))
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
for ax,(iso,d) in zip(axes.flat,groups.items()):
    ax.plot(d.date,d.monthly);ax.set_title(COUNTRIES[iso]);ax.set_ylabel('Unadjusted monthly CPI %')
plot_save('country_inflation')
# %% [markdown]
# ## Same estimand, priors and seasonal structure in each country
# T = oil percentage change in month t-1 / 10; Y = unadjusted monthly CPI inflation in t;
# C = IGREA in t-2 / 100. Beta is monthly inflation percentage points per 10-percentage-point oil change.
#
# Y_t = alpha + beta*T_t + gamma*C_t + s_month(t) + u_t; u_t = phi*u_(t-1) + epsilon_t.
# epsilon ~ Normal(0,sigma^2); initial u_0 ~ Normal(0,sigma^2/(1-phi^2)); |phi|<1.
# Eleven orthonormal Helmert contrasts give seasonal effects summing to zero. Both adjusted and unadjusted
# models retain seasonality; the unadjusted comparison omits C only. Independent working priors:
# alpha~N(0,.5^2), beta~N(0,.3^2), gamma~N(0,.5^2), each seasonal contrast~N(0,.5^2),
# sigma~HalfNormal(.5), phi~Uniform(-1,1). Prior scale justification is provisional, not completed by a
# predictive check alone. All countries use the same priors; no tuning for significant coefficients.
#
# Four Metropolis-within-Gibbs chains: exact Gaussian coefficient draws, transformed sigma/phi proposals
# with Jacobians; 2,000 warmup then 6,000 retained iterations per chain. Adaptation stops after warmup.
# The numerical implementation is in analysis/model.py and is included in the generated notebook.
# %%
assert np.allclose(SEASON.sum(axis=0),0) and np.allclose(SEASON.T@SEASON,np.eye(11))
for iso,d in groups.items():
    X,names,scales=design(d);X=X[:12];y=d.monthly.to_numpy()[:12];p=X.shape[1]
    def independent(z):
        sig=np.exp(z[p]);ph=np.tanh(z[p+1]);i=np.arange(len(y))
        cov=sig**2/(1-ph**2)*ph**np.abs(i[:,None]-i[None,:])
        return (multivariate_normal.logpdf(y,mean=X@z[:p],cov=cov)-.5*np.sum((z[:p]/scales)**2)
                -.5*(sig/.5)**2+z[p]+np.log1p(-ph**2))
    z1=np.r_[np.zeros(p),np.log(.25),np.arctanh(.3)];z2=np.r_[np.ones(p)*.03,np.log(.4),np.arctanh(-.2)]
    assert np.isclose(log_target(z1,X,y,scales)-log_target(z2,X,y,scales),independent(z1)-independent(z2),atol=1e-8)
print('All four AR likelihoods/Jacobians agree with independent covariance calculations.')
rng=np.random.default_rng(50);prior_rows=[];fig,axes=plt.subplots(2,2,figsize=(11,6))
for ax,(iso,d) in zip(axes.flat,groups.items()):
    X,names,scales=design(d);count=1500;b=rng.normal(size=(count,len(names)))*scales
    sig=np.abs(rng.normal(0,.5,count));ph=rng.uniform(-1,1,count);res=np.empty((count,len(d)))
    res[:,0]=rng.normal(size=count)*sig/np.sqrt(1-ph**2)
    for t in range(1,len(d)):res[:,t]=ph*res[:,t-1]+rng.normal(size=count)*sig
    yp=b@X.T+res;lo,hi=np.quantile(yp,[.025,.975]);prior_rows.append({'iso3':iso,'lower95':lo,'upper95':hi})
    ax.hist(yp.ravel(),bins=80,range=(-5,5),density=True);ax.set_title(COUNTRIES[iso]);ax.set_xlabel('Monthly inflation %')
plot_save('prior_predictive');pd.DataFrame(prior_rows).to_csv(OUT/'prior_predictive.csv',index=False)
# %% [markdown]
# ## Country fits and matched confounder comparisons
# ArviZ rank R-hat, bulk/tail ESS, MCSE and traces assess computation. A failed diagnostic is flagged.
# Conditional country intervals are not formal country-difference tests: countries share shocks and errors.
# %%
fits={};tables=[];coefficient_rows=[];acceptance_rows=[]
for i,(iso,d) in enumerate(groups.items()):
    fits[iso]={}
    for adjusted in [False,True]:
        label='adjusted' if adjusted else 'unadjusted';X,names,scales=design(d,adjusted)
        chains,accept=sample_model(X,d.monthly.to_numpy(),scales,seed=22+i*1000+int(adjusted)*50)
        fits[iso][label]=chains;table=summarize(chains,names);table.insert(0,'model',label);table.insert(0,'iso3',iso);tables.append(table)
        beta=table[table.parameter.eq('beta')].iloc[0]
        coefficient_rows.append({'country':COUNTRIES[iso],'iso3':iso,'model':label,'beta':beta['mean'],
                                 'lower95':beta.lower95,'upper95':beta.upper95,'sampling_pass':bool(table.sampling_pass.all())})
        for c,rates in enumerate(accept):acceptance_rows.append({'iso3':iso,'model':label,'chain':c+1,'sigma_accept':rates[0],'phi_accept':rates[1]})
        print(iso,label,'beta',round(beta['mean'],4),'sampling pass',bool(table.sampling_pass.all()),flush=True)
diagnostics=pd.concat(tables,ignore_index=True);coefficients=pd.DataFrame(coefficient_rows)
diagnostics.to_csv(OUT/'posterior_diagnostics.csv',index=False);coefficients.to_csv(OUT/'country_coefficients.csv',index=False)
pd.DataFrame(acceptance_rows).to_csv(OUT/'sampler_acceptance.csv',index=False)
np.savez_compressed(OUT/'posterior_draws.npz',**{iso+'_'+label:a for iso,f in fits.items() for label,a in f.items()})
print(coefficients.round(4).to_string(index=False))
fig,axes=plt.subplots(4,2,figsize=(12,10))
for row,(iso,f) in enumerate(fits.items()):
    for c in range(4):
        axes[row,0].plot(f['adjusted'][c,:,1],lw=.3,alpha=.6);axes[row,1].plot(f['adjusted'][c,:,-1],lw=.3,alpha=.6)
    axes[row,0].set_ylabel(iso+' beta');axes[row,1].set_ylabel(iso+' phi')
plot_save('mcmc_traces')
fig,ax=plt.subplots(figsize=(9,5))
for j,label in enumerate(['unadjusted','adjusted']):
    q=coefficients[coefficients.model.eq(label)].set_index('iso3').loc[list(COUNTRIES)]
    ax.errorbar(q.beta,np.arange(4)+(j-.5)*.15,xerr=np.array([q.beta-q.lower95,q.upper95-q.beta]),fmt='o',label=label)
ax.set_yticks(np.arange(4),list(COUNTRIES.values()));ax.axvline(0,color='grey',lw=.8);ax.legend()
ax.set_xlabel('Monthly inflation percentage points per 10 pp oil change');ax.set_title('Country associations; not a country-difference test')
plot_save('country_coefficients')
# %% [markdown]
# ## Separate simulation validation: seasonality, confounding and dependent errors
# True beta=.15, gamma=.25, sigma=.15, phi=.4. Synthetic observations never enter real-data fits.
# Recovery validates computation under known conditions, not the real-world DAG.
# %%
rng=np.random.default_rng(81);simdata=groups['USA'][['date','month']].copy();N=len(simdata);C=np.zeros(N)
for i in range(1,N):C[i]=.6*C[i-1]+rng.normal(0,.5)
simdata['C']=C;simdata['T']=.8*C+rng.normal(0,.7,N);Xs,names,scales=design(simdata)
truth=np.r_[.2,.15,.25,np.linspace(-.15,.15,11)];u=np.empty(N);u[0]=rng.normal(0,.15/np.sqrt(1-.4**2))
for i in range(1,N):u[i]=.4*u[i-1]+rng.normal(0,.15)
ys=Xs@truth+u;simulation=[]
for adjusted in [False,True]:
    X,names,scales=design(simdata,adjusted);chain,accept=sample_model(X,ys,scales,seed=171+int(adjusted)*100)
    table=summarize(chain,names);table.insert(0,'model','adjusted' if adjusted else 'unadjusted')
    if adjusted:
        table['truth']=np.r_[truth,.15,.4];table['truth_in_interval']=(table.truth>=table.lower95)&(table.truth<=table.upper95)
    simulation.append(table)
simulation=pd.concat(simulation,ignore_index=True);simulation.to_csv(OUT/'simulation_recovery.csv',index=False)
print(simulation[simulation.parameter.isin(['beta','gamma','sigma','phi'])].round(4).to_string(index=False))
# %% [markdown]
# ## Posterior prediction, residual checks and robustness for every country
# Pointwise predictive intervals include seasonal mean uncertainty and dependent residuals. Plug-in
# innovation tests and extremes expose misspecification; satisfactory sampling is not model validation.
# HAC sensitivity regressions retain calendar effects and are not identical to Bayesian AR estimates.
# %%
predictive=[];innovations=[];robustness=[];prior_sensitivity=[];season_rows=[];fig,axes=plt.subplots(2,2,figsize=(13,8))
for i,(ax,(iso,d)) in enumerate(zip(axes.flat,groups.items())):
    X,names,scales=design(d);flat=fits[iso]['adjusted'].reshape(-1,len(names)+2)
    rng=np.random.default_rng(70+i);draw=flat[rng.choice(len(flat),1500,replace=False)];mu=draw[:,:len(names)]@X.T;r=np.empty_like(mu)
    r[:,0]=rng.normal(size=len(draw))*draw[:,-2]/np.sqrt(1-draw[:,-1]**2)
    for t in range(1,len(d)):r[:,t]=draw[:,-1]*r[:,t-1]+rng.normal(size=len(draw))*draw[:,-2]
    rep=mu+r;mean=flat.mean(axis=0);res=d.monthly.to_numpy()-X@mean[:-2];innov=res[1:]-mean[-1]*res[:-1]
    lb=acorr_ljungbox(innov,lags=[6,12],return_df=True).rename_axis('lag').reset_index();lb['iso3']=iso;innovations.append(lb)
    for name,fun in [('lag1',lambda a:pd.Series(a).autocorr()),('maximum',np.max),('minimum',np.min),('std',np.std)]:
        v=np.array([fun(a) for a in rep]);obs=fun(d.monthly.to_numpy());lo,hi=np.quantile(v,[.025,.975])
        predictive.append({'iso3':iso,'statistic':name,'observed':obs,'rep_lower95':lo,'rep_upper95':hi,'fraction_rep_ge_observed':np.mean(v>=obs)})
    ax.fill_between(d.date,*np.quantile(rep,[.025,.975],axis=0),alpha=.15,label='95% predictive')
    ax.fill_between(d.date,*np.quantile(mu,[.025,.975],axis=0),alpha=.4,label='95% mean')
    ax.plot(d.date,mu.mean(axis=0),lw=.8,label='Seasonal regression mean');ax.plot(d.date,d.monthly,color='black',lw=.6,label='Observed')
    ax.set_title(COUNTRIES[iso]);ax.set_ylabel('Monthly CPI %')
    for month in range(1,13):
        s=flat[:,3:14]@SEASON[month-1];season_rows.append({'iso3':iso,'month':month,'mean':s.mean(),'lower95':np.quantile(s,.025),'upper95':np.quantile(s,.975)})
    for label,g in [('all',d),('2000-2019',d[d.date<'2020-01-01']),('2020-2024',d[d.date>='2020-01-01']),('exclude 2020-2022',d[~d.date.dt.year.between(2020,2022)]),('exclude oil changes >30%',d[d['T'].abs()<=3])]:
        fit=sm.OLS(g.monthly,design(g)[0]).fit(cov_type='HAC',cov_kwds={'maxlags':12});ci=np.asarray(fit.conf_int())[1]
        robustness.append({'iso3':iso,'sample':label,'n':len(g),'beta':np.asarray(fit.params)[1],'HAC_low':ci[0],'HAC_high':ci[1]})
    for scale in [.15,.3,.6]:
        lw=-.5*(flat[:,1]/scale)**2+.5*(flat[:,1]/.3)**2;w=np.exp(lw-lw.max());w/=w.sum();idx=np.argsort(flat[:,1])
        q=np.interp([.025,.975],np.cumsum(w[idx]),flat[idx,1]);prior_sensitivity.append({'iso3':iso,'beta_prior_sd':scale,'beta_mean':w@flat[:,1],'lower95':q[0],'upper95':q[1],'importance_ESS':1/(w@w)})
axes[0,0].legend(fontsize=7,ncol=2);plot_save('posterior_predictive')
pd.DataFrame(predictive).to_csv(OUT/'predictive_checks.csv',index=False)
pd.concat(innovations,ignore_index=True).to_csv(OUT/'innovation_ljung_box.csv',index=False)
pd.DataFrame(robustness).to_csv(OUT/'robustness.csv',index=False)
pd.DataFrame(prior_sensitivity).to_csv(OUT/'prior_sensitivity.csv',index=False)
pd.DataFrame(season_rows).to_csv(OUT/'seasonal_effects.csv',index=False)
print(pd.concat(innovations,ignore_index=True).round(4).to_string(index=False))
# %% [markdown]
# ## Country-specific chronological predictions
# Train once on 2000-2019, evaluate one-step predictions over 2020-2024 using observed previous residuals.
# Seasonal coefficients are trained on training data only. Compare persistence, training calendar-month
# means and last year's same-month inflation. Sources are revised; release delays are not reconstructed.
# This is an exploratory retrospective comparison, not an untouched real-time forecasting experiment.
# %%
metrics=[];training=[];orders=[];fig,axes=plt.subplots(2,2,figsize=(13,7))
for i,(ax,(iso,d)) in enumerate(zip(axes.flat,groups.items())):
    train=d[d.date<'2020-01-01'];test=d[d.date>='2020-01-01'];X,names,scales=design(train)
    chain,accept=sample_model(X,train.monthly.to_numpy(),scales,seed=122+i*1000)
    dt=summarize(chain,names);dt['iso3']=iso;training.append(dt);bt=chain.mean(axis=(0,1))
    allmu=design(d)[0]@bt[:-2];allr=d.monthly.to_numpy()-allmu;conditional=allmu[240:]+bt[-1]*allr[239:-1]
    preds={'training mean':np.repeat(train.monthly.mean(),len(test)),'last-month inflation':test.last_monthly.to_numpy(),
           'training calendar-month mean':test.month.map(train.groupby('month').monthly.mean()).to_numpy(),
           'same month last year':test.last_year_same_month.to_numpy(),'Bayesian seasonal adjusted AR1':conditional,
           'seasonal adjusted independent errors':design(test)[0]@np.asarray(sm.OLS(train.monthly,X).fit().params)}
    for label,pred in preds.items():
        err=test.monthly.to_numpy()-pred;metrics.append({'iso3':iso,'model':label,'train_n':len(train),'test_n':len(test),'MAE':np.abs(err).mean(),'RMSE':np.sqrt(np.mean(err**2))})
    ax.plot(test.date,test.monthly,color='black',lw=.8,label='Observed');ax.plot(test.date,conditional,label='Seasonal AR1')
    ax.plot(test.date,preds['training calendar-month mean'],alpha=.6,label='Training-month mean');ax.set_title(COUNTRIES[iso])
    for order in [1,2,3]:
        fit=SARIMAX(d.monthly,exog=design(d)[0],trend='n',order=(order,0,0)).fit(disp=False,maxiter=200)
        orders.append({'iso3':iso,'AR_order':order,'AIC':fit.aic,'beta':np.asarray(fit.params)[1],
                       'optimizer_converged':bool(fit.mle_retvals.get('converged',False)),
                       'innovation_LB_p12':acorr_ljungbox(fit.resid,lags=[12],return_df=True).lb_pvalue.iloc[0]})
    print('Completed training/holdout:',iso,flush=True)
axes[0,0].legend(fontsize=7);plot_save('holdout_predictions')
metrics=pd.DataFrame(metrics);metrics.to_csv(OUT/'holdout_metrics.csv',index=False)
training=pd.concat(training,ignore_index=True);training.to_csv(OUT/'training_diagnostics.csv',index=False)
pd.DataFrame(orders).to_csv(OUT/'AR_order_sensitivity.csv',index=False)
print(metrics.round(4).to_string(index=False))
# %% [markdown]
# ## Interpretation and remaining work
# Read coefficients, sampling diagnostics and predictive checks together. Country intervals are conditional
# associations, not proof of causal oil effects or country differences. National CPI baskets, exchange
# rates, subsidies, policy regimes, omitted demand paths and shared shocks limit the interpretation.
# Four histories do not create 1,200 independent global shocks. Calendar-month effects may leave moving
# holiday and changing-seasonality effects. Gaussian tail and residual failures must remain visible;
# a Student-t extension is future work. See [the project plan](../docs/PROJECT_PLAN.md).
# %%
failures={'full_data':int((~diagnostics.sampling_pass).sum()),'training':int((~training.sampling_pass).sum()),
          'simulation':int((~simulation.sampling_pass).sum())}
print('Sampling diagnostic failures:',failures)
if any(failures.values()):
    raise RuntimeError('Inspect saved sampling diagnostics before interpreting results.')
print('All chains: R-hat < 1.01, bulk/tail ESS > 400. This validates sampling, not causality.')
(OUT/'FINDINGS.md').write_text('# Four-country model results\n\n```text\n'+coefficients.round(5).to_string(index=False)+'\n```\n\n'
    'Unadjusted monthly CPI with country-specific calendar-month effects, separate Gaussian AR(1) models. '
    'Intervals are conditional associations, not identified causal effects or country-difference tests. '
    'Inspect residual, predictive, robustness and holdout tables for limitations.\n',encoding='utf-8')
lb=pd.read_csv(OUT/'innovation_ljung_box.csv')
pp=pd.read_csv(OUT/'predictive_checks.csv')
with (OUT/'FINDINGS.md').open('a',encoding='utf-8') as report:
    report.write('\n## Model checks\n\nAll full-data, training and simulation sampling checks pass. '
                 'This does not imply adequate residual or tail behavior.\n\n')
    for iso,name in COUNTRIES.items():
        p12=lb[(lb.iso3==iso)&(lb.lag==12)].lb_pvalue.iloc[0]
        extreme=pp[(pp.iso3==iso)&pp.statistic.isin(['minimum','maximum'])]
        misses=extreme[(extreme.observed<extreme.rep_lower95)|(extreme.observed>extreme.rep_upper95)]
        tail=', '.join(misses.statistic) if len(misses) else 'none in these two checks'
        report.write(f'- {name}: 12-lag innovation p={p12:.4f}; extreme-statistic misses: {tail}.\n')
    report.write('\n## Retrospective holdout RMSE, 2020–2024\n\n')
    for iso,name in COUNTRIES.items():
        m=metrics[metrics.iso3.eq(iso)]
        model=m[m.model.eq('Bayesian seasonal adjusted AR1')].RMSE.iloc[0]
        simple=m[m.model.isin(['training mean','last-month inflation','training calendar-month mean','same month last year'])].sort_values('RMSE').iloc[0]
        report.write(f'- {name}: seasonal demand-adjusted AR1 {model:.4f}; best simple benchmark '
                     f'{simple.RMSE:.4f} ({simple.model}).\n')
    report.write('\nThese are exploratory one-step predictions from revised data with training-only '
                 'seasonal coefficients. They are not real-time vintage forecasts or a controlled '
                 'test of oil information alone: the benchmarks differ in more than the oil predictor. '
                 'No no-oil AR counterfactual forecast model or joint country-difference test is fitted. '
                 'Gaussian model failures and imperfect demand adjustment remain. Student-t errors '
                 'and moving-holiday treatment are future work.\n')
