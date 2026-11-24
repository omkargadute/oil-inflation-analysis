# %% [markdown]
# # Oil-price changes and consumer inflation across four economies
# **Foundations of Data Science — computational report**
#
# Sunidhi Suri (ss21743), Omkar Gadute (org2029), Hriday Badani (hrb9339)
#
# All nine sections below execute against verified local snapshots. Gaussian and Student-t models are
# refitted for this report; earlier exploratory CSV estimates are not substituted. Repeated execution can
# reuse content-addressed fits keyed to data, model source, priors, seeds and sampler settings.
#
# ## 1. Research question and estimand
# What is the effect of a 10-percentage-point higher monthly Brent oil-price change on next-month
# consumer inflation in the US, China, Japan and Canada, adjusting for prior global economic activity?
#
# Our target is a country-specific one-month continuous-treatment contrast,
# $\tau_i(a)=\mathbb{E}[Y_{it}(a+1)-Y_{it}(a)]$. Treatment is measured in units of 10 percentage points
# of monthly oil-price change. Under a constant linear effect and the identification assumptions below,
# $\tau_i(a)=\beta_i$. Without those assumptions, the regression coefficient remains an association.
# Country labels index separate analyses; 1,200 country-month rows do not represent 1,200 independent
# oil shocks. No formal between-country causal ranking is assumed.
# %%
from pathlib import Path
import sys
import platform
import importlib.metadata as metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'analysis/data.py').exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.data import COUNTRIES, FLOWS, prepare_panel
from analysis.model import design

OUT = ROOT/'results/final_project'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261124
RNG = np.random.default_rng(SEED)
plt.rcParams.update({'figure.figsize': (10, 4), 'figure.dpi': 110,
                     'axes.spines.top': False, 'axes.spines.right': False})
def save_figure(name):
    plt.tight_layout()
    plt.savefig(OUT/(name+'.png'), bbox_inches='tight')
    plt.show()
    plt.close()

environment = pd.DataFrame({'package': ['Python', 'numpy', 'pandas', 'scipy', 'matplotlib', 'arviz'],
                           'version': [platform.python_version()] + [metadata.version(p) for p in
                                       ['numpy','pandas','scipy','matplotlib','arviz']]})
print('Interpreter:', sys.executable)
display(environment)
# %% [markdown]
# ## 2. Data sources, variables and preparation
# Monthly national all-items CPI indices come from OECD: the COICOP 1999 flow for US/China and the
# COICOP 2018 flow for Japan/Canada. Every country's index is unadjusted; seasonal structure is modeled
# consistently rather than mixing adjusted and unadjusted outcomes. National baskets are not perfectly
# harmonized. Brent is nominal USD/barrel (EIA via FRED); IGREA is the Dallas Fed's shipping-based
# industrial-commodity activity proxy, not GDP growth.
#
# | Role | Transformation | Units |
# |---|---|---|
# | Treatment | $T_t=100(P_{t-1}/P_{t-2}-1)/10$ | 10-percentage-point monthly oil change |
# | Outcome | $Y_{it}=100(I_{it}/I_{i,t-1}-1)$ | Monthly inflation percentage points, not annualized |
# | Potential confounder | $C_t=G_{t-2}/100$ | 100 IGREA index points |
#
# Raw history starts in 1999 so lags precede the January 2000 analysis boundary. All shifts are applied
# to an explicit monthly calendar. No interpolation, cross-flow splicing, or automatic downloading is
# performed. Source dates are reference months, not publication dates; revised snapshots do not support
# a claim of real-time information availability. All hashes and exact URLs are stored in source manifests.
# %%
panel, source_audit = prepare_panel(ROOT)
panel.to_csv(ROOT/'data/processed/multicountry_monthly.csv', index=False)
source_audit.to_csv(OUT/'source_verification.csv', index=False)
coverage = panel.groupby('iso3', sort=False).agg(
    country=('country','first'), first_month=('date','min'), last_month=('date','max'),
    observations=('date','size'), inflation_available=('monthly','count'),
    treatment_available=('T','count'), confound_available=('C','count'))
assert coverage.observations.eq(300).all()
assert panel.groupby('date')['T'].nunique().eq(1).all()
assert panel.groupby('date')['C'].nunique().eq(1).all()
coverage.to_csv(OUT/'coverage.csv')
display(source_audit[['file','verified','recorded_retrieval_utc']])
display(coverage)
groups = {iso: panel.loc[panel.iso3.eq(iso)].reset_index(drop=True) for iso in COUNTRIES}
# Check boundary timing against raw source dates independently of row shifts.
oil = pd.read_csv(ROOT/'data/raw/brent_direct.csv', index_col=0, parse_dates=True).iloc[:,0]
activity = pd.read_csv(ROOT/'data/raw/igrea.csv', index_col=0, parse_dates=True).iloc[:,0]
for date in pd.to_datetime(['2000-01-01','2019-12-01','2020-01-01','2024-12-01']):
    observed = panel.loc[(panel.iso3=='USA') & (panel.date==date)].iloc[0]
    previous, before = date-pd.DateOffset(months=1), date-pd.DateOffset(months=2)
    assert np.isclose(observed['T'], 10*(oil.loc[previous]/oil.loc[before]-1))
    assert np.isclose(observed['C'], activity.loc[before]/100)
print('Four complete calendars and pre-treatment lag boundaries verified; no imputation.')
# %% [markdown]
# ## 3. Exploratory analysis
# We summarize the outcomes separately by country and the shared global predictors only once. This
# avoids presenting repeated oil observations as independent measurements. Descriptive lag plots and
# seasonal patterns are not causal evidence and do not determine whether an effect must be positive.
# %%
def describe(series):
    return pd.Series({'n':series.count(), 'minimum':series.min(), 'maximum':series.max(),
                      'mean':series.mean(), 'median':series.median(), 'sd':series.std(ddof=1)})

inflation_summary = panel.groupby('country')['monthly'].apply(describe).unstack()
global_summary = pd.DataFrame({name:describe(groups['USA'][column]) for name,column in
                               [('Brent USD/barrel','oil_usd'),('Oil treatment T','T'),('Prior global activity C','C')]}).T
inflation_summary.to_csv(OUT/'inflation_summary.csv')
global_summary.to_csv(OUT/'global_predictor_summary.csv')
display(inflation_summary.round(4))
display(global_summary.round(4))
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for ax, col, label in zip(axes, ['oil_usd','T','C'], ['Brent USD/barrel','Lagged oil change / 10','Prior IGREA / 100']):
    ax.plot(groups['USA'].date, groups['USA'][col], linewidth=1)
    ax.set_ylabel(label); ax.grid(alpha=.2)
save_figure('global_predictors')
# %%
fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, sharey=True)
for ax,(iso,g) in zip(axes.flat,groups.items()):
    ax.plot(g.date,g.monthly,linewidth=.8);ax.axhline(0,color='grey',linewidth=.5)
    ax.set_title(COUNTRIES[iso]);ax.set_ylabel('Monthly inflation %');ax.grid(alpha=.2)
save_figure('inflation_histories')
fig, axes = plt.subplots(2, 2, figsize=(11, 5), sharex=True, sharey=True)
bins = np.linspace(panel.monthly.min()-.1,panel.monthly.max()+.1,35)
for ax,(iso,g) in zip(axes.flat,groups.items()):
    ax.hist(g.monthly,bins=bins,density=True,alpha=.8)
    ax.set_title(COUNTRIES[iso]);ax.set_xlabel('Monthly inflation %');ax.set_ylabel('Density')
save_figure('inflation_distributions')
# %%
seasonal = panel.groupby(['iso3','month']).monthly.agg(['mean','std','count']).reset_index()
seasonal.to_csv(OUT/'descriptive_month_patterns.csv',index=False)
fig, axes = plt.subplots(2,2,figsize=(11,5),sharex=True,sharey=True)
for ax,(iso,g) in zip(axes.flat,groups.items()):
    s=seasonal[seasonal.iso3.eq(iso)]
    ax.plot(s.month,s['mean'],marker='o');ax.axhline(0,color='grey',linewidth=.5)
    ax.set_xticks([1,3,6,9,12]);ax.set_xlabel('Calendar month');ax.set_title(COUNTRIES[iso])
    ax.set_ylabel('Mean monthly inflation %')
save_figure('calendar_patterns')
print('Calendar means are descriptive full-period summaries, not training-period forecast predictors.')
print('Moving holidays and changes in seasonal patterns remain potential model failures.')
# %% [markdown]
# ## 4. Causal model and identification assumptions
# Our three-node hypothesis is $C_t\rightarrow T_t$, $C_t\rightarrow Y_{it}$ and
# $T_t\rightarrow Y_{it}$. Higher global demand may raise both oil prices and domestic inflation.
# Measuring the proxy before the treatment avoids treating an obvious post-treatment measurement as a
# confounder, but does not make the proxy exogenous or sufficient. We do not label oil movements clean
# supply shocks or interpret Bayesian computation as establishing causality.
#
# A causal interpretation additionally needs consistency of the treatment contrast, sufficient overlap,
# no remaining relevant confounding after the proposed adjustment, and an adequate temporal model.
# Exchange rates, supply constraints, taxes, administered prices and policy feedback can violate that
# argument. Country-specific seasonality and AR errors are statistical nuisance structure; they do not
# automatically satisfy a strict three-variable causal restriction. We must justify this scope explicitly.
#
# [Presno and Prestipino (2024)](https://www.federalreserve.gov/econres/notes/feds-notes/oil-price-shocks-and-inflation-in-a-dsge-model-of-the-global-economy-20240802.html)
# motivate consumer-energy and production-cost pathways. Their structural supply-shock effect and
# horizon differ from our nominal-price observational contrast, so their effect size is not copied as our prior.
# %%
fig, ax = plt.subplots(figsize=(9,2.3));ax.axis('off')
positions={'C':(.5,.83),'T':(.18,.2),'Y':(.82,.2)}
for start,end in [('C','T'),('C','Y'),('T','Y')]:
    ax.annotate('',xy=positions[end],xytext=positions[start],xycoords='axes fraction',
                arrowprops={'arrowstyle':'-|>','linewidth':1.4,'shrinkA':54,
                            'shrinkB':85 if start=='T' else 54})
for key,label in [('C','Prior global activity C'),('T','Oil-price change T'),('Y','Subsequent inflation Y')]:
    ax.text(*positions[key],label,transform=ax.transAxes,ha='center',va='center',fontsize=11,
            bbox={'boxstyle':'round,pad=.45','fc':'white','ec':'black'})
save_figure('causal_dag')
# %% [markdown]
# ## 5. Baseline model, priors and prior predictive checks
# For each country, separately:
# $$Y_{it}=\alpha_i+\beta_iT_t+\gamma_iC_t+s_{i,m(t)}+u_{it},\quad
# u_{it}=\phi_i u_{i,t-1}+\epsilon_{it},\quad \epsilon_{it}\sim\mathcal N(0,\sigma_i^2).$$
# The initial residual has $u_{i0}\sim\mathcal N(0,\sigma_i^2/(1-\phi_i^2))$ and $|\phi_i|<1$.
# Eleven fixed orthonormal month contrasts represent seasonal effects summing to zero. Omitting C will
# remove gamma only; seasonal structure, likelihood and prior choices remain matched.
#
# Independent working priors are $\alpha_i,\gamma_i,\delta_{ik}\sim\mathcal N(0,0.5^2)$,
# $\beta_i\sim\mathcal N(0,0.3^2)$, $\sigma_i\sim\mathrm{HalfNormal}(0.5)$,
# and $\phi_i\sim\mathrm{Uniform}(-1,1)$. These are the existing baseline priors in interpretable units.
# For external scale context, the [Fed's 2% long-run **PCE**, not CPI, objective](https://www.federalreserve.gov/faqs/economy_14400.htm)
# corresponds to approximately 0.165% monthly growth. It motivates small monthly units, not a CPI target
# or an assumed target for all four countries. Centering alpha at zero with SD 0.5 allows deflation and
# substantial departures from that context. Beta's central 95% prior range is about ±0.59 percentage
# points per 10-point oil movement; this is deliberately broad relative to the modest transmission in
# Presno/Prestipino's different annual supply-shock setting. We do not equate their estimand to ours.
# Gamma and seasonal contrast SDs of 0.5 allow large monthly deviations while shrinking unstable terms;
# the half-Normal residual scale allows occasional substantial shocks. Uniform phi permits either sign
# and broad persistence; prior prediction exposes the near-unit-root tail. These are weak modeling
# judgments rather than parameters learned from our outcomes; their sensitivity is reported below.
#
# The alternative fixes nu=5 before final estimation: innovations have Student-t scale sigma, with SD
# sigma*sqrt(5/3). Its initial residual is independently t_5(0,sigma/sqrt(1-phi²)). This matches the AR
# stationary variance but is **not** asserted to be the exact stationary t-AR marginal distribution.
# Identical priors are used on scale parameters, so innovation variance priors are larger for Student-t.
# We compare predictive adequacy, not an allegedly prior-free Bayes factor. Heavy tails cannot fix
# omitted confounding or every remaining dependence pattern. Fixed nu also avoids tuning for oil signs.
# Explicitly, the alternative has $\epsilon_{it}\sim t_5(0,\sigma_i)$ and
# $\sqrt{1-\phi_i^2}\,u_{i0}\sim t_5(0,\sigma_i)$, independently of subsequent innovations.
# %%
# Shared dates, T and C imply identical prior-predictive distributions for all countries.
X,names,scales=design(groups['USA'])
for g in groups.values():
    assert np.allclose(design(g)[0],X)
count=3000
coefficients=RNG.normal(size=(count,len(names)))*scales
sigma=np.abs(RNG.normal(0,.5,count));phi=RNG.uniform(-1,1,count)
prior_rows=[]
for family in ['gaussian','student']:
    noise=(RNG.standard_t(5,size=(count,len(X))) if family=='student' else RNG.normal(size=(count,len(X))))*sigma[:,None]
    errors=np.empty_like(noise);errors[:,0]=noise[:,0]/np.sqrt(1-phi**2)
    for j in range(1,len(X)):errors[:,j]=phi*errors[:,j-1]+noise[:,j]
    predicted=coefficients@X.T+errors
    lo,mid,hi=np.quantile(predicted,[.025,.5,.975]);outside=np.mean(np.abs(predicted)>5)
    prior_rows.append({'family':family,'lower95':lo,'median':mid,'upper95':hi,'fraction_abs_inflation_above_5':outside})
    plt.hist(predicted.ravel(),bins=np.linspace(-5,5,81),histtype='step',label=family)
prior_summary=pd.DataFrame(prior_rows)
plt.title('Shared-country prior prediction: Gaussian versus Student-t')
plt.xlabel('Prior-predicted monthly inflation %');plt.ylabel('Draw count in displayed bins');plt.legend()
save_figure('prior_predictive')
prior_summary.to_csv(OUT/'prior_predictive_summary.csv',index=False)
display(prior_summary.round(4))
print('Histogram display is limited to [-5,5]; the table reports tail mass, not hidden truncation.')
print('The prior is conditional on shared predictors, before observing inflation; no country inflation is used.')
# %% [markdown]
# ## 6. Simulation validation
# We generate a separate 300-month system with C driving both T and Y, seasonal effects, and dependent
# errors. True beta=.15, gamma=.25, sigma=.15 and phi=.4. Each likelihood is fitted to its corresponding
# simulated data, with and without C. No synthetic row is merged with real measurements. One recovery
# experiment per family tests implementation under known assumptions; it is not a repeated-coverage study.
#
# The Student-t sampler uses a Normal/Gamma scale mixture: lambda_t~Gamma(nu/2,nu/2) in shape/rate
# parameterization; whitened residuals
# conditional on lambda_t are Normal(0,sigma²/lambda_t). Coefficients have exact Gaussian Gibbs updates;
# log(sigma) and atanh(phi) use Metropolis updates with Jacobians. Each fit has four seeded chains,
# 1,500 warmup and 4,000 retained iterations per chain. Adaptation stops after warmup; there is no thinning.
# Independent unit tests check the integrated mixture density and Gaussian joint AR covariance.
# %%
from analysis.robust_model import (specification, fit_cached, simulation_data,
                                   posterior_replications, one_step_forecast, point_loglik)
from analysis.model import summarize
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.special import logsumexp
import statsmodels.api as sm

simulation_tables=[];simulation_betas=[]
for fi,family in enumerate(['gaussian','student']):
    simulated,truth=simulation_data(family)
    for adjusted in [False,True]:
        fit=fit_cached(simulated,adjusted=adjusted,family=family,seed=3000+fi*100+int(adjusted),
                       label=f'simulation/{family}/adjusted={adjusted}')
        table=fit['diagnostics'].copy();table['family']=family;table['adjusted']=adjusted
        if adjusted:
            table['truth']=truth
            table['truth_in_95']=(table.truth>=table.lower95)&(table.truth<=table.upper95)
        simulation_tables.append(table)
        row=table[table.parameter.eq('beta')].iloc[0]
        simulation_betas.append({'family':family,'adjusted':adjusted,'mean':row['mean'],'lower95':row.lower95,'upper95':row.upper95,'truth':.15})
simulation=pd.concat(simulation_tables,ignore_index=True)
simulation.to_csv(OUT/'simulation_recovery.csv',index=False)
display(simulation[simulation.parameter.isin(['beta','gamma'])][['family','adjusted','parameter','mean','lower95','upper95','truth','truth_in_95']].round(4))
fig,ax=plt.subplots(figsize=(9,3))
for j,row in enumerate(simulation_betas):
    ax.errorbar(row['mean'],j,xerr=[[row['mean']-row['lower95']],[row['upper95']-row['mean']]],fmt='o')
ax.axvline(.15,color='black',linestyle='--',label='True beta = .15')
ax.set_yticks(range(4),[r['family']+(' + C' if r['adjusted'] else ' omit C') for r in simulation_betas])
ax.set_xlabel('Simulated oil coefficient');ax.legend();save_figure('simulation_recovery')
assert simulation.sampling_pass.all(), 'Inspect synthetic-chain diagnostics before continuing'
# %% [markdown]
# ## 7. Real-data estimation and sampling checks
# Both families are fitted separately in every country, retaining or omitting only C. All full-data
# coefficients and priors share the stated units. Reproducible fits are cached locally using model-source,
# data, prior and sampler hashes; cache hits are recorded rather than presented as new sampling runs.
# Rank-normalized R-hat <1.01 and bulk/tail ESS >400 are required for every parameter before interpretation.
# Conditional oil intervals are not formal tests of country differences, and model failures remain relevant.
# %%
fits={};diagnostic_tables=[];coefficient_rows=[];fit_records=[]
for ci,(iso,g) in enumerate(groups.items()):
    fits[iso]={}
    for fi,family in enumerate(['gaussian','student']):
        for adjusted in [False,True]:
            fit=fit_cached(g,adjusted=adjusted,family=family,seed=10000+ci*1000+fi*100+int(adjusted),
                           label=f'full/{iso}/{family}/adjusted={adjusted}')
            fits[iso][(family,adjusted)]=fit
            table=fit['diagnostics'].copy();table['iso3']=iso;table['family']=family;table['adjusted']=adjusted
            diagnostic_tables.append(table)
            row=table[table.parameter.eq('beta')].iloc[0]
            coefficient_rows.append({'iso3':iso,'family':family,'adjusted':adjusted,'beta':row['mean'],
                                     'lower95':row.lower95,'upper95':row.upper95,'P_beta_positive':np.mean(fit['chains'][:,:,1]>0)})
            fit_records.append({'iso3':iso,'family':family,'adjusted':adjusted,'cache_key':fit['key'],'cache_hit':fit['cache_hit']})
diagnostics=pd.concat(diagnostic_tables,ignore_index=True);estimates=pd.DataFrame(coefficient_rows)
diagnostics.to_csv(OUT/'posterior_diagnostics.csv',index=False);estimates.to_csv(OUT/'country_estimates.csv',index=False)
pd.DataFrame(fit_records).to_csv(OUT/'fit_provenance.csv',index=False)
sampling=diagnostics.groupby(['iso3','family','adjusted']).agg(max_Rhat=('r_hat','max'),min_bulk_ESS=('ess_bulk','min'),min_tail_ESS=('ess_tail','min'))
display(estimates.round(4));display(sampling.round(2))
print('P_beta_positive is a finite-draw fraction; a displayed 1.0 is not logical certainty or a causal probability.')
assert diagnostics.sampling_pass.all(), 'Inspect failed real-data sampling diagnostics'
fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True,sharey=True)
for ax,family in zip(axes,['gaussian','student']):
    for j,adjusted in enumerate([False,True]):
        q=estimates[(estimates.family==family)&(estimates.adjusted==adjusted)].set_index('iso3').loc[list(COUNTRIES)]
        ax.errorbar(q.beta,np.arange(4)+(j-.5)*.15,xerr=[q.beta-q.lower95,q.upper95-q.beta],fmt='o',label='include C' if adjusted else 'omit C')
    ax.axvline(0,color='grey',linewidth=.7);ax.set_title(family);ax.set_xlabel('Inflation pp per 10 pp oil change');ax.legend(loc='upper left',fontsize=8)
axes[0].set_yticks(range(4),list(COUNTRIES.values()));save_figure('country_coefficients')
fig,axes=plt.subplots(4,2,figsize=(11,8))
for i,(iso,g) in enumerate(groups.items()):
    a=fits[iso][('student',True)]['chains']
    for chain in range(4):
        axes[i,0].plot(a[chain,:,1],linewidth=.3,alpha=.65)
        axes[i,1].plot(a[chain,:,-1],linewidth=.3,alpha=.65)
    axes[i,0].set_ylabel(iso+' beta');axes[i,1].set_ylabel(iso+' phi')
save_figure('student_traces')
# %% [markdown]
# ## 8. Model evaluation and sensitivity
# Posterior replications include regression-mean uncertainty, parameter uncertainty and new dependent
# residuals. Student-t comparisons use the stated initial distribution. We check extrema, variance and
# persistence, plus plug-in innovation dependence at 6/12 lags. These approximate diagnostics are not a
# guarantee of an adequate time-series likelihood. Full-sample seasonal plots are not forecast inputs.
# %%
pp_rows=[];lb_rows=[];innovations={};fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
for ci,(ax,(iso,g)) in enumerate(zip(axes.flat,groups.items())):
    for fi,family in enumerate(['gaussian','student']):
        fit=fits[iso][(family,True)];mu,rep=posterior_replications(fit,seed=800+ci*10+fi,count=1500)
        mean=fit['chains'].mean(axis=(0,1));res=g.monthly.to_numpy()-fit['X']@mean[:-2]
        innov=res[1:]-mean[-1]*res[:-1]
        lb=acorr_ljungbox(innov,lags=[6,12],return_df=True).rename_axis('lag').reset_index()
        lb['iso3']=iso;lb['family']=family;lb_rows.append(lb)
        for name,fun in [('minimum',np.min),('maximum',np.max),('sd',np.std),('lag1',lambda x:pd.Series(x).autocorr())]:
            values=np.array([fun(v) for v in rep]);obs=fun(g.monthly.to_numpy());lo,hi=np.quantile(values,[.025,.975])
            pp_rows.append({'iso3':iso,'family':family,'statistic':name,'observed':obs,'rep_lower95':lo,'rep_upper95':hi,
                            'P_rep_ge_observed':np.mean(values>=obs),'outside_rep_95':bool(obs<lo or obs>hi)})
        if family=='student':
            innovations[iso]=innov
            ax.fill_between(g.date,*np.quantile(rep,[.025,.975],axis=0),alpha=.15,label='95% predictive')
            ax.fill_between(g.date,*np.quantile(mu,[.025,.975],axis=0),alpha=.4,label='95% mean')
            ax.plot(g.date,mu.mean(axis=0),linewidth=.8,label='Posterior regression mean')
            ax.plot(g.date,g.monthly,color='black',linewidth=.6,label='Observed');ax.set_title(COUNTRIES[iso])
            ax.set_ylabel('Monthly inflation %')
axes[0,0].legend(fontsize=7,ncol=2);save_figure('student_posterior_predictive')
predictive=pd.DataFrame(pp_rows);lb=pd.concat(lb_rows,ignore_index=True)
predictive.to_csv(OUT/'predictive_checks.csv',index=False);lb.to_csv(OUT/'innovation_checks.csv',index=False)
display(predictive[predictive.statistic.isin(['minimum','maximum'])].round(4))
display(lb[lb.lag.eq(12)].round(4))
innovation_correlation=pd.DataFrame(innovations).corr()
innovation_correlation.to_csv(OUT/'cross_country_innovation_correlation.csv')
display(Markdown('**Shared-shock check:** residual innovation correlations below reinforce why separate country fits do not form an independent pooled sample.'))
display(innovation_correlation.round(3))
# %%
prior_rows=[];sensitivity=[]
for iso,g in groups.items():
    fit=fits[iso][('student',True)];flat=fit['chains'].reshape(-1,len(fit['names'])+2)
    for label,logweight in [('beta SD .15',-.5*(flat[:,1]/.15)**2+.5*(flat[:,1]/.3)**2),
                            ('beta SD .60',-.5*(flat[:,1]/.6)**2+.5*(flat[:,1]/.3)**2),
                            ('phi transformed Beta(2,2)',np.log1p(-flat[:,-1]**2))]:
        w=np.exp(logweight-logweight.max());w/=w.sum();order=np.argsort(flat[:,1])
        lo,hi=np.interp([.025,.975],np.cumsum(w[order]),flat[order,1])
        prior_rows.append({'iso3':iso,'alternative_prior':label,'beta':w@flat[:,1],'lower95':lo,'upper95':hi,
                           'weight_ESS':1/(w@w)})
    for label,h in [('2000-2019',g[g.date<'2020-01-01']),('2020-2024',g[g.date>='2020-01-01']),
                    ('exclude 2020-2022',g[~g.date.dt.year.between(2020,2022)]),('exclude |oil change| >30%',g[g['T'].abs()<=3])]:
        r=sm.OLS(h.monthly,specification(h)[0]).fit(cov_type='HAC',cov_kwds={'maxlags':12});ci=np.asarray(r.conf_int())[1]
        sensitivity.append({'iso3':iso,'sample':label,'n':len(h),'beta_OLS':np.asarray(r.params)[1],'HAC_lower':ci[0],'HAC_upper':ci[1]})
priors=pd.DataFrame(prior_rows);robustness=pd.DataFrame(sensitivity)
priors.to_csv(OUT/'prior_sensitivity.csv',index=False);robustness.to_csv(OUT/'subperiod_sensitivity.csv',index=False)
display(priors.round(4));display(robustness.round(4))
display(Markdown('Prior reweighting ESS assesses weight concentration, not replacement-chain mixing. HAC OLS checks use a different estimator and are exploratory; removing calendar segments also complicates lag-based inference. The extension suite below tests nu=3,5,10 and alternative dependence on matched conditional samples.'))
# %% [markdown]
# ### Retrospective prediction and matched no-oil comparison
# Train on 2000–2019 only, then predict each month of 2020–2024 using observed previous residuals.
# Student-t models with/without oil have the same C, month structure, priors on retained parameters and
# likelihood. Gaussian oil forecasts and simple benchmarks provide context. Fixed coefficients and
# training-only seasonal estimates prevent test-outcome refitting. Revised source data and publication
# assumptions still prevent a real-time forecast claim. The test period informed earlier exploratory work.
# %%
forecast_rows=[];forecast_detail=[];training_tables=[];oil_rows=[];fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
for ci,(ax,(iso,g)) in enumerate(zip(axes.flat,groups.items())):
    train=g[g.date<'2020-01-01'];test=g[g.date>='2020-01-01'];forecasts={}
    for fi,family in enumerate(['gaussian','student']):
        for oil in ([True] if family=='gaussian' else [True,False]):
            fit=fit_cached(train,family=family,oil=oil,seed=20000+ci*1000+fi*100+int(oil),
                           label=f'training/{iso}/{family}/oil={oil}')
            table=fit['diagnostics'].copy();table['iso3']=iso;table['family']=family;table['oil']=oil;training_tables.append(table)
            result=one_step_forecast(fit,g,seed=400+ci*10+fi+int(oil))
            label=family+(' + oil' if oil else ' no oil');forecasts[label]=result
            forecast_rows.append({'iso3':iso,'model':label,'test_n':60,**{k:result[k] for k in ['MAE','RMSE','coverage95','mean_log_predictive_density']}})
            for j,date in enumerate(test.date):
                forecast_detail.append({'iso3':iso,'model':label,'date':date,'actual':test.monthly.iloc[j],
                                        'prediction':result['prediction'][j],'lower95':result['lower95'][j],'upper95':result['upper95'][j]})
    for label,pred in [('last month',test.last_monthly.to_numpy()),('training month mean',test.month.map(train.groupby('month').monthly.mean()).to_numpy()),
                       ('training mean',np.repeat(train.monthly.mean(),60)),('same month last year',test.last_year_same_month.to_numpy())]:
        error=test.monthly.to_numpy()-pred
        forecast_rows.append({'iso3':iso,'model':label,'test_n':60,'MAE':np.abs(error).mean(),'RMSE':np.sqrt(np.mean(error**2))})
    full,nooil=forecasts['student + oil'],forecasts['student no oil']
    oil_rows.append({'iso3':iso,'RMSE_with_oil':full['RMSE'],'RMSE_without_oil':nooil['RMSE'],
                     'RMSE_reduction_percent':100*(nooil['RMSE']-full['RMSE'])/nooil['RMSE'],
                     'log_score_gain':full['mean_log_predictive_density']-nooil['mean_log_predictive_density']})
    ax.plot(test.date,test.monthly,color='black',linewidth=.8,label='Observed')
    ax.plot(test.date,full['prediction'],label='Student-t with oil')
    ax.plot(test.date,nooil['prediction'],label='Student-t without oil',alpha=.7);ax.set_title(COUNTRIES[iso]);ax.set_ylabel('Monthly inflation %')
axes[0,0].legend(fontsize=7);save_figure('holdout_predictions')
forecasts=pd.DataFrame(forecast_rows);oil_comparison=pd.DataFrame(oil_rows);training=pd.concat(training_tables,ignore_index=True)
forecasts.to_csv(OUT/'forecast_metrics.csv',index=False);oil_comparison.to_csv(OUT/'oil_forecast_comparison.csv',index=False)
pd.DataFrame(forecast_detail).to_csv(OUT/'forecast_predictions.csv',index=False);training.to_csv(OUT/'training_diagnostics.csv',index=False)
display(forecasts.round(4));display(oil_comparison.round(3))
assert training.sampling_pass.all(), 'Inspect training-model sampling diagnostics'
# %% [markdown]
# ### Strengthening experiments: matched conditional comparisons
# We compare AR1, AR2 and multiplicative seasonal AR errors on identical outcomes, conditioning on
# the first 13 months in every case. These sensitivity fits differ from the stationary-initial baseline.
# Three chronological five-year evaluation windows compare oil/no-oil forecasts, with paired 12-month
# block uncertainty. Forty independently simulated datasets test recovery across four scenarios.
# Further checks vary tail shape, oil timing and pre/post-2020 periods. No model is selected by significance.
# %%
from analysis.strengthening import run_suite, OUT as STRENGTHENING_OUT
from IPython.display import Image
extensions=run_suite()
display(Markdown('**Matched dependence and tail diagnostics**'))
display(extensions['dependence_tail_checks'].round(4))
display(Markdown('**Oil coefficients across dependence, tail, lag and period specifications**'))
display(extensions['coefficient_sensitivity'].round(4))
display(Markdown('**Rolling prediction: 180 evaluation months per country**'))
display(extensions['rolling_aggregate'].round(4))
display(Markdown('**Repeated recovery: Monte Carlo uncertainty matters**'))
display(extensions['simulation_summary'].round(4))
display(Markdown('**Extremes retained, with explicit conditioning-prefix status**'))
display(extensions['extreme_observation_audit'].round(4))
for figure in ['innovation_ACF.png','baseline_PACF.png','dependence_coefficients.png',
               'rolling_gains.png','simulation_recovery.png']:
    display(Image(filename=str(STRENGTHENING_OUT/figure)))
display(Markdown((STRENGTHENING_OUT/'FINDINGS.md').read_text(encoding='utf-8')))
# %% [markdown]
# ### Extension through the latest common monthly release
# Current snapshots are separately versioned. We preserve missing calendar months, fit segmented
# conditional AR1 models, and forecast 2025 onward using coefficients trained only through 2024.
# Daily oil observations provide recent context, with no substitution for monthly averages.
# %%
from analysis.latest_analysis import run_latest
from analysis.latest_data import OUT as LATEST_OUT
latest=run_latest()
display(Markdown('**Current source availability and historical revisions**'))
display(latest['availability']);display(latest['historical_revisions'].round(7))
display(Markdown('**Matched current-vintage coefficients: through 2024 versus extended**'))
display(latest['coefficients'].drop(columns='key').round(4))
display(Markdown('**Fixed-training recent forecasts, with calendar-safe exclusions**'))
display(latest['forecast_metrics'].round(4));display(latest['oil_forecast_gains'].round(3))
for figure in ['recent_history.png','recent_forecasts.png']:
    display(Image(filename=str(LATEST_OUT/figure)))
display(Markdown((LATEST_OUT/'FINDINGS.md').read_text(encoding='utf-8')))
# %% [markdown]
# ## 9. Findings, interpretation and limitations
# The following summary is generated from the completed fits, not from anticipated coefficient signs.
# The adjusted/unadjusted difference in real data is not itself a measurement of causal bias removal.
# Single-country intervals do not account for shared omitted causes and do not establish country rankings.
# %%
lines=['# Completed four-country computational analysis','',
       'All source hashes and 1,200 country-month rows verified. All simulation, full-data and training sampling checks passed.','',
       '## Oil–inflation associations (Student-t, demand-adjusted)','']
for iso,name in COUNTRIES.items():
    a=estimates[(estimates.iso3==iso)&(estimates.family=='student')&estimates.adjusted].iloc[0]
    b=estimates[(estimates.iso3==iso)&(estimates.family=='student')&~estimates.adjusted].iloc[0]
    lines.append(f'- **{name}:** beta {a.beta:.4f}, 95% credible interval [{a.lower95:.4f}, {a.upper95:.4f}]; omitting demand gives {b.beta:.4f}.')
lines+=['','Units are monthly inflation percentage points per 10-percentage-point higher monthly oil change. These are model-conditional associations.','',
        '## Predictive checks and residual dependence','']
for iso,name in COUNTRIES.items():
    p12=lb[(lb.iso3==iso)&(lb.family=='student')&(lb.lag==12)].lb_pvalue.iloc[0]
    misses=predictive[(predictive.iso3==iso)&(predictive.family=='student')&predictive.outside_rep_95]
    stats=', '.join(misses.statistic) or 'none among the four checked summaries'
    lines.append(f'- **{name}:** Student-t 12-lag innovation p={p12:.4f}; predictive-summary misses: {stats}.')
lines+=['','## Does oil improve the matched retrospective forecast?','']
for _,r in oil_comparison.iterrows():
    direction='reduces' if r.RMSE_reduction_percent>0 else 'increases'
    lines.append(f'- **{COUNTRIES[r.iso3]}:** adding oil {direction} RMSE by {abs(r.RMSE_reduction_percent):.2f}% ({r.RMSE_without_oil:.4f} without oil versus {r.RMSE_with_oil:.4f} with oil); mean log-score gain {r.log_score_gain:.4f}.')
lines+=['','## Conclusions and future work','',
        'Oil coefficients, confounder adjustment and forecast gains answer different questions. The simulation shows what omitted-confounder bias can look like under a known DAG; real-data adjustment cannot certify that IGREA blocks every relevant path. Heavy-tailed innovations may improve extremes while leaving serial or calendar dependence. We report both families rather than selecting by significance.',
        '', 'Interpretation depends on consistency, overlap, adequate temporal structure and sufficient confounding control. Global-demand measurement error, exchange rates, subsidies, taxes, moving holidays and shared shocks remain limitations. Student-t nu=5 and the variance-matched initial distribution are additional assumptions; sigma is a scale, not the Student-t innovation SD. Prior reweighting and HAC checks do not cover every form of model uncertainty.',
        '', 'The strengthening experiments above test alternative tails, dependence, lags and periods; their generated findings qualify the baseline results. Moving holidays, local-currency oil exposure and better-identified supply shocks remain future work. Interest rates and pooled country-effect comparisons require a separate expanded design. No causal ranking across countries or real-time forecast claim follows from this analysis.']
lines += ['', '## Completed strengthening experiments', '', (STRENGTHENING_OUT/'FINDINGS.md').read_text(encoding='utf-8')]
lines += ['', '## Current-release extension', '', (LATEST_OUT/'FINDINGS.md').read_text(encoding='utf-8')]
report='\n'.join(lines).rstrip()+'\n'
(OUT/'FINDINGS.md').write_text(report,encoding='utf-8')
display(Markdown(report))
print('Computational sections complete. Submission PDF pagination and the group screencast remain separate deliverables.')
