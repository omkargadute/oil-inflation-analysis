# %% [markdown]
# # Oil-price changes and inflation across four economies, 2000–2026
# **Foundations of Data Science — final computational notebook**
#
# Sunidhi Suri (ss21743), Omkar Gadute (org2029), Hriday Badani (hrb9339)
#
# **One main dataset: January 2000–October 2026.** Every real-data descriptive chart, main fit,
# posterior check and sensitivity analysis below uses this combined calendar. Forecast models use
# earlier chronological training samples and evaluate later outcomes through October 2026.
# The former 2000–2024 estimates appear only as an explicitly labeled historical comparison.
#
# ## 1. Research question and estimand
# How does a 10-percentage-point higher monthly Brent price change relate to next-month consumer
# inflation in the United States, China, Japan and Canada, adjusting for prior global activity?
# Our country-specific target is $\tau_i(a)=E[Y_{it}(a+1)-Y_{it}(a)]$. Under a constant linear effect
# and the identification assumptions in Section 4, this equals $\beta_i$; otherwise beta is an
# association conditional on the statistical model. Shared oil histories are not independent country shocks.
# %%
from pathlib import Path
import sys,platform,importlib.metadata as metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from IPython.display import display,Markdown,Image

ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'analysis/data.py').exists())
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from analysis.data import COUNTRIES
from analysis.latest_data import prepare_latest,write_data_audit,RAW
from analysis.robust_model import specification,simulation_data
from analysis.combined_model import fit_combined,innovations,INITIAL
from analysis.combined_analysis import run_combined,OUT

RNG=np.random.default_rng(20261124)
plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False})
def save_figure(name):
    plt.tight_layout();plt.savefig(OUT/(name+'.png'),bbox_inches='tight');plt.show();plt.close()
def full_period_axis(ax):
    ax.set_xlim(pd.Timestamp('2000-01-01'),END)
    ax.set_xticks(pd.to_datetime([f'{year}-01-01' for year in [2000,2005,2010,2015,2020,2026]]))
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
display(pd.DataFrame({'package':['Python','numpy','pandas','scipy','arviz','numba'],
    'version':[platform.python_version()]+[metadata.version(p) for p in ['numpy','pandas','scipy','arviz','numba']]}))
# %% [markdown]
# ## 2. Data, provenance and complete calendar
# CPI is the OECD national all-items **unadjusted** index in every country. US/China use the COICOP
# 1999 flow and Japan/Canada the COICOP 2018 flow, without within-country splicing. National baskets
# are not perfectly harmonized. Brent is nominal USD/barrel from EIA via FRED; IGREA is the Dallas
# Fed shipping-based industrial-commodity activity proxy, not GDP growth.
#
# | Role | Definition | Units |
# |---|---|---|
# | T | $100(P_{t-1}/P_{t-2}-1)/10$ | 10-percentage-point monthly oil movement |
# | Y | $100(I_{it}/I_{i,t-1}-1)$ | Monthly inflation percentage points, not annualized |
# | C | $G_{t-2}/100$ | 100 global-activity index points |
#
# History begins in 1999 for calendar-aligned transformations. The shared monthly endpoint is
# **October 2026**: 322 months per country, 1,288 calendar rows and 1,286 observed monthly outcomes.
# IGREA through September supports the required lag. Daily oil through November 17 provides context
# only; incomplete November prices are never substituted for a monthly average.
#
# **Missing US CPI:** October 2025 was not published following the shutdown
# ([BLS explanation](https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm)).
# Consequently October **and November** monthly inflation are undefined. We reindex before taking
# growth, use no interpolation/forward fill, and preserve these two NaNs. Section 5 describes Bayesian
# integration during fitting; those latent draws are never written as observed CPI or inflation.
# Revised reference-month snapshots do not establish real-time publication availability.
# %%
panel,daily=write_data_audit()
_,source_audit,availability,missing,_=prepare_latest()
panel.to_csv(ROOT/'data/processed/multicountry_monthly.csv',index=False)
source_audit.to_csv(OUT/'source_verification.csv',index=False)
groups={iso:panel[panel.iso3.eq(iso)].reset_index(drop=True) for iso in COUNTRIES}
END=panel.date.max()
coverage=panel.groupby('iso3',sort=False).agg(country=('country','first'),first=('date','min'),
    last=('date','max'),calendar_rows=('date','size'),observed_monthly_outcomes=('monthly','count'))
coverage.to_csv(OUT/'coverage.csv')
assert coverage.calendar_rows.eq(322).all() and panel.monthly.notna().sum()==1286
assert END==pd.Timestamp('2026-10-01')
assert panel.groupby('date')['T'].nunique().eq(1).all()
assert panel.groupby('date')['C'].nunique().eq(1).all()
display(source_audit[['file','verified','retrieved_utc']]);display(availability);display(coverage);display(missing)
oil=pd.read_csv(RAW/'brent_direct.csv',index_col=0,parse_dates=True).iloc[:,0]
activity=pd.read_csv(RAW/'igrea.csv',index_col=0,parse_dates=True).iloc[:,0]
for date in pd.to_datetime(['2000-01-01','2024-12-01','2025-01-01','2026-10-01']):
    row=groups['USA'].set_index('date').loc[date]
    assert np.isclose(row['T'],10*(oil.loc[date-pd.DateOffset(months=1)]/oil.loc[date-pd.DateOffset(months=2)]-1))
    assert np.isclose(row['C'],activity.loc[date-pd.DateOffset(months=2)]/100)
print('Verified hashes, common endpoint, missing-month preservation and treatment timing.')
# %% [markdown]
# ## 3. Exploratory analysis — entire combined period
# We summarize each country's outcome separately and shared global predictors once. All time-series
# plots in this section end in October 2026. The US gap appears as a break rather than a joined line.
# %%
summary=panel.groupby('country').monthly.agg(['count','mean','median','std','min','max'])
summary.to_csv(OUT/'inflation_summary.csv');display(summary.round(4))
extremes=[]
for iso,g in groups.items():
    for event,index in [('minimum',g.monthly.idxmin()),('maximum',g.monthly.idxmax())]:
        extremes.append({'country':COUNTRIES[iso],'event':event,'date':g.loc[index,'date'],
                         'monthly_inflation':g.loc[index,'monthly'],'action':'retained'})
extremes=pd.DataFrame(extremes);extremes.to_csv(OUT/'extreme_observation_audit.csv',index=False);display(extremes)
display(groups['USA'][['oil_usd','T','C']].describe().round(4))
fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True)
for ax,col,label in zip(axes,['oil_usd','T','C'],['Brent USD/barrel','Lagged oil change / 10','Prior IGREA / 100']):
    ax.plot(groups['USA'].date,groups['USA'][col],linewidth=.9);ax.set_ylabel(label);ax.grid(alpha=.2)
full_period_axis(axes[-1]);axes[0].set_title('Shared global predictors: January 2000–October 2026')
save_figure('global_predictors')
# %%
fig,axes=plt.subplots(2,2,figsize=(12,6),sharex=True,sharey=True)
for ax,(iso,g) in zip(axes.flat,groups.items()):
    ax.plot(g.date,g.monthly,linewidth=.8);ax.axhline(0,color='grey',linewidth=.5)
    ax.set_title(COUNTRIES[iso]);ax.set_ylabel('Monthly inflation %');full_period_axis(ax)
fig.suptitle('Observed inflation: January 2000–October 2026');save_figure('inflation_histories')
fig,axes=plt.subplots(2,2,figsize=(11,5),sharex=True,sharey=True)
bins=np.linspace(panel.monthly.min()-.1,panel.monthly.max()+.1,35)
for ax,(iso,g) in zip(axes.flat,groups.items()):
    ax.hist(g.monthly.dropna(),bins=bins,density=True);ax.set_title(COUNTRIES[iso]);ax.set_xlabel('Monthly inflation %')
fig.suptitle('Full-period inflation distributions');save_figure('inflation_distributions')
# %%
seasonal=panel.groupby(['iso3','month']).monthly.agg(['mean','std','count']).reset_index()
seasonal.to_csv(OUT/'descriptive_month_patterns.csv',index=False)
fig,axes=plt.subplots(2,2,figsize=(11,5),sharex=True,sharey=True)
for ax,(iso,g) in zip(axes.flat,groups.items()):
    q=seasonal[seasonal.iso3.eq(iso)];ax.plot(q.month,q['mean'],'o-');ax.axhline(0,color='grey')
    ax.set_title(COUNTRIES[iso]);ax.set_xlabel('Calendar month');ax.set_ylabel('Mean monthly inflation %')
fig.suptitle('Descriptive calendar patterns, 2000–October 2026');save_figure('calendar_patterns')
print('These full-period calendar summaries are descriptive; forecast seasonality is estimated using training data only.')
# %% [markdown]
# ## 4. Causal hypothesis and assumptions
# Our three substantive constructs form $C\rightarrow T$, $C\rightarrow Y$, $T\rightarrow Y$.
# Global demand may raise oil prices and inflation. Earlier measurement of C avoids an obvious
# post-treatment adjustment, but does not establish exogeneity or sufficient confounding control.
# Consistency, overlap, no remaining relevant confounding, and an adequate temporal specification are
# needed for a causal interpretation. Exchange rates, expectations, taxes, subsidies, supply constraints
# and policy feedback remain possible violations. Shared histories prevent treating countries as
# independent exposures or ranking their causal effects from separate intervals.
#
# Month contrasts and AR errors are statistical structure, not extra proven confounder adjustment.
# Their role must be justified under the course's three-variable scope.
# [Presno and Prestipino (2024)](https://www.federalreserve.gov/econres/notes/feds-notes/oil-price-shocks-and-inflation-in-a-dsge-model-of-the-global-economy-20240802.html)
# motivate energy-cost transmission, but their structural supply shock and horizon differ from our
# observational nominal-price estimand. We do not classify price movements as identified supply shocks.
# %%
fig,ax=plt.subplots(figsize=(9,2.4));ax.axis('off')
positions={'C':(.5,.82),'T':(.18,.2),'Y':(.82,.2)}
for a,b in [('C','T'),('C','Y'),('T','Y')]:
    ax.annotate('',xy=positions[b],xytext=positions[a],xycoords='axes fraction',
        arrowprops={'arrowstyle':'-|>','shrinkA':55,'shrinkB':80 if a=='T' else 55})
for k,label in [('C','Prior global activity C'),('T','Oil-price change T'),('Y','Subsequent inflation Y')]:
    ax.text(*positions[k],label,ha='center',va='center',transform=ax.transAxes,
            bbox={'boxstyle':'round,pad=.45','fc':'white','ec':'black'})
save_figure('causal_dag')
# %% [markdown]
# ## 5. Statistical model, missing outcomes and priors
# Separately for each country,
# $$Y_{it}=\alpha_i+\beta_iT_t+\gamma_iC_t+s_{i,m(t)}+u_{it},\qquad
# u_{it}=\phi_i u_{i,t-1}+\epsilon_{it}.$$
# Innovations are Gaussian $N(0,\sigma_i^2)$ or Student-$t_5(0,\sigma_i)$; sigma is a Student-t scale,
# with innovation SD $\sigma\sqrt{5/3}$. Eleven orthonormal month contrasts sum to zero.
# All models **condition on the first 13 observed months**, so family/dependence comparisons share
# the same initial boundary without asserting an exact stationary Student-t initial distribution.
# Full-period models have 309 subsequent calendar positions per country; two US outcomes are latent.
#
# Priors: $\alpha,\gamma,\delta_k\sim N(0,0.5^2)$; $\beta\sim N(0,0.3^2)$;
# $\sigma\sim\operatorname{HalfNormal}(0.5)$; $\phi\sim U(-1,1)$, independently.
# For scale context, the [Fed's 2% PCE objective](https://www.federalreserve.gov/faqs/economy_14400.htm)
# is approximately 0.165% monthly growth; it is neither a CPI target nor a target for every country.
# The beta prior allows roughly ±0.59 inflation percentage points per 10-point oil movement.
# These broad working judgments permit deflation and large departures; prior prediction and sensitivity
# assess their implications. Equal sigma priors across families do not imply equal variance priors.
#
# **Missing-state likelihood:** let M denote the two undefined US monthly outcomes and O the observed
# ones after the conditioning prefix. The target uses
# $$p(\theta\mid Y_O,X,Y_{0:12})\propto p(\theta)
# \int \prod_{t=13}^{n-1}p(Y_t\mid Y_{t-1:t-p},X,\theta)\,dY_M.$$
# Student-t errors use a Normal/Gamma mixture. Conditional on mixture weights and parameters, missing
# outcomes have an exact multivariate Gaussian Gibbs update. Thus later observations inform fitting
# across the gap; we do not join nonadjacent months or discard all later US observations.
# Raw NaNs stay unchanged. We model observed monthly growth, without additionally imposing the known
# two-month CPI-level ratio as a nonlinear constraint. This leaves some index-level information unused.
#
# Sensitivities use AR2 in stable partial-autocorrelation coordinates and
# $(1-\phi B)(1-\Phi B^{12})u_t=\epsilon_t$ for seasonal AR, plus nu=3/10 under AR1.
# Forecast scoring requires observed outcomes at the model's actual lag positions; latent test outcomes
# are never supplied as observations. Computation uses four seeded chains, warmup-only adaptation,
# R-hat <1.01 and bulk/tail ESS >400 for every parameter **and latent outcome**, with longer retries if needed.
# %%
X,names,scales=specification(groups['USA']);count=2000
b=RNG.normal(size=(count,len(names)))*scales
sigma=np.abs(RNG.normal(0,.5,count));phi=RNG.uniform(-1,1,count)
prior_rows=[]
for family in ['gaussian','student']:
    mu=b@X.T;residual=np.empty_like(mu)
    residual[:,:13]=groups['USA'].monthly.to_numpy()[:13]-mu[:,:13]
    noise=RNG.standard_t(5,(count,len(X)-13)) if family=='student' else RNG.normal(size=(count,len(X)-13))
    for j in range(13,len(X)):residual[:,j]=phi*residual[:,j-1]+sigma*noise[:,j-13]
    yrep=(mu+residual)[:,13:];lo,mid,hi=np.quantile(yrep,[.025,.5,.975])
    prior_rows.append({'family':family,'lower95':lo,'median':mid,'upper95':hi,'fraction_abs_above5':np.mean(np.abs(yrep)>5)})
    plt.hist(yrep.ravel(),bins=np.linspace(-5,5,81),histtype='step',label=family)
plt.title('Illustrative US prior prediction conditional on the 13-month initial history');plt.xlabel('Monthly inflation %');plt.legend()
save_figure('prior_predictive');display(pd.DataFrame(prior_rows).round(4))
pd.DataFrame(prior_rows).to_csv(OUT/'prior_predictive_summary.csv',index=False)
print('Displayed bins stop at ±5; the table reports tail mass. Only the conditioning prefix supplies outcomes here.')
# %% [markdown]
# ## 6. Synthetic validation, separate from real observations
# True beta=.15, gamma=.25, scale=.15 and AR1 persistence=.4. We fit separately generated Gaussian
# and Student-t datasets with/without demand. The known simulation is not merged with measured data.
# Independent tests verify the missing-state Gaussian conditional against an analytic covariance model
# and verify the new Student-t kernel matches the established kernel on a complete calendar.
# Forty additional independent synthetic datasets cross weak/strong confounding and low/high persistence;
# their existing results use the same complete-calendar conditional Student-t AR1 likelihood.
# %%
simulation=[];parameter_recovery=[]
for family in ['gaussian','student']:
    synthetic,truth=simulation_data(family)
    for adjusted in [False,True]:
        fit=fit_combined(synthetic,family=family,adjusted=adjusted,label=f'SIM/{family}/adjusted={adjusted}')
        q=fit['diagnostics'].query("parameter == 'beta'").iloc[0]
        simulation.append({'family':family,'adjusted':adjusted,'truth':.15,'beta':q['mean'],'lower95':q.lower95,'upper95':q.upper95})
        if adjusted:
            table=fit['diagnostics'].copy();table['truth']=truth;table['family']=family
            table['covered']=(table.truth>=table.lower95)&(table.truth<=table.upper95)
            parameter_recovery.append(table)
display(pd.DataFrame(simulation).round(4))
recovery=pd.concat(parameter_recovery,ignore_index=True)
recovery.to_csv(OUT/'synthetic_parameter_recovery.csv',index=False)
display(recovery[recovery.parameter.isin(['beta','gamma','sigma','kappa1'])][['family','parameter','truth','mean','lower95','upper95','covered']].round(4))
stress=pd.read_csv(ROOT/'results/strengthening/simulation_summary.csv')
display(stress.round(4))
display(Image(filename=str(ROOT/'results/strengthening/simulation_recovery.png')))
print('Ten replications per scenario give imprecise coverage; Wilson Monte Carlo intervals are reported.')
# %% [markdown]
# ## 7. Main model fitting — January 2000–October 2026
# **This cell trains the main real-data models.** Each call fits one country, family and adjustment
# combination using the entire combined calendar. Unchanged code/data/settings reuse a recorded local
# posterior cache. Latent US monthly outcomes are sampled internally; they remain missing in the dataset.
# %%
fits={};estimate_rows=[];diagnostic_tables=[]
for iso,g in groups.items():
    for family in ['gaussian','student']:
        for adjusted in [False,True]:
            fit=fit_combined(g,family=family,adjusted=adjusted,label=f'{iso}/full/{family}/adjusted={adjusted}')
            fits[(iso,family,adjusted)]=fit
            q=fit['diagnostics'].query("parameter == 'beta'").iloc[0]
            estimate_rows.append({'iso3':iso,'family':family,'adjusted':adjusted,'beta':q['mean'],
                'lower95':q.lower95,'upper95':q.upper95,'observed_after_prefix':fit['n_observed'],
                'latent_outcomes':len(fit['missing']),'last_month':END})
            d=fit['diagnostics'].copy();d['iso3']=iso;d['family']=family;d['adjusted']=adjusted;diagnostic_tables.append(d)
estimates=pd.DataFrame(estimate_rows);diagnostics=pd.concat(diagnostic_tables,ignore_index=True)
estimates.to_csv(OUT/'country_estimates.csv',index=False);diagnostics.to_csv(OUT/'main_sampling_diagnostics.csv',index=False)
assert diagnostics.sampling_pass.all()
display(estimates.round(4))
display(diagnostics.groupby(['iso3','family','adjusted']).agg(max_Rhat=('r_hat','max'),min_bulk_ESS=('ess_bulk','min'),min_tail_ESS=('ess_tail','min')))
fig,axes=plt.subplots(1,2,figsize=(12,4),sharex=True,sharey=True)
for ax,family in zip(axes,['gaussian','student']):
    for offset,adjusted in [(-.1,False),(.1,True)]:
        q=estimates[estimates.family.eq(family)&estimates.adjusted.eq(adjusted)].set_index('iso3').loc[list(COUNTRIES)]
        ax.errorbar(q.beta,np.arange(4)+offset,xerr=[q.beta-q.lower95,q.upper95-q.beta],fmt='o',label='adjust demand' if adjusted else 'omit demand')
    ax.axvline(0,color='grey');ax.set_title(family+' — full combined period');ax.set_xlabel('Oil coefficient and 95% credible interval')
axes[0].set_yticks(range(4),list(COUNTRIES.values()));axes[0].legend();save_figure('country_coefficients')
# %% [markdown]
# ## 8. Posterior evaluation, forecasting and robustness
# The following suite recomputes every real-data sensitivity on the combined calendar. Rolling training
# cutoffs are December 2009, 2014 and 2019; disjoint evaluation windows end December 2014, December
# 2019 and **October 2026**. Forecasts condition on available lagged actual residuals without refitting
# coefficients on test outcomes. A separate pre-2025 training sample evaluates recent events.
# These are retrospective revised-data predictions, not real-time forecasts or an untouched confirmatory test.
# %%
analysis=run_combined()
checks=analysis['innovation_checks'];predictive=analysis['predictive_checks']
display(checks.round(4));display(predictive.round(4))
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
for ax,(iso,g) in zip(axes.flat,groups.items()):
    mu,rep=analysis['replicated'][(iso,'student')]
    dates=g.date.iloc[13:]
    ax.fill_between(dates,*np.quantile(rep[:,13:],[.025,.975],axis=0),alpha=.15,label='95% predictive')
    ax.fill_between(dates,*np.quantile(mu[:,13:],[.025,.975],axis=0),alpha=.4,label='95% conditional mean')
    ax.plot(g.date,g.monthly,color='black',linewidth=.7,label='Observed')
    ax.set_title(COUNTRIES[iso]);full_period_axis(ax);ax.set_ylabel('Monthly inflation %')
axes[0,0].legend(fontsize=7);fig.suptitle('Student-t AR1 posterior checks: combined period');save_figure('posterior_predictive')
fig,axes=plt.subplots(2,2,figsize=(11,6),sharex=True,sharey=True)
innovation_series={}
for ax,(iso,g) in zip(axes.flat,groups.items()):
    e=innovations(fits[(iso,'student',True)]);innovation_series[iso]=e
    acf_values=[pd.Series(e).autocorr(lag=k) for k in range(1,25)]
    ax.stem(range(1,25),acf_values);ax.set_title(COUNTRIES[iso]);ax.set_xlabel('Calendar lag (months)')
fig.suptitle('Calendar-aligned innovation correlations; unavailable pairs omitted');save_figure('innovation_ACF')
display(pd.DataFrame(innovation_series).corr().round(3))
display(Markdown('Ljung–Box statistics use the **longest contiguous observed innovation run**, whose dates and size are shown above. We never compress the US gap into adjacent months. These approximate diagnostics do not certify adequacy.'))
# %%
forecast=analysis['rolling_predictions'];aggregate=analysis['rolling_aggregate']
display(analysis['rolling_metrics'].round(4));display(aggregate.round(3))
display(Markdown('**Simple benchmarks and AR1 forecasts on identical eligible months within each window**'))
display(analysis['benchmark_metrics'].round(4))
last_window=forecast.window.iloc[-1]
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
for ax,iso in zip(axes.flat,COUNTRIES):
    for oil in [False,True]:
        q=forecast[forecast.iso3.eq(iso)&forecast.kind.eq('ar1')&forecast.oil.eq(oil)&forecast.window.eq(last_window)]
        ax.plot(q.date,q.prediction,label='with oil' if oil else 'without oil')
    ax.plot(q.date,q.actual,color='black',linewidth=.8,label='Observed');ax.set_title(COUNTRIES[iso]);ax.set_xlim(q.date.min(),END)
axes[0,0].legend(fontsize=7);fig.suptitle('Fixed through-2019 training: one-step predictions, 2020–October 2026');save_figure('holdout_predictions')
fig,axes=plt.subplots(1,2,figsize=(12,5))
labels=[r.iso3+' '+r.kind for r in aggregate.itertuples()];positions=np.arange(len(labels))
for ax,center,low,high,title in [(axes[0],'oil_gain_percent','oil_gain_lower95','oil_gain_upper95','Adding oil'),
    (axes[1],'structure_gain_percent','structure_lower95','structure_upper95','Structure versus AR1 with oil')]:
    ax.errorbar(aggregate[center],positions,xerr=[aggregate[center]-aggregate[low],aggregate[high]-aggregate[center]],fmt='o')
    ax.axvline(0,color='grey');ax.set_yticks(positions,labels);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('RMSE gain %, paired block interval')
fig.suptitle('Matched rolling evaluation through October 2026');save_figure('rolling_gains')
display(Markdown('Structure and oil comparisons use the same eligible calendar months across all six models within each country/window. Intervals resample paired circular 12-month blocks within windows, 2,000 replicates. All points are conditional on fitted forecasts; inspected model rankings remain exploratory.'))
# %%
recent=analysis['recent_predictions'];display(analysis['recent_metrics'].round(4));display(analysis['recent_gains'].round(3))
fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
for ax,iso in zip(axes.flat,COUNTRIES):
    q=recent[recent.iso3.eq(iso)&recent.oil]
    ax.plot(q.date,q.actual,color='black',label='Observed');ax.plot(q.date,q.prediction,label='Pre-2025 fit + oil')
    ax.fill_between(q.date,q.lower95,q.upper95,alpha=.2);ax.set_title(COUNTRIES[iso]);ax.set_xlim(q.date.min(),END)
axes[0,0].legend(fontsize=7);fig.suptitle('Recent-event evaluation: January 2025–October 2026');save_figure('recent_forecasts')
reference=analysis['historical_reference'];current=estimates[estimates.family.eq('student')&estimates.adjusted]
display(reference.merge(current,on='iso3',suffixes=('_through2024','_combined'))[
    ['iso3','beta_through2024','lower95_through2024','upper95_through2024','beta_combined','lower95_combined','upper95_combined']].round(4))
display(Markdown('The table above is the **historical reference comparison only**. Main estimates and charts use the combined period. October oil predicts November inflation under our lag-one question; that outcome is not yet observed.'))
# %%
sensitivity=analysis['coefficients'];display(sensitivity[sensitivity.experiment.ne('full')].round(4))
full_sensitivity=sensitivity[sensitivity.experiment.eq('full')]
display(full_sensitivity[full_sensitivity.variant.str.contains('student')].round(4))
prior=[]
for iso in COUNTRIES:
    f=fits[(iso,'student',True)];flat=f['chains'].reshape(-1,len(f['names']));p=f['p']
    for label,lw in [('beta SD .15',-.5*(flat[:,1]/.15)**2+.5*(flat[:,1]/.3)**2),
                     ('beta SD .60',-.5*(flat[:,1]/.6)**2+.5*(flat[:,1]/.3)**2),
                     ('phi transformed Beta(2,2)',np.log1p(-flat[:,p+1]**2))]:
        w=np.exp(lw-lw.max());w/=w.sum();order=np.argsort(flat[:,1])
        lo,hi=np.interp([.025,.975],np.cumsum(w[order]),flat[order,1])
        prior.append({'iso3':iso,'prior':label,'beta':w@flat[:,1],'lower95':lo,'upper95':hi,'weight_ESS':1/(w@w)})
prior=pd.DataFrame(prior);prior.to_csv(OUT/'prior_sensitivity.csv',index=False);display(prior.round(4))
display(Markdown('Prior-weight ESS measures weight concentration, not chain mixing. Alternative single-lag fits change treatment timing; they do not estimate a cumulative response. Pre/post-2020 fits have different precision and do not form a formal causal regime test. Fixed nu=3/5/10 changes variance as well as tail shape under a common scale prior.'))
# %% [markdown]
# ## 9. Findings, interpretation and limitations
# The following conclusions are generated from the combined-period fits and chronological evaluation.
# Sampling convergence is distinct from model adequacy, forecast usefulness and causal identification.
# %%
lines=['# Combined-period findings: January 2000–October 2026','',
    'Verified 1,288 calendar rows and 1,286 observed monthly inflation outcomes. Two US outcomes are integrated as latent states; raw gaps remain unchanged. All retained parameter and latent-state sampling checks pass.','',
    '## Main Student-t AR1 associations','']
for iso,name in COUNTRIES.items():
    a=estimates[estimates.iso3.eq(iso)&estimates.family.eq('student')&estimates.adjusted].iloc[0]
    b=estimates[estimates.iso3.eq(iso)&estimates.family.eq('student')&~estimates.adjusted].iloc[0]
    lines.append(f'- **{name}:** beta {a.beta:.4f} [{a.lower95:.4f}, {a.upper95:.4f}]; omitting demand gives {b.beta:.4f}.')
lines+=['','Units: monthly inflation percentage points per 10-percentage-point higher previous-month oil change. These are model-conditional associations, not identified geopolitical supply-shock effects.','',
    '## Matched chronological evaluation, 2010–October 2026','']
for iso,name in COUNTRIES.items():
    q=aggregate[aggregate.iso3.eq(iso)];best=q.sort_values('mean_log_score',ascending=False).iloc[0]
    base=q[q.kind.eq('ar1')].iloc[0]
    lines.append(f'- **{name}:** best inspected predictive score: {best.kind}; RMSE {best.RMSE:.4f}, AR1 {base.RMSE:.4f}. '
        f'Structure gain {best.structure_gain_percent:.2f}% [{best.structure_lower95:.2f}%, {best.structure_upper95:.2f}%]. '
        f'AR1 oil gain {base.oil_gain_percent:.2f}% [{base.oil_gain_lower95:.2f}%, {base.oil_gain_upper95:.2f}%], n={int(base.n_test)} matched months.')
lines+=['','## Recent-event forecasts: pre-2025 training only','']
for q in analysis['recent_gains'].itertuples():
    lines.append(f'- **{COUNTRIES[q.iso3]}:** oil RMSE gain {q.RMSE_gain_percent:.2f}%, {int(q.n_test)} scored months; '
        f'3-month block interval [{q.block3_lower95:.2f}%, {q.block3_upper95:.2f}%], '
        f'6-month block interval [{q.block6_lower95:.2f}%, {q.block6_upper95:.2f}%]. The short window gives limited precision.')
lines+=['','## What the checks establish','',
    'Repeated simulation shows demand adjustment reduces bias under the known simulated DAG, particularly with strong confounding. That does not establish sufficient real-world adjustment. Tail, lag, period and prior comparisons are reported without selecting coefficients by significance.',
    '', 'The common Student-t AR1 remains the reference; inspect seasonal/AR2 alternatives jointly by prediction and residual behavior. Shared country shocks, imperfect global-demand measurement, exchange rates, taxes, moving holidays and policy changes remain limitations. No formal causal country ranking follows.',
    '', 'All model families now condition on the same initial 13 months. Missing monthly outcomes are integrated, rather than assigned observed values; their known two-month index-level relation is not added to the likelihood. Residual diagnostics use available calendar pairs and a reported contiguous run, not a compressed gap.',
    '', 'Forecast coefficients use only their declared training outcomes. They condition on observed lagged residuals and revised source snapshots; this is not a real-time forecasting exercise. Model comparisons use inspected evaluation periods. October oil exposures require November inflation to assess their next-month response, so the newest event is not fully observed.',
    '', 'The three substantive variables are oil, inflation and prior activity. Seasonality/dynamics add statistical structure that must be justified under the course scope. Interest rates and identified supply shocks require an expanded design. Submission PDF pagination and the group screencast are separate deliverables.']
report='\n'.join(lines).rstrip()+'\n';(OUT/'FINDINGS.md').write_text(report,encoding='utf-8')
display(Markdown(report))
