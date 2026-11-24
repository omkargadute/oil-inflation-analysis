"""Full-period conditional models, missing-state integration and calendar-safe evaluation."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import norm,t
from scipy.special import logsumexp
from statsmodels.stats.diagnostic import acorr_ljungbox
from numba import __version__ as numba_version
from analysis.robust_model import specification
from analysis.extension_model import INITIAL,ar_terms,diagnose
from analysis.combined_sampler import draw_combined

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.local-archive/combined-model-cache'
CACHE.mkdir(parents=True,exist_ok=True)

def fit_combined(frame,kind='ar1',family='student',nu=5,adjusted=True,oil=True,label=''):
    X,names,scales=specification(frame,adjusted,oil);y=frame.monthly.to_numpy()
    if not np.isfinite(X).all() or not np.isfinite(y[:INITIAL]).all():
        raise ValueError('Predictors and conditioning prefix must be observed')
    if not np.all(np.diff(frame.date.dt.to_period('M').astype('int64'))==1):
        raise ValueError('Preserve every calendar month, including missing outcomes')
    if len(y)<=INITIAL+len(names):raise ValueError('Insufficient analysis history')
    code={'ar1':1,'ar2':2,'sar12':12}[kind];d=1 if kind=='ar1' else 2
    missing=np.flatnonzero(~np.isfinite(y));p=len(names)
    names=names+['sigma']+[f'kappa{k+1}' for k in range(d)]+['latent_'+frame.date.iloc[j].strftime('%Y%m') for j in missing]
    h=hashlib.sha256(json.dumps([kind,family,nu,adjusted,oil,INITIAL,numba_version]).encode())
    for file in ['combined_model.py','combined_sampler.py','compiled_sampler.py','extension_model.py','robust_model.py','model.py']:
        h.update((ROOT/'analysis'/file).read_bytes())
    for value in [X,y,scales]:h.update(value.tobytes())
    base=h.hexdigest();seed=int(base[:8],16)
    for attempt,(warm,keep) in enumerate([(1200,2500),(2200,5500),(3500,11000)],1):
        key=hashlib.sha256((base+str((warm,keep))).encode()).hexdigest()
        file=CACHE/(key+'.npz');table_file=CACHE/(key+'.csv')
        if file.exists() and table_file.exists():
            with np.load(file) as z:chains=z['chains']
            table=pd.read_csv(table_file)
        else:
            chains=np.array([draw_combined(np.ascontiguousarray(X),np.ascontiguousarray(y),scales,code,
                nu if family=='student' else 0,(seed+j*100)%(2**32-1),warm,keep)[0] for j in range(4)])
            table=diagnose(chains,names)
            np.savez_compressed(file,chains=chains);table.to_csv(table_file,index=False)
        if table.sampling_pass.all():break
    if not table.sampling_pass.all():raise RuntimeError('Combined model failed sampling checks: '+label)
    print(label or (kind+'/'+family),'pass; attempt',attempt,flush=True)
    return {'chains':chains,'diagnostics':table,'names':names,'X':X,'y':y,'kind':kind,'family':family,
            'nu':nu,'adjusted':adjusted,'oil':oil,'key':key,'p':p,'d':d,'attempt':attempt,
            'n_factors':len(y)-INITIAL,'n_observed':int(np.isfinite(y[INITIAL:]).sum()),'missing':missing}

def forecast_combined(fit,frame,start,seed=42):
    X,_,_=specification(frame,fit['adjusted'],fit['oil']);p=fit['p'];d=fit['d']
    flat=fit['chains'].reshape(-1,len(fit['names']))[::4]
    mu=flat[:,:p]@X.T;y=frame.monthly.to_numpy();residual=y[None,:]-mu
    prediction=mu.copy();valid=np.isfinite(y)
    for i,theta in enumerate(flat):
        lags,a=ar_terms(fit['kind'],theta[p+1:p+1+d])
        for lag,value in zip(lags,a):
            prediction[i,lag:]+=value*residual[i,:-lag]
    lags,_=ar_terms(fit['kind'],flat[0,p+1:p+1+d])
    for lag in lags:valid &= np.r_[np.zeros(lag,dtype=bool),np.isfinite(y[:-lag])]
    valid &= frame.date.ge(pd.Timestamp(start)).to_numpy()
    has_prediction=np.isfinite(prediction).all(axis=0)
    has_prediction[:max(lags)]=False
    center=prediction.mean(axis=0);center[~has_prediction]=np.nan
    rng=np.random.default_rng(seed)
    noise=rng.standard_t(fit['nu'],prediction.shape) if fit['family']=='student' else rng.normal(size=prediction.shape)
    yp=prediction+noise*flat[:,p,None];lo=np.full(len(y),np.nan);hi=lo.copy()
    lo[has_prediction],hi[has_prediction]=np.quantile(yp[:,has_prediction],[.025,.975],axis=0)
    scaled=(y[None,valid]-prediction[:,valid])/flat[:,p,None]
    logp=(t.logpdf(scaled,df=fit['nu']) if fit['family']=='student' else norm.logpdf(scaled))-np.log(flat[:,p,None])
    scores=np.full(len(y),np.nan);scores[valid]=logsumexp(logp,axis=0)-np.log(len(flat))
    detail=pd.DataFrame({'date':frame.date,'actual':y,'prediction':center,'lower95':lo,'upper95':hi,
                         'eligible':valid,'log_score':scores})
    return detail[detail.date.ge(pd.Timestamp(start))].reset_index(drop=True)

def forecast_metrics(detail):
    q=detail[detail.eligible];e=q.actual-q.prediction
    return {'n_test':len(q),'RMSE':np.sqrt(np.mean(e**2)),'MAE':np.mean(np.abs(e)),
            'log_score':q.log_score.mean(),'coverage95':np.mean(q.actual.between(q.lower95,q.upper95))}

def replications(fit,count=800,seed=42):
    rng=np.random.default_rng(seed);p=fit['p'];d=fit['d'];n=len(fit['y'])
    flat=fit['chains'].reshape(-1,len(fit['names']))
    theta=flat[rng.choice(len(flat),count,replace=False)]
    mu=theta[:,:p]@fit['X'].T;residual=np.empty_like(mu);mean_residual=np.empty_like(mu)
    residual[:,:INITIAL]=fit['y'][:INITIAL]-mu[:,:INITIAL];mean_residual[:,:INITIAL]=residual[:,:INITIAL]
    for i,state in enumerate(theta):
        lags,a=ar_terms(fit['kind'],state[p+1:p+1+d])
        noise=rng.standard_t(fit['nu'],n-INITIAL) if fit['family']=='student' else rng.normal(size=n-INITIAL)
        for j in range(INITIAL,n):
            residual[i,j]=sum(v*residual[i,j-lag] for lag,v in zip(lags,a))+state[p]*noise[j-INITIAL]
            mean_residual[i,j]=sum(v*mean_residual[i,j-lag] for lag,v in zip(lags,a))
    return mu+mean_residual,mu+residual

def innovations(fit):
    mean=fit['chains'].mean(axis=(0,1));p=fit['p'];d=fit['d'];r=fit['y']-fit['X']@mean[:p]
    e=r.copy();lags,a=ar_terms(fit['kind'],mean[p+1:p+1+d])
    for lag,value in zip(lags,a):e[lag:]-=value*r[:-lag]
    e[:INITIAL]=np.nan
    return e

def innovation_checks(fit):
    e=innovations(fit);indices=np.flatnonzero(np.isfinite(e))
    runs=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1);run=max(runs,key=len)
    lb=acorr_ljungbox(e[run],lags=[12,24],model_df=fit['d'],return_df=True)
    result={'diagnostic_run_n':len(run),'run_first_index':run[0],'run_last_index':run[-1],
            'LB_p12':lb.lb_pvalue.loc[12],'LB_p24':lb.lb_pvalue.loc[24]}
    for lag in [1,12,24]:
        keep=np.isfinite(e[lag:])&np.isfinite(e[:-lag])
        result[f'calendar_ACF{lag}']=np.corrcoef(e[lag:][keep],e[:-lag][keep])[0,1]
    return result
