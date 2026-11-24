"""Segmented conditional AR1 fits and calendar-safe recent forecasts."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.special import logsumexp
from numba import __version__ as numba_version
from analysis.robust_model import specification
from analysis.extension_model import diagnose,INITIAL
from analysis.latest_sampler import draw_observed_chain

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.local-archive/latest-model-cache'
CACHE.mkdir(parents=True,exist_ok=True)

def observed_indices(frame):
    y=frame.monthly.to_numpy()
    valid=np.isfinite(y)&np.r_[False,np.isfinite(y[:-1])]
    valid[:INITIAL]=False
    return np.flatnonzero(valid)

def fit_latest(frame,oil=True,label=''):
    X,names,scales=specification(frame,oil=oil);y=frame.monthly.to_numpy()
    if not np.isfinite(X).all():raise ValueError('Missing model predictors')
    if not np.all(np.diff(frame.date.dt.to_period('M').astype('int64'))==1):
        raise ValueError('Calendar must retain missing months')
    indices=observed_indices(frame)
    if len(indices)<=len(names)+2:raise ValueError('Insufficient observed transitions')
    names=names+['sigma','kappa1']
    h=hashlib.sha256(('nu=5;conditional-AR1;'+numba_version+str(oil)).encode())
    for file in ['latest_model.py','latest_sampler.py','compiled_sampler.py','extension_model.py','robust_model.py','model.py']:
        h.update((ROOT/'analysis'/file).read_bytes())
    for value in [X,y,scales,indices]:h.update(value.tobytes())
    base=h.hexdigest();seed=int(base[:8],16)
    for warm,keep in [(1200,2500),(2000,5000),(3000,10000)]:
        key=hashlib.sha256((base+str((warm,keep))).encode()).hexdigest()
        file=CACHE/(key+'.npz');table_file=CACHE/(key+'.csv')
        if file.exists() and table_file.exists():
            with np.load(file) as z:chains=z['chains']
            table=pd.read_csv(table_file)
        else:
            chains=np.array([draw_observed_chain(np.ascontiguousarray(X),np.nan_to_num(y),scales,indices,5,
                (seed+j*100)%(2**32-1),warm,keep)[0] for j in range(4)])
            table=diagnose(chains,names)
            np.savez_compressed(file,chains=chains);table.to_csv(table_file,index=False)
        if table.sampling_pass.all():break
    if not table.sampling_pass.all():raise RuntimeError('Latest sampling diagnostics failed: '+label)
    print('Current-data fit passed:',label,flush=True)
    return {'chains':chains,'diagnostics':table,'names':names,'X':X,'y':y,'indices':indices,
            'oil':oil,'key':key,'n_eff':len(indices)}

def forecast_latest(fit,frame,start='2025-01-01',seed=42):
    X,_,_=specification(frame,oil=fit['oil']);p=X.shape[1]
    flat=fit['chains'].reshape(-1,len(fit['names']))[::4]
    mu=flat[:,:p]@X.T;y=frame.monthly.to_numpy();residual=y[None,:]-mu
    prediction=mu[:,1:]+flat[:,-1,None]*residual[:,:-1]
    prediction=np.column_stack([np.full(len(flat),np.nan),prediction])
    valid=np.isfinite(y)&np.isfinite(prediction).all(axis=0)&frame.date.ge(start).to_numpy()
    rng=np.random.default_rng(seed)
    draws=prediction+rng.standard_t(5,prediction.shape)*flat[:,p,None]
    # Quantile only over available conditional predictions, avoiding all-NaN slices.
    center=prediction.mean(axis=0);lo=np.full(len(y),np.nan);hi=lo.copy()
    has_prediction=np.isfinite(center)
    lo[has_prediction],hi[has_prediction]=np.quantile(draws[:,has_prediction],[.025,.975],axis=0)
    logscore=np.full(len(y),np.nan)
    logp=t.logpdf((y[None,valid]-prediction[:,valid])/flat[:,p,None],df=5)-np.log(flat[:,p,None])
    logscore[valid]=logsumexp(logp,axis=0)-np.log(len(flat))
    detail=pd.DataFrame({'date':frame.date,'actual':y,'prediction':center,'lower95':lo,'upper95':hi,
                         'eligible':valid,'log_score':logscore})
    detail['exclusion_reason']=np.where(~np.isfinite(y),'missing monthly outcome',
                                  np.where(~has_prediction,'missing previous monthly residual',''))
    return detail[detail.date.ge(start)].reset_index(drop=True)
