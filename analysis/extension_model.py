"""Matched conditional Student-t AR experiments; all condition on 13 initial months.

AR2 uses partial-autocorrelation coordinates; SAR12 uses (1-phi B)(1-Phi B^12).
Uniform(-1,1) priors on those coordinates guarantee stability. Regression priors
match the baseline; sigma is a Student-t scale. The conditioning prefix is fixed,
not assigned a falsely stationary multivariate-t initial distribution.
"""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import arviz as az
from scipy.linalg import cho_solve,solve_triangular
from scipy.special import logsumexp
from scipy.stats import t
from analysis.robust_model import specification

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.local-archive/strengthening-cache'
CACHE.mkdir(parents=True,exist_ok=True)
INITIAL=13

def ar_terms(kind,kappa):
    k=np.asarray(kappa)
    if kind=='ar1':return [1],np.array([k[0]])
    if kind=='ar2':return [1,2],np.array([k[0]*(1-k[1]),k[1]])
    if kind=='sar12':return [1,12,13],np.array([k[0],k[1],-k[0]*k[1]])
    raise ValueError(kind)

def conditional_loglike(y,X,b,sigma,kappa,kind,nu=5):
    residual=np.asarray(y)-X@b;lags,a=ar_terms(kind,kappa)
    e=residual[INITIAL:].copy()
    for lag,value in zip(lags,a):e-=value*residual[INITIAL-lag:len(y)-lag]
    return t.logpdf(e/sigma,df=nu)-np.log(sigma)

def sample_conditional(X,y,scales,kind,nu,seed,warm,keep):
    from analysis.compiled_sampler import draw_chain
    code={'ar1':1,'ar2':2,'sar12':12}[kind]
    chains=[];rates=[]
    for chain in range(4):
        draws,accept=draw_chain(np.ascontiguousarray(X),np.ascontiguousarray(y),scales,
                                code,nu,(seed+chain*100)%(2**32-1),warm,keep)
        chains.append(draws);rates.append(accept)
    return np.asarray(chains),np.asarray(rates)


def diagnose(chains,names):
    result=az.summary(az.from_dict(posterior={name:chains[:,:,j] for j,name in enumerate(names)}),round_to=6)
    result=result.rename_axis('parameter').reset_index()
    result['lower95']=np.quantile(chains,.025,axis=(0,1));result['upper95']=np.quantile(chains,.975,axis=(0,1))
    result['sampling_pass']=(result.r_hat<1.01)&(result.ess_bulk>400)&(result.ess_tail>400)
    return result

def fit_extension(frame,kind='ar1',nu=5,adjusted=True,oil=True,label='',simulation=False):
    X,names,scales=specification(frame,adjusted,oil);y=frame.monthly.to_numpy()
    if len(y)<=INITIAL+len(names):raise ValueError('Insufficient observations for the requested conditional model')
    if not np.all(np.diff(frame.date.dt.to_period('M').astype('int64'))==1):
        raise ValueError('Conditional AR fits require a consecutive monthly calendar')
    d=1 if kind=='ar1' else 2
    names=names+['sigma']+[f'kappa{j+1}' for j in range(d)]
    code=(Path(__file__).read_bytes()+(ROOT/'analysis/model.py').read_bytes()
          +(ROOT/'analysis/robust_model.py').read_bytes()+(ROOT/'analysis/compiled_sampler.py').read_bytes())
    from numba import __version__ as numba_version
    settings={'backend':'numba-'+numba_version,'kind':kind,'nu':nu,'adjusted':adjusted,'oil':oil,'initial':INITIAL}
    digest=hashlib.sha256(code+json.dumps(settings,sort_keys=True).encode())
    for value in [X,y,scales]:digest.update(np.asarray(value,dtype=np.float64).tobytes())
    base=digest.hexdigest();seed=int(base[:8],16)
    attempts=[(900,1800),(1500,3500),(2000,6500)] if simulation else [(1200,2500),(2000,5000),(2500,8000)]
    for attempt,(warm,keep) in enumerate(attempts):
        key=hashlib.sha256((base+f'/{warm}/{keep}').encode()).hexdigest();file=CACHE/(key+'.npz');table_file=CACHE/(key+'.csv')
        state={'label':label,'kind':kind,'nu':nu,'attempt':attempt+1,'warm':warm,'keep':keep,'key':key,
               'status':'cached' if file.exists() else 'fitting','time_utc':datetime.now(timezone.utc).isoformat()}
        (CACHE/'progress.json').write_text(json.dumps(state,indent=2))
        hit=file.exists() and table_file.exists()
        if hit:
            with np.load(file) as z:chains,rates=z['chains'],z['rates']
            table=pd.read_csv(table_file)
        else:
            chains,rates=sample_conditional(X,y,scales,kind,nu,seed,warm,keep)
            table=diagnose(chains,names)
            np.savez_compressed(file,chains=chains,rates=rates);table.to_csv(table_file,index=False)
        state.update(status='complete',sampling_pass=bool(table.sampling_pass.all()),cache_hit=hit)
        with (CACHE/'log.jsonl').open('a') as log:log.write(json.dumps(state)+'\n')
        if table.sampling_pass.all():break
    if not table.sampling_pass.all():raise RuntimeError(f'Sampling did not pass after extension attempts: {label}')
    (CACHE/'progress.json').write_text(json.dumps(state,indent=2))
    print(label,'cache' if hit else 'fresh','pass',flush=True)
    return {'chains':chains,'diagnostics':table,'names':names,'X':X,'y':y,'kind':kind,'nu':nu,
            'adjusted':adjusted,'oil':oil,'n_eff':len(y)-INITIAL,'key':key,'attempt':attempt+1}

def forecast_extension(fit,frame,start,seed=1):
    X,_,_=specification(frame,fit['adjusted'],fit['oil']);p=X.shape[1]
    flat=fit['chains'].reshape(-1,len(fit['names']))[::4];b=flat[:,:p];sigma=flat[:,p];kappa=flat[:,p+1:]
    mu=b@X.T;r=frame.monthly.to_numpy()[None,:]-mu;conditional=mu[:,start:].copy()
    for j,k in enumerate(kappa):
        lags,a=ar_terms(fit['kind'],k)
        for lag,value in zip(lags,a):conditional[j]+=value*r[j,start-lag:len(frame)-lag]
    y=frame.monthly.to_numpy()[start:];err=y-conditional.mean(axis=0)
    logp=t.logpdf((y-conditional)/sigma[:,None],df=fit['nu'])-np.log(sigma[:,None])
    rng=np.random.default_rng(seed);yp=conditional+rng.standard_t(fit['nu'],conditional.shape)*sigma[:,None]
    lo,hi=np.quantile(yp,[.025,.975],axis=0)
    return {'prediction':conditional.mean(axis=0),'errors':err,'lower95':lo,'upper95':hi,
            'MAE':np.abs(err).mean(),'RMSE':np.sqrt(np.mean(err*err)),
            'coverage95':np.mean((y>=lo)&(y<=hi)),
            'log_score':np.mean(logsumexp(logp,axis=0)-np.log(len(flat)))}

def replicate_extension(fit,count=1000,seed=1):
    rng=np.random.default_rng(seed);p=fit['X'].shape[1]
    flat=fit['chains'].reshape(-1,len(fit['names']));draw=flat[rng.choice(len(flat),count,replace=False)]
    mu=draw[:,:p]@fit['X'].T;n=mu.shape[1]
    residual=np.empty_like(mu);mean_residual=np.empty_like(mu)
    residual[:,:INITIAL]=fit['y'][:INITIAL]-mu[:,:INITIAL];mean_residual[:,:INITIAL]=residual[:,:INITIAL]
    for i,theta in enumerate(draw):
        lags,a=ar_terms(fit['kind'],theta[p+1:]);noise=rng.standard_t(fit['nu'],n-INITIAL)*theta[p]
        for j in range(INITIAL,n):
            residual[i,j]=sum(v*residual[i,j-lag] for lag,v in zip(lags,a))+noise[j-INITIAL]
            mean_residual[i,j]=sum(v*mean_residual[i,j-lag] for lag,v in zip(lags,a))
    return (mu+mean_residual)[:,INITIAL:],(mu+residual)[:,INITIAL:]

def point_innovations(fit):
    mean=fit['chains'].mean(axis=(0,1));p=fit['X'].shape[1];r=fit['y']-fit['X']@mean[:p]
    lags,a=ar_terms(fit['kind'],mean[p+1:]);e=r[INITIAL:].copy()
    for lag,v in zip(lags,a):e-=v*r[INITIAL-lag:len(r)-lag]
    return e,mean[p]
