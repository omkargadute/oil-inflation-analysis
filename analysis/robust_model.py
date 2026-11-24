"""Heavy-tailed AR regression, reproducible fitting, and predictive evaluation.

Student-t model: independent scaled initial residual and innovations, all t_nu.
The initial scale sigma/sqrt(1-phi^2) matches stationary variance, not the full
stationary marginal distribution of a t-innovation AR process.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import logsumexp
from scipy.stats import norm, t
from analysis.model import design, sample_model, summarize

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'.local-archive/final-model-cache'
CACHE.mkdir(parents=True, exist_ok=True)

def specification(frame, adjusted=True, oil=True):
    X, names, scales = design(frame, adjusted)
    if not oil:
        keep = [j for j,name in enumerate(names) if name != 'beta']
        X, scales = X[:,keep], scales[keep]
        names = [names[j] for j in keep]
    return X, names, scales

def point_loglik(y, X, draws, family='student', nu=5):
    """Draw-by-time log likelihood including the initial transformation Jacobian."""
    residual = np.asarray(y)[None,:] - draws[:,:-2]@X.T
    whitened = np.empty_like(residual)
    phi, sigma = draws[:,-1], draws[:,-2]
    whitened[:,0] = np.sqrt(1-phi**2)*residual[:,0]
    whitened[:,1:] = residual[:,1:] - phi[:,None]*residual[:,:-1]
    logp = (t.logpdf(whitened/sigma[:,None],df=nu) if family=='student'
            else norm.logpdf(whitened/sigma[:,None])) - np.log(sigma[:,None])
    logp[:,0] += .5*np.log1p(-phi**2)
    return logp

def sample_student(X,y,scales,seed,nu=5,warm=1500,keep=4000):
    """Normal/Gamma mixture Gibbs draws plus transformed sigma/phi Metropolis."""
    p, n = X.shape[1], len(y)
    P0 = np.diag(1/scales**2)
    initial = np.linalg.lstsq(X,y,rcond=None)[0]
    residual = y-X@initial
    ph0 = np.clip(np.corrcoef(residual[:-1],residual[1:])[0,1],-.8,.8)
    sg0 = max(np.std(residual[1:]-ph0*residual[:-1]),.03)*np.sqrt((nu-2)/nu)
    chains, acceptance = [], []
    for chain in range(4):
        rng = np.random.default_rng(seed+chain*100)
        b = initial.copy(); ls = np.log(sg0)+rng.normal(0,.2); eta = np.arctanh(ph0)+rng.normal(0,.2)
        step = np.array([.08,.15]); block = np.zeros(2); accepted = np.zeros(2); saved = []
        for iteration in range(warm+keep):
            phi, var = np.tanh(eta), np.exp(2*ls)
            residual = y-X@b
            v = np.r_[np.sqrt(1-phi**2)*residual[0], residual[1:]-phi*residual[:-1]]
            weights = rng.gamma((nu+1)/2, 2/(nu+v*v/var))
            Z = np.vstack([np.sqrt(1-phi**2)*X[0], X[1:]-phi*X[:-1]])
            q = np.r_[np.sqrt(1-phi**2)*y[0], y[1:]-phi*y[:-1]]
            L = np.linalg.cholesky((Z.T*weights)@Z/var+P0)
            mean = cho_solve((L,True), Z.T@(weights*q)/var, check_finite=False)
            b = mean+solve_triangular(L.T,rng.normal(size=p),lower=False,check_finite=False)
            residual = y-X@b
            first = weights[0]*residual[0]**2
            ra = weights[1:]@(residual[1:]**2)
            rb = weights[1:]@(residual[1:]*residual[:-1])
            rc = weights[1:]@(residual[:-1]**2)
            def conditional(a,e):
                if abs(a)>15 or abs(e)>8: return -np.inf
                ph = np.tanh(e)
                ss = (1-ph**2)*first+ra-2*ph*rb+ph**2*rc
                return -n*a+1.5*np.log1p(-ph**2)-.5*ss/np.exp(2*a)-.5*(np.exp(a)/.5)**2+a
            current = conditional(ls,eta)
            for j in range(2):
                a = ls+step[j]*rng.normal() if j==0 else ls
                e = eta+step[j]*rng.normal() if j==1 else eta
                proposed = conditional(a,e)
                if np.log(rng.uniform())<proposed-current:
                    ls,eta,current = a,e,proposed; block[j]+=1
                    if iteration>=warm: accepted[j]+=1
            if iteration<warm and (iteration+1)%100==0:
                step*=np.exp(np.clip(block/100-.44,-.2,.2)); block[:]=0
            if iteration>=warm: saved.append(np.r_[b,np.exp(ls),np.tanh(eta)])
        chains.append(saved); acceptance.append(accepted/keep)
    return np.asarray(chains),np.asarray(acceptance)

def fit_cached(frame,y=None,adjusted=True,oil=True,family='student',seed=1,label='',nu=5):
    X,names,scales = specification(frame,adjusted,oil)
    y = frame.monthly.to_numpy() if y is None else np.asarray(y)
    code = (ROOT/'analysis/model.py').read_bytes() + Path(__file__).read_bytes()
    settings = {'family':family,'nu':nu,'seed':seed,'warm':1500,'keep':4000}
    digest = hashlib.sha256(code+json.dumps(settings,sort_keys=True).encode())
    for value in [X,y,scales]: digest.update(np.asarray(value,dtype=np.float64).tobytes())
    key = digest.hexdigest(); file = CACHE/(key+'.npz')
    state = {'label':label,'key':key,'settings':settings,'status':'cached' if file.exists() else 'fitting',
             'time_utc':datetime.now(timezone.utc).isoformat()}
    (CACHE/'progress.json').write_text(json.dumps(state,indent=2))
    hit = file.exists()
    if hit:
        with np.load(file) as saved: chains,acceptance = saved['chains'],saved['acceptance']
    else:
        if family=='student':
            chains,acceptance = sample_student(X,y,scales,seed,nu=nu)
        elif family=='gaussian':
            chains,acceptance = sample_model(X,y,scales,seed=seed,warm=1500,keep=4000)
        else: raise ValueError(family)
        np.savez_compressed(file,chains=chains,acceptance=acceptance)
    table = summarize(chains,names)
    state.update(status='complete',cache_hit=hit,sampling_pass=bool(table.sampling_pass.all()))
    (CACHE/'progress.json').write_text(json.dumps(state,indent=2))
    with (CACHE/'run_log.jsonl').open('a') as log: log.write(json.dumps(state)+'\n')
    print(label, 'cache' if hit else 'fresh', 'sampling pass:',state['sampling_pass'],flush=True)
    return {'chains':chains,'diagnostics':table,'names':names,'scales':scales,'X':X,'y':y,
            'family':family,'nu':nu,'key':key,'cache_hit':hit,'acceptance':acceptance}

def posterior_replications(fit,seed=1,count=2000):
    rng = np.random.default_rng(seed)
    flat = fit['chains'].reshape(-1,len(fit['names'])+2)
    draws = flat[rng.choice(len(flat),min(count,len(flat)),replace=False)]
    mu = draws[:,:-2]@fit['X'].T
    noise = (rng.standard_t(fit['nu'],size=mu.shape) if fit['family']=='student' else rng.normal(size=mu.shape))
    noise *= draws[:,-2,None]
    residual = np.empty_like(noise)
    residual[:,0] = noise[:,0]/np.sqrt(1-draws[:,-1]**2)
    for j in range(1,mu.shape[1]): residual[:,j]=draws[:,-1]*residual[:,j-1]+noise[:,j]
    return mu,mu+residual

def one_step_forecast(fit,frame,start=240,seed=1):
    """Use training posterior and observed prior residuals; no refitting on test data."""
    X,_,_ = specification(frame,adjusted='gamma' in fit['names'],oil='beta' in fit['names'])
    flat = fit['chains'].reshape(-1,len(fit['names'])+2)[::4]
    mean = flat[:,:-2]@X.T
    previous = frame.monthly.to_numpy()[None,start-1:-1]-mean[:,start-1:-1]
    conditional = mean[:,start:]+flat[:,-1,None]*previous
    y = frame.monthly.to_numpy()[start:]
    residual = (y[None,:]-conditional)/flat[:,-2,None]
    logp = (t.logpdf(residual,df=fit['nu']) if fit['family']=='student' else norm.logpdf(residual))-np.log(flat[:,-2,None])
    rng = np.random.default_rng(seed)
    noise = (rng.standard_t(fit['nu'],size=conditional.shape) if fit['family']=='student' else rng.normal(size=conditional.shape))
    predictive = conditional + noise*flat[:,-2,None]
    lo,hi = np.quantile(predictive,[.025,.975],axis=0)
    prediction = conditional.mean(axis=0);error = y-prediction
    return {'prediction':prediction,'lower95':lo,'upper95':hi,'errors':error,
            'MAE':np.abs(error).mean(),'RMSE':np.sqrt(np.mean(error**2)),
            'coverage95':np.mean((y>=lo)&(y<=hi)),
            'mean_log_predictive_density':np.mean(logsumexp(logp,axis=0)-np.log(len(flat)))}

def simulation_data(family='student',seed=81):
    import pandas as pd
    rng = np.random.default_rng(seed); n = 300
    frame = pd.DataFrame({'date':pd.date_range('2000-01-01',periods=n,freq='MS')})
    frame['month'] = frame.date.dt.month
    C = np.zeros(n)
    for j in range(1,n): C[j]=.6*C[j-1]+rng.normal(0,.5)
    frame['C'] = C; frame['T'] = .8*C+rng.normal(0,.7,n)
    X,_,_ = specification(frame)
    truth = np.r_[.2,.15,.25,np.linspace(-.15,.15,11)]
    noise = (rng.standard_t(5,n) if family=='student' else rng.normal(size=n))*.15
    residual = np.empty(n); residual[0]=noise[0]/np.sqrt(1-.4**2)
    for j in range(1,n): residual[j]=.4*residual[j-1]+noise[j]
    frame['monthly'] = X@truth+residual
    return frame,np.r_[truth,.15,.4]
