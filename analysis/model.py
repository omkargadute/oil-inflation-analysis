"""Seasonal Gaussian AR(1) regression and Metropolis-within-Gibbs sampling."""
import numpy as np
import pandas as pd
import arviz as az
from scipy.linalg import helmert, cho_solve, solve_triangular

SEASON = helmert(12, full=False).T

def design(d, adjusted=True):
    columns = [np.ones(len(d)), d['T'].to_numpy()]
    names = ['alpha', 'beta']
    scales = [.5, .3]
    if adjusted:
        columns.append(d['C'].to_numpy())
        names.append('gamma'); scales.append(.5)
    X = np.column_stack(columns + [SEASON[d.month.to_numpy()-1]])
    return X, names + [f'season_{i+1}' for i in range(11)], np.array(scales + [.5]*11)

def log_target(z, X, y, scales):
    p = X.shape[1]; b = z[:p]; ls = z[p]; phi = np.tanh(z[p+1]); sigma = np.exp(ls)
    r = y-X@b
    ss = (1-phi**2)*r[0]**2 + np.sum((r[1:]-phi*r[:-1])**2)
    return (-len(y)*ls + .5*np.log1p(-phi**2) - .5*ss/sigma**2
            - .5*np.sum((b/scales)**2) - .5*(sigma/.5)**2 + ls + np.log1p(-phi**2))

def sample_model(X, y, scales, seed=22, warm=2000, keep=6000):
    """Four chains: exact coefficient Gibbs draws, scalar transformed sigma/phi MH draws.

    Proposal scale adaptation stops after warmup. Priors are independent of sigma.
    The stationary initial observation is included in the likelihood and Gibbs precision.
    """
    p = X.shape[1]; prior_precision = np.diag(1/scales**2)
    first = np.outer(X[0],X[0]); A = X[1:].T@X[1:]
    B = X[1:].T@X[:-1]; D = X[:-1].T@X[:-1]
    h0 = X[0]*y[0]; h1 = X[1:].T@y[1:]
    hcross = X[1:].T@y[:-1] + X[:-1].T@y[1:]; hlag = X[:-1].T@y[:-1]
    initial = np.linalg.lstsq(X,y,rcond=None)[0]; resid = y-X@initial
    phi0 = np.clip(pd.Series(resid).autocorr(),-.8,.8)
    sigma0 = max(np.std(resid[1:]-phi0*resid[:-1]),.03)
    chains = []; acceptance = []
    for chain in range(4):
        rng = np.random.default_rng(seed+chain*100)
        ls = np.log(sigma0)+rng.normal(0,.2); eta = np.arctanh(phi0)+rng.normal(0,.2)
        step = np.array([.08,.15]); block = np.zeros(2); accepted = np.zeros(2); saved = []
        for it in range(warm+keep):
            phi = np.tanh(eta); var = np.exp(2*ls)
            G = (1-phi**2)*first+A-phi*(B+B.T)+phi**2*D
            h = (1-phi**2)*h0+h1-phi*hcross+phi**2*hlag
            L = np.linalg.cholesky(G/var+prior_precision)
            mean = cho_solve((L,True),h/var,check_finite=False)
            b = mean + solve_triangular(L.T,rng.normal(size=p),lower=False,check_finite=False)
            r = y-X@b; rfirst = r[0]**2
            ra = r[1:]@r[1:]; rb = r[1:]@r[:-1]; rc = r[:-1]@r[:-1]
            def conditional(a,e):
                if abs(a)>15 or abs(e)>8:
                    return -np.inf
                ph = np.tanh(e); ss = (1-ph**2)*rfirst+ra-2*ph*rb+ph**2*rc
                return (-len(y)*a+1.5*np.log1p(-ph**2)-.5*ss/np.exp(2*a)
                        -.5*(np.exp(a)/.5)**2+a)
            current = conditional(ls,eta)
            for j in range(2):
                a = ls+step[j]*rng.normal() if j==0 else ls
                e = eta+step[j]*rng.normal() if j==1 else eta
                proposed = conditional(a,e)
                if np.log(rng.uniform())<proposed-current:
                    ls,eta,current = a,e,proposed; block[j]+=1
                    if it>=warm:
                        accepted[j]+=1
            if it<warm and (it+1)%100==0:
                step *= np.exp(np.clip(block/100-.44,-.2,.2)); block[:]=0
            if it>=warm:
                saved.append(np.r_[b,np.exp(ls),np.tanh(eta)])
        chains.append(saved); acceptance.append(accepted/keep)
    return np.asarray(chains), np.asarray(acceptance)

def summarize(chains,names):
    names = names + ['sigma','phi']
    posterior = {name:chains[:,:,j] for j,name in enumerate(names)}
    table = az.summary(az.from_dict(posterior=posterior),hdi_prob=.95,round_to=6)
    table = table.rename_axis('parameter').reset_index()
    table['lower95'] = np.quantile(chains,.025,axis=(0,1))
    table['upper95'] = np.quantile(chains,.975,axis=(0,1))
    table['sampling_pass'] = (table.r_hat<1.01)&(table.ess_bulk>400)&(table.ess_tail>400)
    return table
