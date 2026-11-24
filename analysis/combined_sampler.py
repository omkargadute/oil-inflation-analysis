"""Conditional AR regression with Bayesian integration of missing monthly outcomes.

nu=0 selects Gaussian innovations; nu>2 selects Student-t. The first 13 observed
months are conditioned on. Unknown later monthly outcomes receive exact Gaussian
Gibbs updates conditional on the scale mixture, parameters and observed outcomes.
They remain missing in the released dataset. No calendar segment is compressed.
"""
import numpy as np
from numba import njit
from analysis.compiled_sampler import coefficients,target

@njit(cache=True)
def latent_conditional(y,X,b,sigma,weights,kind,kappa,missing):
    n=len(y);initial=13
    if kind==1:lags=np.array([1])
    elif kind==2:lags=np.array([1,2])
    else:lags=np.array([1,12,13])
    a=coefficients(kind,kappa);r=y-X@b;e=r[initial:].copy()
    H=np.zeros((n-initial,len(missing)))
    for i in range(n-initial):
        j=i+initial
        for k in range(len(lags)):e[i]-=a[k]*r[j-lags[k]]
        for m in range(len(missing)):
            if j==missing[m]:H[i,m]+=1.
            for k in range(len(lags)):
                if j-lags[k]==missing[m]:H[i,m]-=a[k]
    Q=H.T@(weights.reshape(-1,1)*H)
    mean=y[missing]-np.linalg.solve(Q,H.T@(weights*e))
    return mean,Q

@njit(cache=True)
def draw_combined(X,y_input,scales,kind,nu,seed,warm,keep):
    np.random.seed(seed)
    missing=np.flatnonzero(~np.isfinite(y_input));nm=len(missing)
    y=np.nan_to_num(y_input.copy())
    initial_months=13;p=X.shape[1];n=len(y);ne=n-initial_months
    if kind==1:lags=np.array([1]);d=1
    elif kind==2:lags=np.array([1,2]);d=2
    else:lags=np.array([1,12,13]);d=2
    nl=len(lags);P0=np.diag(1/(scales*scales))
    b=np.linalg.solve(X.T@X+np.eye(p)*1e-6,X.T@y)
    residual=y-X@b
    phi=np.corrcoef(residual[1:],residual[:-1])[0,1]
    phi=min(.7,max(-.7,phi))
    sig=max(np.std(residual[1:]-phi*residual[:-1]),.04)*(np.sqrt((nu-2)/nu) if nu>2 else 1.)
    ls=np.log(sig)+np.random.normal(0,.15)
    eta=np.random.normal(0,.15,d);eta[0]+=np.arctanh(phi)
    step=np.full(d+1,.12);step[0]=.08
    block=np.zeros(d+1);accepted=np.zeros(d+1)
    saved=np.empty((keep,p+1+d+nm));Z=np.empty((ne,p));q=np.empty(ne);weights=np.empty(ne)
    R=np.empty((ne,nl))
    for it in range(warm+keep):
        a=coefficients(kind,np.tanh(eta));var=np.exp(2*ls)
        for i in range(ne):
            j=i+initial_months
            q[i]=y[j]
            for h in range(p):Z[i,h]=X[j,h]
            for k in range(nl):
                q[i]-=a[k]*y[j-lags[k]]
                for h in range(p):Z[i,h]-=a[k]*X[j-lags[k],h]
            error=q[i]-np.dot(Z[i],b)
            weights[i]=np.random.gamma((nu+1)/2,2/(nu+error*error/var)) if nu>2 else 1.
        precision=Z.T@(weights.reshape(-1,1)*Z)/var+P0
        L=np.linalg.cholesky(precision)
        mean=np.linalg.solve(precision,Z.T@(weights*q)/var)
        b=mean+np.linalg.solve(L.T,np.random.normal(0,1,p))
        residual=y-X@b;current_res=residual[initial_months:]
        for i in range(ne):
            for k in range(nl):R[i,k]=residual[i+initial_months-lags[k]]
        rr=np.dot(weights,current_res*current_res);h=R.T@(weights*current_res);G=R.T@(weights.reshape(-1,1)*R)
        current=target(ls,eta,kind,ne,rr,h,G)
        for j in range(d+1):
            proposal_ls=ls;proposal_eta=eta.copy()
            if j==0:proposal_ls+=step[j]*np.random.normal()
            else:proposal_eta[j-1]+=step[j]*np.random.normal()
            value=target(proposal_ls,proposal_eta,kind,ne,rr,h,G)
            if np.log(np.random.random())<value-current:
                ls=proposal_ls;eta=proposal_eta;current=value;block[j]+=1
                if it>=warm:accepted[j]+=1
        if nm:
            mean_missing,Q=latent_conditional(y,X,b,np.exp(ls),weights,kind,np.tanh(eta),missing)
            Lm=np.linalg.cholesky(Q)
            y[missing]=mean_missing+np.exp(ls)*np.linalg.solve(Lm.T,np.random.normal(0,1,nm))
        if it<warm and (it+1)%100==0:
            for j in range(d+1):step[j]*=np.exp(min(.2,max(-.2,block[j]/100-.44)))
            block[:]=0
        if it>=warm:
            index=it-warm;saved[index,:p]=b;saved[index,p]=np.exp(ls);saved[index,p+1:p+1+d]=np.tanh(eta)
            if nm:saved[index,p+1+d:]=y[missing]
    return saved,accepted/keep
