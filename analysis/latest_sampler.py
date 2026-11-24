"""AR1 transition sampler on observed consecutive pairs.

Missing outcomes are zero-filled only in storage and initialization; no factor
uses a missing target or predecessor. Each observed run conditions on its first
residual, with the original first 13 months also conditioned out. Thus gaps restart
the conditional likelihood rather than joining non-adjacent calendar months.
Specialized from compiled_sampler.draw_chain; same priors and transition updates.
"""
import numpy as np
from numba import njit
from analysis.compiled_sampler import coefficients,target

@njit(cache=True)
def draw_observed_chain(X,y,scales,indices,nu,seed,warm,keep):
    np.random.seed(seed)
    kind=1;initial_months=13;p=X.shape[1];n=len(y);ne=len(indices)
    if kind==1:lags=np.array([1]);d=1
    elif kind==2:lags=np.array([1,2]);d=2
    else:lags=np.array([1,12,13]);d=2
    nl=len(lags);P0=np.diag(1/(scales*scales))
    b=np.linalg.solve(X.T@X+np.eye(p)*1e-6,X.T@y)
    residual=y-X@b
    phi=np.corrcoef(residual[1:],residual[:-1])[0,1]
    phi=min(.7,max(-.7,phi))
    sig=max(np.std(residual[1:]-phi*residual[:-1]),.04)*np.sqrt((nu-2)/nu)
    ls=np.log(sig)+np.random.normal(0,.15)
    eta=np.random.normal(0,.15,d);eta[0]+=np.arctanh(phi)
    step=np.full(d+1,.12);step[0]=.08
    block=np.zeros(d+1);accepted=np.zeros(d+1)
    saved=np.empty((keep,p+1+d));Z=np.empty((ne,p));q=np.empty(ne);weights=np.empty(ne)
    R=np.empty((ne,nl))
    for it in range(warm+keep):
        a=coefficients(kind,np.tanh(eta));var=np.exp(2*ls)
        for i in range(ne):
            j=indices[i]
            q[i]=y[j]
            for h in range(p):Z[i,h]=X[j,h]
            for k in range(nl):
                q[i]-=a[k]*y[j-lags[k]]
                for h in range(p):Z[i,h]-=a[k]*X[j-lags[k],h]
            error=q[i]-np.dot(Z[i],b)
            weights[i]=np.random.gamma((nu+1)/2,2/(nu+error*error/var))
        precision=Z.T@(weights.reshape(-1,1)*Z)/var+P0
        L=np.linalg.cholesky(precision)
        mean=np.linalg.solve(precision,Z.T@(weights*q)/var)
        b=mean+np.linalg.solve(L.T,np.random.normal(0,1,p))
        residual=y-X@b;current_res=residual[indices]
        for i in range(ne):
            for k in range(nl):R[i,k]=residual[indices[i]-lags[k]]
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
        if it<warm and (it+1)%100==0:
            for j in range(d+1):step[j]*=np.exp(min(.2,max(-.2,block[j]/100-.44)))
            block[:]=0
        if it>=warm:
            index=it-warm;saved[index,:p]=b;saved[index,p]=np.exp(ls);saved[index,p+1:]=np.tanh(eta)
    return saved,accepted/keep
