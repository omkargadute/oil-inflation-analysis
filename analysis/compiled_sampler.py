"""Compiled transition kernel for the conditional Student-t models.

The chain state and priors are unchanged; the seeded RNG is Numba's independent
legacy NumPy stream per chain. Compilation and backend version enter cache keys.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def coefficients(kind,k):
    if kind==1:return np.array([k[0]])
    if kind==2:return np.array([k[0]*(1-k[1]),k[1]])
    return np.array([k[0],k[1],-k[0]*k[1]])

@njit(cache=True)
def target(ls,eta,kind,n,rr,h,G):
    if abs(ls)>15 or np.max(np.abs(eta))>7:return -np.inf
    k=np.tanh(eta);a=coefficients(kind,k)
    ss=rr-2*np.dot(a,h)+np.dot(a,np.dot(G,a))
    return -n*ls-.5*ss/np.exp(2*ls)-.5*(np.exp(ls)/.5)**2+ls+np.log(1-k*k).sum()

@njit(cache=True)
def draw_chain(X,y,scales,kind,nu,seed,warm,keep):
    np.random.seed(seed)
    initial_months=13;p=X.shape[1];n=len(y);ne=n-initial_months
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
            j=i+initial_months
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
        if it<warm and (it+1)%100==0:
            for j in range(d+1):step[j]*=np.exp(min(.2,max(-.2,block[j]/100-.44)))
            block[:]=0
        if it>=warm:
            index=it-warm;saved[index,:p]=b;saved[index,p]=np.exp(ls);saved[index,p+1:]=np.tanh(eta)
    return saved,accepted/keep
