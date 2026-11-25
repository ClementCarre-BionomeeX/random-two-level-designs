import numpy as np, warnings; from sklearn.linear_model import Lasso; from common import *
warnings.filterwarnings('ignore')
p=100; P=pairs(p); N=p+len(P)
def truth(rng, theta=0.5, n_main=5, n_int=5):
    th=np.zeros(N); S=np.r_[rng.choice(p,n_main,replace=False), p+rng.choice(len(P),n_int,replace=False)]
    th[S]=theta*rng.choice([-1,1],len(S)); return th,S
def debiased_z(F,y,sigma=1.0):
    m=len(y); lam=sigma*np.sqrt(2*np.log(F.shape[1])/m)
    yc=y-y.mean(); Fc=F-F.mean(0)
    th=Lasso(alpha=lam,fit_intercept=False,max_iter=20000).fit(Fc,yc).coef_
    r=yc-Fc@th; df=(th!=0).sum()
    d=th+Fc.T@r/(m-df); se=np.linalg.norm(r)/(np.sqrt(m)*(1-df/m))/np.sqrt(m)
    return d/se, th
def topk_hits(score,S,k):
    top=np.argsort(-score)[:k]; return np.isin(S,top)
