# Test 1: ranking (true effect in top k) vs Bonferroni, as a function of m. theta=0.5 sigma.
import numpy as np, sys; from multiprocessing import Pool; from scipy.stats import norm; from rank_common import *
K=[10,20,50]; theta=0.5
def one(a):
    m,rep=a; rng=np.random.default_rng(10007*m+rep)
    th,S=truth(rng,theta); F=features(rademacher(m,p,rng),P); y=F@th+rng.standard_normal(m)
    z,lc=debiased_z(F,y); az=np.abs(z); thr=norm.ppf(1-0.05/(2*N))
    null=np.ones(N,bool); null[S]=False
    out={'bonf':(az[S]>thr), 'fwer':float((az[null]>thr).any())}
    for k in K: out[f'z{k}']=topk_hits(az,S,k)
    sc=np.abs(lc)+1e-9*az  # lasso magnitude, ties broken by |z|
    for k in K: out[f'l{k}']=topk_hits(sc,S,k)
    return m,out
if __name__=='__main__':
    ms=[50,75,100,150,200,300]
    with Pool(4) as pool: res=pool.map(one,[(m,r) for m in ms for r in range(100)])
    thr=norm.ppf(1-0.05/(2*N))
    print("m lambda | Bonf power(main,int) FWER | top10 top20 top50 (|z|) | top10 top20 top50 (|lasso|) | oracle top10 top50 Bonf")
    for m in ms:
        R=[o for mm,o in res if mm==m]; lam=theta*np.sqrt(m)
        def rec(key): a=np.array([o[key] for o in R]); return a.mean()
        bm=np.array([o['bonf'] for o in R]); 
        orc=lambda t: norm.sf(t-lam)+norm.cdf(-t-lam)
        print(m, round(lam,2),'|', round(bm[:,:5].mean(),2), round(bm[:,5:].mean(),2), round(np.mean([o['fwer'] for o in R]),2),'|',
              *[round(rec(f'z{k}'),2) for k in K],'|',*[round(rec(f'l{k}'),2) for k in K],'|',
              round(orc(norm.ppf(1-10/(2*N))),2), round(orc(norm.ppf(1-50/(2*N))),2), round(orc(thr),2))
