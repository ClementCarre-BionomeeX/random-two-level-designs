# Control for Section 9 (reviewer): family-wise error of Bonferroni on the debiased z, 1000 replicates per m,
# with the truth of sim_rank.py (5 main effects + 5 interactions, theta = 0.5) and under the global null (theta = 0).
# Usage: python3 sim_fwer.py reps workers  -> prints one JSON line per (truth, m)
import numpy as np, sys, json; from multiprocessing import Pool; from scipy.stats import norm; from rank_common import *
R=int(sys.argv[1]) if len(sys.argv)>1 else 1000; W=int(sys.argv[2]) if len(sys.argv)>2 else 4
def one(a):
    th0,m,rep=a; rng=np.random.default_rng(7919*m+rep+(0 if th0>0 else 10**6))
    th,S=truth(rng,max(th0,1e-12)); th=th if th0>0 else np.zeros_like(th); S=S if th0>0 else np.array([],int)
    F=features(rademacher(m,p,rng),P); y=F@th+rng.standard_normal(m)
    z,_=debiased_z(F,y); az=np.abs(z); thr=norm.ppf(1-0.05/(2*N))
    null=np.ones(N,bool); null[S]=False
    return th0,m,float((az[null]>thr).any()),float(np.mean(az[null]>norm.ppf(.975))),float(np.std(z[null]))
if __name__=='__main__':
    jobs=[(t,m,r) for t in [0.5,0.0] for m in [75,100,150,200,300] for r in range(R)]
    with Pool(W) as pool: res=pool.map(one,jobs,chunksize=20)
    for t in [0.5,0.0]:
        for m in [75,100,150,200,300]:
            A=np.array([x[2:] for x in res if x[0]==t and x[1]==m])
            f=A[:,0].mean(); print(json.dumps({'theta':t,'m':m,'reps':len(A),'fwer':round(f,4),'fwer_se':round(float(np.sqrt(f*(1-f)/len(A))),4),
                'per_test_alpha_0.05':round(float(A[:,1].mean()),4),'sd_null_z':round(float(A[:,2].mean()),3)}),flush=True)
