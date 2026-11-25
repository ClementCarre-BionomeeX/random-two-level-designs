# Number of runs needed by the random fraction as a function of the number of active effects s, at fixed p.
# Random strategy identical to regime_fast.py (screening of K pairs + lasso with lambda = sqrt(2 log N / M) + debiasing of
# all N effects through the exact identity (X^T diag(v) X)_ab). No classical strategy: the budget M is the swept variable.
# Truth: s active effects, s_M = s//2 main effects and s_I = s - s_M interactions cycling through 2x1D, 1x1D, 0x1D; all of
# absolute size theta (sigma = 1), random signs. Measure: recall in the top s (all types, and by type).
# One call = one chain (p, s, theta): M increases along MLIST; the chain stops after two consecutive M with recall >= STOP.
# Usage: python3 sweep_runs.py p s theta reps [K]  -> one JSON line per M on stdout
import numpy as np, sys, json, time, warnings, zlib
from sklearn.linear_model import Lasso; warnings.filterwarnings('ignore')
p=int(sys.argv[1]); s=int(sys.argv[2]); theta=float(sys.argv[3]); R=int(sys.argv[4])
K=int(sys.argv[5]) if len(sys.argv)>5 else 20000
MLIST=[32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096]; STOP=0.99
iu=np.triu_indices(p,1); PA,PB=iu[0].astype(np.int32),iu[1].astype(np.int32); NP=len(PA); N=p+NP
SM=s//2; SI=s-SM
def pid(j,k):
    a,b=min(j,k),max(j,k); return p+a*p-a*(a+1)//2+(b-a-1)
def col(X,e): return X[:,e] if e<p else X[:,PA[e-p]]*X[:,PB[e-p]]
def truth(rng):
    mains=rng.choice(p,SM,replace=False); rest=rng.permutation(np.setdiff1d(np.arange(p),mains)); ints=[]; types=[]
    kinds=(['both','one','neither']*SI)[:SI] if SM>=2 else (['one','neither']*SI)[:SI]
    for kd in kinds:
        while True:
            if kd=='both': a,b=rng.choice(mains,2,replace=False)
            elif kd=='one': a,b=rng.choice(mains),rest[rng.integers(0,len(rest))]
            else: a,b=rest[rng.choice(len(rest),2,replace=False)]
            e=pid(a,b)
            if e not in ints: ints.append(e); types.append(kd); break
    S=np.r_[mains,ints].astype(int); th=np.zeros(N); th[S]=theta*rng.choice([-1,1],len(S))
    return th,S,np.array(['main']*SM+types)
def random_fr(M,th,S,rng):
    X=rng.choice([-1.,1.],size=(M,p)); y=sum(th[e]*col(X,e) for e in S)+rng.standard_normal(M); yc=y-y.mean()
    Xc=X-X.mean(0); G0=(X.T@X)/M
    def pair_dot(v): G=X.T@(v[:,None]*X); return (G-G0*v.sum())[PA,PB]
    kk=min(K,NP); keep=np.sort(np.argpartition(-np.abs(pair_dot(yc)),kk-1)[:kk])
    F=X[:,PA[keep]]*X[:,PB[keep]]; Fk=np.hstack([Xc,F-F.mean(0)]); del F
    lam=np.sqrt(2*np.log(N)/M); t=Lasso(alpha=lam,fit_intercept=False,max_iter=20000,tol=1e-4).fit(Fk,yc).coef_
    r=yc-Fk@t; df=min((t!=0).sum(),M-2); c=1.0/(M-df)
    d=np.empty(N); d[:p]=t[:p]+c*(Xc.T@r); tp=np.zeros(NP); tp[keep]=t[p:]; d[p:]=tp+c*pair_dot(r)
    return np.argsort(-np.abs(d))
hits=0
for M in MLIST:
    if M<s: continue
    t0=time.time(); A=[]; T=None
    for rep in range(R):
        rng=np.random.default_rng(zlib.crc32(f'sweep|{p}|{s}|{theta}|{M}|{rep}'.encode()))
        th,S,T=truth(rng); A.append(np.isin(S,random_fr(M,th,S,rng)[:s]))
    A=np.array(A); out={'p':p,'N':N,'s':s,'sM':SM,'sI':SI,'theta':theta,'M':M,'reps':R,'K':K,'sec':round(time.time()-t0,1),
        'recall':round(float(A.mean()),4),'by_type':{tp:round(float(A[:,T==tp].mean()),4) for tp in ['main','both','one','neither'] if (T==tp).any()},
        'all_found':round(float(A.all(1).mean()),4)}
    print(json.dumps(out),flush=True)
    hits=hits+1 if out['recall']>=STOP else 0
    if hits>=2: break
