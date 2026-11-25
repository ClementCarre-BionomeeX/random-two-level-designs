# Regime map for large p (e.g. 1000 factors, N = p(p+1)/2 candidate effects) without storing the full feature matrix.
# Same truth, seeds, classical strategy and metric as regime.py. Random strategy: blockwise marginal screening keeps all main
# effects + the K pairs with largest |Phi_j^T y| (sure independence screening), lasso on the kept columns with the SAME
# penalty sqrt(2 log N / M) as regime.py, then the debiased statistic is computed blockwise for ALL N effects.
# Speed-up (exact algebra, no approximation): for any vector v, the inner products of ALL pair columns with v are
#   sum_i X_ia X_ib v_i = (X^T diag(v) X)_ab, and for centred pair columns subtract mean_ab * sum(v), mean_ab = (X^T X)_ab / M.
# So screening and debiasing cost two p x p matrix products instead of building the M x p(p-1)/2 pair matrix.
# Usage: python3 regime_big.py p scen s theta budget reps [K]
import numpy as np, itertools, sys, json, time, warnings, zlib; from scipy.linalg import hadamard
from sklearn.linear_model import Lasso; warnings.filterwarnings('ignore')
import os; RANK=os.environ.get('RANK','staged'); SKIP_RANDOM=os.environ.get('SKIP_RANDOM')=='1'
p=int(sys.argv[1]); scen=sys.argv[2]; slev=sys.argv[3]; theta=float(sys.argv[4]); bk=sys.argv[5]; R=int(sys.argv[6])
K=int(sys.argv[7]) if len(sys.argv)>7 else 20000; BLK=20000
iu=np.triu_indices(p,1); PA,PB=iu[0].astype(np.int32),iu[1].astype(np.int32); NP=len(PA); N=p+NP
def pid(j,k):
    a,b=min(j,k),max(j,k); return p+a*p-a*(a+1)//2+(b-a-1)       # index of pair (a,b) in triu order
SM,SI={'low':(3,3),'mid':(5,6),'high':(10,12)}[slev]
nreg=int(2**np.ceil(np.log2(p+1))); H=hadamard(nreg).astype(np.float32)
FOLD=(bk=='fold'); n1=nreg*(2 if FOLD else 1); FF=np.array(list(itertools.product([-1.,1.],repeat=5))); M=n1+32
def col(X,e):
    return X[:,e] if e<p else X[:,PA[e-p]]*X[:,PB[e-p]]
def signal(X,th,S): return sum(th[e]*col(X,e) for e in S)
def truth(rng):
    mains=rng.choice(p,SM,replace=False); rest=rng.permutation(np.setdiff1d(np.arange(p),mains)); ints=[]; types=[]
    kinds={'both':['both']*SI,'one':['one']*SI,'neither':['neither']*SI,'mixed':(['both','one','neither']*SI)[:SI]}[scen]
    for kd in kinds:
        while True:
            if kd=='both': a,b=rng.choice(mains,2,replace=False)
            elif kd=='one': a,b=rng.choice(mains),rest[rng.integers(0,len(rest))]
            else: a,b=rest[rng.choice(len(rest),2,replace=False)]
            e=pid(a,b)
            if e not in ints: ints.append(e); types.append(kd); break
    S=np.r_[mains,ints]; th=np.zeros(N); th[S]=theta*rng.choice([-1,1],len(S))
    return th,S,np.array(['main']*SM+types)
def ols_t(Z,y):
    b=np.linalg.lstsq(Z,y,rcond=None)[0]; df=len(y)-Z.shape[1]; s2=np.sum((y-Z@b)**2)/df
    return b/np.sqrt(s2*np.diag(np.linalg.pinv(Z.T@Z)))
def classical(th,S,rng):
    X1=H[:,1+rng.choice(nreg-1,p,replace=False)].astype(float)
    if FOLD: X1=np.vstack([X1,-X1])
    y1=signal(X1,th,S)+rng.standard_normal(len(X1)); t1=np.abs(ols_t(np.column_stack([np.ones(len(X1)),X1]),y1)[1:])
    sel=np.sort(np.argsort(-t1)[:5]); X2=np.tile(rng.choice([-1.,1.],p),(32,1)); X2[:,sel]=FF
    y2=signal(X2,th,S)+rng.standard_normal(32); pr=list(itertools.combinations(range(5),2))
    Z=np.column_stack([np.ones(32),FF]+[FF[:,a]*FF[:,b] for a,b in pr]); t2=np.abs(ols_t(Z,y2)[1:])
    eff=np.array(list(sel)+[pid(sel[a],sel[b]) for a,b in pr])
    if RANK=='pooled':
        score=np.full(N,-1.0); score[:p]=t1; score[eff]=t2
    else:
        score=np.full(N,-1.0); score[:p]=t1/1e3; score[eff]=1e3+t2
    return np.argsort(-score)
def blocks():
    for s0 in range(0,NP,BLK): yield s0, PA[s0:s0+BLK], PB[s0:s0+BLK]
def pair_block(X,a,b):
    F=X[:,a]*X[:,b]; return F-F.mean(0)
def random_fr(th,S,rng):
    X=rng.choice([-1.,1.],size=(M,p)); y=signal(X,th,S)+rng.standard_normal(M); yc=y-y.mean()
    Xc=X-X.mean(0); G0=(X.T@X)/M
    def pair_dot(v): G=X.T@(v[:,None]*X); return (G-G0*v.sum())[PA,PB]   # centred pair columns ^T v, all pairs
    sc=np.abs(pair_dot(yc))
    keep=np.sort(np.argpartition(-sc,min(K,NP)-1)[:min(K,NP)])
    Fk=np.hstack([Xc,pair_block(X,PA[keep],PB[keep])])
    lam=np.sqrt(2*np.log(N)/M); t=Lasso(alpha=lam,fit_intercept=False,max_iter=20000,tol=1e-4).fit(Fk,yc).coef_
    r=yc-Fk@t; df=(t!=0).sum(); c=1.0/(M-df)
    d=np.empty(N,dtype=np.float32); d[:p]=t[:p]+c*(Xc.T@r)
    tp=np.zeros(NP); tp[keep]=t[p:]
    d[p:]=tp+c*pair_dot(r)
    return np.argsort(-np.abs(d))
t0=time.time(); res={'classical':[],'random':[]}
for rep in range(R):
    rng=np.random.default_rng(zlib.crc32(f'{p}|{scen}|{slev}|{theta}|{bk}|{rep}'.encode()))
    th,S,types=truth(rng); k=len(S)
    for nm,fn in [('classical',classical)]+([] if SKIP_RANDOM else [('random',random_fr)]):
        res[nm].append(np.isin(S,fn(th,S,rng)[:k]))
out={'rank':RANK,'p':p,'scen':scen,'s':slev,'theta':theta,'budget':bk,'runs':M,'reps':R,'K':K,'sec':round(time.time()-t0,1)}
for nm in [k for k in res if res[k]]:
    A=np.array(res[nm]); out[nm]={tp:round(float(A[:,types==tp].mean()),3) for tp in ['main','both','one','neither'] if (types==tp).any()}
print(json.dumps(out))
