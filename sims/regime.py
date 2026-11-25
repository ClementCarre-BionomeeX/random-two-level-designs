# Regime map: random fraction + debiased lasso vs classical strategies, by interaction type.
# Usage: python3 regime.py p scenario s_level theta budget_kind reps  -> prints one JSON line
import numpy as np, itertools, sys, json, time, warnings, zlib; from scipy.linalg import hadamard
from sklearn.linear_model import Lasso; warnings.filterwarnings('ignore')
import os; RANK=os.environ.get('RANK','staged'); SKIP_RANDOM=os.environ.get('SKIP_RANDOM')=='1'
p=int(sys.argv[1]); scen=sys.argv[2]; slev=sys.argv[3]; theta=float(sys.argv[4]); bk=sys.argv[5]; R=int(sys.argv[6])
P=np.array(list(itertools.combinations(range(p),2))); N=p+len(P); idx={tuple(q):i for i,q in enumerate(P)}
def pid(j,k): return p+idx[(min(j,k),max(j,k))]
SM,SI={'low':(3,3),'mid':(5,6),'high':(10,12)}[slev]
nreg=int(2**np.ceil(np.log2(p+1))); H=hadamard(nreg).astype(float)
FOLD=(bk=='fold'); n1=nreg*(2 if FOLD else 1); FF=np.array(list(itertools.product([-1.,1.],repeat=5))); M=n1+32
def feats(X): return np.hstack([X, X[:,P[:,0]]*X[:,P[:,1]]])
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
def recall_at(order,S,k): return np.isin(S,order[:k])
def classical(th,rng):
    X1=H[:,1+rng.choice(nreg-1,p,replace=False)]
    if FOLD: X1=np.vstack([X1,-X1])
    y1=feats(X1)@th+rng.standard_normal(len(X1)); t1=np.abs(ols_t(np.column_stack([np.ones(len(X1)),X1]),y1)[1:])
    sel=np.sort(np.argsort(-t1)[:5]); X2=np.tile(rng.choice([-1.,1.],p),(32,1)); X2[:,sel]=FF
    y2=feats(X2)@th+rng.standard_normal(32); pr=list(itertools.combinations(range(5),2))
    Z=np.column_stack([np.ones(32),FF]+[FF[:,a]*FF[:,b] for a,b in pr]); t2=np.abs(ols_t(Z,y2)[1:])
    eff=np.array(list(sel)+[pid(sel[a],sel[b]) for a,b in pr])
    if RANK=='pooled':                                               # ablation: one pooled |t| scale for both stages
        score=np.full(N,-1.0); score[:p]=t1; score[eff]=t2
    else:
        score=np.full(N,-1.0); score[:p]=t1/1e3; score[eff]=1e3+t2      # stage-2 effects first, then other mains by stage-1 |t|
    return np.argsort(-score)
def random_fr(th,rng):
    X=rng.choice([-1.,1.],size=(M,p)); F=feats(X); y=F@th+rng.standard_normal(M); yc=y-y.mean(); Fc=F-F.mean(0)
    lam=np.sqrt(2*np.log(N)/M); t=Lasso(alpha=lam,fit_intercept=False,max_iter=20000,tol=1e-4).fit(Fc,yc).coef_
    r=yc-Fc@t; df=(t!=0).sum(); d=t+Fc.T@r/(M-df); return np.argsort(-np.abs(d))
t0=time.time(); res={'classical':[],'random':[]}
for rep in range(R):
    rng=np.random.default_rng(zlib.crc32(f'{p}|{scen}|{slev}|{theta}|{bk}|{rep}'.encode()))  # deterministic seed
    th,S,types=truth(rng); k=len(S)
    for nm,fn in [('classical',classical)]+([] if SKIP_RANDOM else [('random',random_fr)]):
        res[nm].append(recall_at(fn(th,rng),S,k))
out={'rank':RANK,'p':p,'scen':scen,'s':slev,'theta':theta,'budget':bk,'runs':M,'reps':R,'sec':round(time.time()-t0,1)}
for nm in [k for k in res if res[k]]:
    A=np.array(res[nm]); out[nm]={tp:round(float(A[:,types==tp].mean()),3) for tp in ['main','both','one','neither'] if (types==tp).any()}
print(json.dumps(out))
