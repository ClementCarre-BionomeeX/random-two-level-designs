# Second run-count sweep: same random strategy and truth as sweep_runs.py, with the four weaknesses of the first sweep removed.
# 1. Detection without oracle: besides the top s, effects are called by Benjamini-Hochberg at level Q on the debiased z
#    (z = d / se, se = sigma_hat * sqrt(M) / (M - df), sigma_hat^2 = |r|^2 / (M - df)). Power and realised FDP are reported.
# 2. "All found": fraction of replicates where every active effect is found (top s, and BH).
# 3. Effect sizes: 'equal' (all |effect| = theta, as before) or 'spread' (|effect| = theta * exp(U(-ln 2, ln 2)), from theta/2
#    to 2 theta, geometric mean theta).
# 4. Finer M grid (ratio 2^(1/4)). Points far below the threshold use RC replicates, points near it R replicates: a point is
#    run with R replicates once the RC-replicate pass gives all_found(BH) >= NEAR.
# The chain stops after two consecutive M where recall(top s) >= 0.99, all_found(top s) >= 0.95, recall(BH) >= 0.99 and
# all_found(BH) >= 0.95.
# Usage: python3 sweep_runs2.py p s theta sizes reps [K]  -> one JSON line per M on stdout
import numpy as np, sys, json, time, warnings, zlib
from scipy.special import ndtr
from sklearn.linear_model import Lasso; warnings.filterwarnings('ignore')
p=int(sys.argv[1]); s=int(sys.argv[2]); theta=float(sys.argv[3]); SIZES=sys.argv[4]; R=int(sys.argv[5])
K=int(sys.argv[6]) if len(sys.argv)>6 else 20000
MLIST=sorted({int(round(32*2**(i/4))) for i in range(33)}); RC=20; NEAR=0.3; Q=0.05
TARGETS={'recall':0.99,'all_found':0.95,'bh_recall':0.99,'bh_all_found':0.95}
iu=np.triu_indices(p,1); PA,PB=iu[0].astype(np.int32),iu[1].astype(np.int32); NP=len(PA); N=p+NP
SM=s//2; SI=s-SM; TYPES=['main','both','one','neither']
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
    if SIZES=='spread': th[S]*=np.exp(rng.uniform(-np.log(2),np.log(2),len(S)))
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
    se=np.sqrt(r@r/(M-df))*np.sqrt(M)/(M-df); return d/se
def bh(z):
    # Indices called by Benjamini-Hochberg at level Q on two-sided normal p-values.
    order=np.argsort(-np.abs(z)); pv=2*ndtr(-np.abs(z[order])); ok=np.nonzero(pv<=Q*np.arange(1,N+1)/N)[0]
    return order[:ok[-1]+1] if len(ok) else order[:0]
def point(M,reps):
    top=[]; called=[]; fdp=[]; nd=[]; T=None
    for rep in range(reps):
        rng=np.random.default_rng(zlib.crc32(f'sweep2|{p}|{s}|{theta}|{SIZES}|{M}|{rep}'.encode()))
        th,S,T=truth(rng); z=random_fr(M,th,S,rng); B=bh(z)
        top.append(np.isin(S,np.argsort(-np.abs(z))[:s])); called.append(np.isin(S,B))
        fdp.append(float((~np.isin(B,S)).sum()/max(len(B),1))); nd.append(len(B))
    top=np.array(top); called=np.array(called)
    return {'recall':round(float(top.mean()),4),'all_found':round(float(top.all(1).mean()),4),
        'bh_recall':round(float(called.mean()),4),'bh_all_found':round(float(called.all(1).mean()),4),
        'bh_fdr':round(float(np.mean(fdp)),4),'bh_ndisc':round(float(np.mean(nd)),1),
        'by_type':{tp:round(float(top[:,T==tp].mean()),4) for tp in TYPES if (T==tp).any()},
        'bh_by_type':{tp:round(float(called[:,T==tp].mean()),4) for tp in TYPES if (T==tp).any()}}
hits=0
for M in MLIST:
    if M<s: continue
    t0=time.time(); reps=RC; m=point(M,RC)
    if m['bh_all_found']>=NEAR: reps=R; m=point(M,R)
    out={'p':p,'N':N,'s':s,'sM':SM,'sI':SI,'theta':theta,'sizes':SIZES,'M':M,'reps':reps,'K':K,'Q':Q,'sec':round(time.time()-t0,1),**m}
    print(json.dumps(out),flush=True)
    hits=hits+1 if reps==R and all(m[k]>=v for k,v in TARGETS.items()) else 0
    if hits>=2: break
