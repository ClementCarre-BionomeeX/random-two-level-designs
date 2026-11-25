# Fair comparison of designs at equal budget, all analysed by the SAME estimator (reviewer point 1-3, 5).
# Designs (budget M depends on p through n_reg = 2^ceil(log2(p+1))):
#   ssd     : Lin (1993) supersaturated design, half fraction of a Hadamard matrix of order n_reg (M = n_reg/2 < p)
#   res3    : regular resolution III fraction, p random non-constant columns of a Sylvester-Hadamard matrix (M = n_reg)
#   res4    : res3 followed by its fold-over (M = 2 n_reg)
#   dsd     : definitive screening design from a Paley conference matrix of order q+1, q prime = 1 mod 4 (M = 2(q+1)+1)
#   random  : i.i.d. uniform +-1 runs, same M as the structured design it is paired with
# Estimator (identical for every design, no oracle): columns of all main effects and of the K pairs with the largest
# |standardised marginal correlation| with y are centred and scaled to unit variance; scaled lasso (Sun & Zhang 2012:
# lambda = sigma_hat sqrt(2 log N / M), sigma_hat = ||r||/sqrt(M), iterated); ranking by |coefficient|, unselected effects
# after selected ones by |standardised correlation with the final residual|. For random designs a debiased ranking is
# also reported (random_db), as in regime.py.
# Truth: 'equal' (s_M mains + s_I interactions cycling 2x1D, 1x1D, 0x1D, all |theta| = th) or 'heredity'
# (interactions drawn among all pairs with per-pair weights 1 : 0.136 : 0.0145 for 2, 1, 0 parents carrying a main effect,
# i.e. the ratios 0.33 : 0.045 : 0.0048 of the heredity rates of Li, Sudarsanam & Frey 2006 as used by Mak et al. 2021;
# sizes log-uniform: mains th*2^U(0,1), interactions th*2^U(-1,0), so interactions are smaller than mains).
# Metrics (per type): recall in the top s (oracle s, for continuity) and selection without oracle: true positive rate of
# the scaled-lasso support, number of false selections, false discovery proportion.
# Env SIGMA_KNOWN=1: same lasso with lambda = sqrt(2 log N / M) from the true sigma (as in the run-count sweeps).
# Usage: python3 compare_designs.py p design truth sM sI theta reps [K]   -> one JSON line
import numpy as np, sys, json, time, warnings, zlib; from scipy.linalg import hadamard
from sklearn.linear_model import Lasso; warnings.filterwarnings('ignore')
p=int(sys.argv[1]); design=sys.argv[2]; truthkind=sys.argv[3]; SM=int(sys.argv[4]); SI=int(sys.argv[5])
theta=float(sys.argv[6]); R=int(sys.argv[7]); K=int(sys.argv[8]) if len(sys.argv)>8 else 20000
iu=np.triu_indices(p,1); PA,PB=iu[0],iu[1]; NP=len(PA); N=p+NP
nreg=int(2**np.ceil(np.log2(p+1)))
import os; SIGMA_KNOWN=os.environ.get('SIGMA_KNOWN','0')=='1'   # 1: lambda from the true noise level (sigma = 1) for every design
def pid(a,b):
    a,b=min(a,b),max(a,b); return p+a*p-a*(a+1)//2+(b-a-1)
def is_prime(n): return n>1 and all(n%d for d in range(2,int(n**.5)+1))
def paley_conference(p):
    q=p if p%2 else p+1
    while not (is_prime(q) and q%4==1): q+=1
    sq={(i*i)%q for i in range(1,q)}; chi=np.array([0]+[1 if i in sq else -1 for i in range(1,q)])
    Q=np.array([[chi[(j-i)%q] for j in range(q)] for i in range(q)])
    C=np.zeros((q+1,q+1)); C[0,1:]=1; C[1:,0]=1; C[1:,1:]=Q; return C
H=hadamard(nreg).astype(float)
if design=='dsd': CONF=paley_conference(p)
M={'ssd':nreg//2,'res3':nreg,'res4':2*nreg,'dsd':2*(CONF.shape[0] if design=='dsd' else 0)+1}[design]
import os; RFRAC=float(os.environ.get('RFRAC','1')); MR=int(round(M*RFRAC))   # random fraction size (RFRAC<1: smaller than the structured design)
def structured(rng):
    if design=='ssd':
        br=1+rng.integers(nreg-1); rows=H[:,br]>0; cols=np.setdiff1d(np.arange(1,nreg),[br])
        X=H[rows][:,rng.choice(cols,p,replace=False)]
    elif design in ('res3','res4'):
        X=H[:,1+rng.choice(nreg-1,p,replace=False)]
        if design=='res4': X=np.vstack([X,-X])
    else:
        C=CONF[:,rng.choice(CONF.shape[0],p,replace=False)]; X=np.vstack([C,-C,np.zeros((1,p))])
    return X*rng.choice([-1.,1.],p)                     # random column choice and sign switching
def truth(rng):
    mains=rng.choice(p,SM,replace=False); ism=np.zeros(p,bool); ism[mains]=True
    if truthkind=='equal':
        rest=np.setdiff1d(np.arange(p),mains); ints=[]; types=[]
        for kd in (['both','one','neither']*SI)[:SI]:
            while True:
                if kd=='both': a,b=rng.choice(mains,2,replace=False)
                elif kd=='one': a,b=rng.choice(mains),rng.choice(rest)
                else: a,b=rng.choice(rest,2,replace=False)
                e=pid(a,b)
                if e not in ints: ints.append(e); types.append(kd); break
        S=np.r_[mains,ints].astype(int); th=np.zeros(N); th[S]=theta*rng.choice([-1,1],len(S))
    else:
        npar=ism[PA].astype(int)+ism[PB]; w=np.array([0.0145,0.136,1.0])[npar]; w/=w.sum()
        pick=rng.choice(NP,SI,replace=False,p=w); ints=list(p+pick)
        types=[['neither','one','both'][k] for k in npar[pick]]
        S=np.r_[mains,ints].astype(int); th=np.zeros(N)
        th[mains]=theta*2**rng.uniform(0,1,SM)*rng.choice([-1,1],SM)
        th[p+pick]=theta*2**rng.uniform(-1,0,SI)*rng.choice([-1,1],SI)
    return th,S,np.array(['main']*SM+types)
def col(X,e): return X[:,e] if e<p else X[:,PA[e-p]]*X[:,PB[e-p]]
def analyse(X,y,debias=False):
    Mx=len(y); yc=y-y.mean()
    mu1=X.mean(0); sd1=X.std(0); sd1[sd1==0]=np.inf; Xs=(X-mu1)/sd1
    G1=(X.T@X)/Mx; X2=X*X; G2=(X2.T@X2)/Mx
    mu2=G1[PA,PB]; sd2=np.sqrt(np.maximum(G2[PA,PB]-mu2**2,0)); sd2[sd2<1e-9]=np.inf
    def pair_score(v):                                  # standardised pair columns ^T v, all pairs at once
        G=X.T@(v[:,None]*X); return (G[PA,PB]-mu2*v.sum())/sd2
    kk=min(K,NP); keep=np.argpartition(-np.abs(pair_score(yc)),kk-1)[:kk]
    F=(X[:,PA[keep]]*X[:,PB[keep]]-mu2[keep])/sd2[keep]; Fk=np.hstack([Xs,F])
    sig=np.std(yc); m=Lasso(fit_intercept=False,max_iter=20000,tol=1e-4,warm_start=True)
    if SIGMA_KNOWN:                                     # oracle variant: lambda from the true sigma = 1, one fit
        m.alpha=np.sqrt(2*np.log(N)/Mx); m.fit(Fk,yc); r=yc-Fk@m.coef_
    for _ in range(0 if SIGMA_KNOWN else 6):
        m.alpha=sig*np.sqrt(2*np.log(N)/Mx); m.fit(Fk,yc); r=yc-Fk@m.coef_
        new=max(np.linalg.norm(r)/np.sqrt(Mx),1e-6)
        if abs(new-sig)<1e-3*sig: sig=new; break
        sig=new
    t=m.coef_; full=np.zeros(N); full[:p]=t[:p]; full[p+keep]=t[p:]
    sel=np.flatnonzero(full)
    tie=np.r_[np.abs(Xs.T@r),np.abs(pair_score(r))]/Mx
    order=np.argsort(-(np.abs(full)*1e6+tie))
    out={'order':order,'sel':sel}
    if debias:
        df=min((t!=0).sum(),Mx-2); c=1.0/(Mx-df)
        d=full.copy(); d[:p]+=c*(Xs.T@r); d[p:]+=c*pair_score(r); out['order_db']=np.argsort(-np.abs(d))
    return out
TY=['main','both','one','neither']
t0=time.time(); acc={k:{'rec':{t:[0,0] for t in TY},'tpr':{t:[0,0] for t in TY},'fp':[],'fdp':[],'allrec':[]} for k in ['struct','random','random_db']}
def add(d,T,hit):
    for t,h in zip(T,hit): d[t][0]+=int(h); d[t][1]+=1
for rep in range(R):
    rng=np.random.default_rng(zlib.crc32(f'cmp|{p}|{design}|{truthkind}|{SM}|{SI}|{theta}|{rep}'.encode()))
    th,S,T=truth(rng); s=len(S)
    for nm,X in [('struct',structured(rng)),('random',rng.choice([-1.,1.],size=(MR,p)))]:
        y=sum(th[e]*col(X,e) for e in S)+rng.standard_normal(len(X))
        o=analyse(X,y,debias=(nm=='random'))
        for key,order in [(nm,o['order'])]+([('random_db',o['order_db'])] if nm=='random' else []):
            hit=np.isin(S,order[:s]); add(acc[key]['rec'],T,hit); acc[key]['allrec'].append(hit.all())
        found=np.isin(S,o['sel']); nfp=len(o['sel'])-found.sum(); add(acc[nm]['tpr'],T,found)
        acc[nm]['fp'].append(nfp); acc[nm]['fdp'].append(nfp/max(len(o['sel']),1))
out={'sigma_known':SIGMA_KNOWN,'rfrac':RFRAC,'M_random':MR,'p':p,'design':design,'truth':truthkind,'sM':SM,'sI':SI,'theta':theta,'M':M,'N':N,'reps':R,'K':K,'sec':round(time.time()-t0,1)}
for k,v in acc.items():
    res={'recall_top_s':{t:round(a/n,4) for t,(a,n) in v['rec'].items() if n},'n_eff':{t:n for t,(a,n) in v['rec'].items() if n},
         'all_in_top_s':round(float(np.mean(v['allrec'])),4)}
    if v['fp']:
        res['tpr']={t:round(a/n,4) for t,(a,n) in v['tpr'].items() if n}
        res['false_sel_mean']=round(float(np.mean(v['fp'])),2); res['fdp_mean']=round(float(np.mean(v['fdp'])),4)
    out[k]=res
print(json.dumps(out))
