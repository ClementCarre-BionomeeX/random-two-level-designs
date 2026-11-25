import numpy as np
from itertools import combinations
def pairs(p): return np.array(list(combinations(range(p),2)))
def features(X, P=None):
    """Phi(X): main effects then all pairwise products (Walsh functions of degree <=2)."""
    p = X.shape[1]
    if P is None: P = pairs(p)
    return np.hstack([X, X[:,P[:,0]]*X[:,P[:,1]]]).astype(float)
def rademacher(m,p,rng): return rng.choice([-1.0,1.0], size=(m,p))
