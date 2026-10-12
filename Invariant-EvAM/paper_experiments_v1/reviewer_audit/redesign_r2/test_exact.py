import numpy as np
from scipy.stats import fisher_exact
from run_simulations import conditional_tail,exact_sign
rng=np.random.default_rng(123)
for _ in range(100):
 tab=rng.integers(0,100,(2,2));lo,hi=conditional_tail([tab]);assert abs(lo-fisher_exact(tab,alternative='less').pvalue)<1e-10;assert abs(hi-fisher_exact(tab,alternative='greater').pvalue)<1e-10
for tab in [np.array([[10,0],[0,10]]),np.array([[0,10],[10,0]]),np.array([[10,0],[10,0]])]:
 assert np.isfinite(conditional_tail([tab])).all()
 assert exact_sign([tab],eta=1) in [exact_sign([tab]),0]
print('PASS: 100 single-stratum tails match scipy Fisher; degenerate margins and sensitivity monotonicity checked')
from scipy.stats import nchypergeom_fisher
for _ in range(100):
 tab=rng.integers(0,100,(2,2));a,b,c,d=tab.ravel();eta=float(rng.uniform(-3,3));lo,hi=conditional_tail([tab],eta);assert abs(lo-nchypergeom_fisher.cdf(a,a+b+c+d,a+b,a+c,np.exp(eta)))<1e-9;assert abs(hi-nchypergeom_fisher.sf(a-1,a+b+c+d,a+b,a+c,np.exp(eta)))<1e-9
print('PASS: 100 noncentral tails match scipy within 1e-9')
