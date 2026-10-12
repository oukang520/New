"""Numerical resolution check for the classical comparator; no outcome-based tuning."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np,pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2
from concurrent.futures import ProcessPoolExecutor
from run_g2 import G,P,PUB,operators,Likelihood,fit
D=G/'profile_refinement_v2';D.mkdir(exist_ok=True)
def one(path):
 z=json.loads(path.read_text());dest=D/path.name
 if dest.exists():return
 m=Likelihood([np.array(c) for c in z['full_counts']],operators(PUB['operators'][z['design']]));s=np.r_[[-1.,-1.,-1.,0.,0.],np.zeros(m.dim-5)];starts=[s.copy()]
 for u,v in [(-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)]:
  t=s.copy();t[3:5]=[u,v];starts.append(t)
 r=fit(m,starts=starts,maxiter=1000);cut=chi2.ppf(.95,1);cache={}
 for repair in range(4):
  cache={}
  def val(v):
   if v not in cache:cache[v]=fit(m,starts=[r['x']]+starts,fixed=(3,v),maxiter=1000)
   return 2*(cache[v]['nll']-r['nll'])-cut
  lower=val(-3.);upper=val(3.)
  lo=-np.inf if lower<=0 else brentq(val,-3.,r['x'][3],xtol=1e-4)
  hi=np.inf if upper<=0 else brentq(val,r['x'][3],3.,xtol=1e-4)
  bestpoint=min(cache.values(),key=lambda t:t['nll'])
  if bestpoint['nll']>=r['nll']-1e-4:break
  if repair<3:r=fit(m,starts=[bestpoint['x'],r['x']]+starts,maxiter=1000)
 valid=bool(r['success'] and all(t['success'] for t in cache.values()));truth=z['truth'];called=valid and (lo>0 or hi<0);wrong=called and ((lo>0 and truth<=0)or(hi<0 and truth>=0));best=min([r['nll']]+[t['nll'] for t in cache.values()]);valid=bool(valid and best>=r['nll']-1e-4)
 o=dict(design=z['design'],n=z['n'],truth=truth,replicate=z['replicate'],method='CLASSIC_PROFILE_REFINED_V2',lo=lo,hi=hi,valid=valid,covered=bool(valid and lo<=truth<=hi),finite=bool(valid and np.isfinite([lo,hi]).all()),called=bool(called and valid),correct=bool(called and not wrong and valid),wrong=bool(wrong and valid),width=hi-lo,evaluations=len(cache),scope='continuous root within fixed [-3,3] search; boundary extends to infinity; local profile, not global certification')
 dest.write_text(json.dumps(o))
if __name__=='__main__':
 code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();f=G/'configs/PROFILE_REFINEMENT_V2_FREEZE.json'
 if f.exists():assert json.loads(f.read_text())['code']==code
 else:f.write_text(json.dumps(dict(code=code,reason='technical repair: retain continuous LR roots; use symmetric fixed interaction starts and refit unconstrained maximum when constrained fit improves it; retain v1 outputs separately',root_tolerance=1e-4,optimizer_maxiter=1000,starts=5,repair_passes=4,rule='all evaluation records, no selective reruns, same cutoff and parameter bounds',scope='technical comparator check, no changes to exact method'),indent=2))
 with ProcessPoolExecutor(3) as pool:
  while True:
   inputs=sorted(p for p in (G/'runs').glob('*.json') if not (D/p.name).exists())
   for i,_ in enumerate(pool.map(one,inputs),1):
    if i%60==0:print('refined',len(list(D.glob('*.json'))),flush=True)
   if len(list(D.glob('*.json')))>=1800:break
   time.sleep(20)
 df=pd.DataFrame([json.loads(p.read_text()) for p in D.glob('*.json')]);df.to_csv(G/'results/evaluation_refined_profile_v2.tsv',sep='\t',index=False);print(df.groupby(['design','n']).agg(coverage=('covered','mean'),correct=('correct','mean'),wrong=('wrong','mean'),finite=('finite','mean'),valid=('valid','mean')).to_string(),flush=True)
