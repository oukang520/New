"""Numerical audit: exact constrained LR crossings, no cutoff calibration.
All 108 preselected independent profiles; independent of core outcomes.
"""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2
G=Path(__file__).resolve().parents[1];sys.path.insert(0,str(G/'src'))
from g2_engine import operators,Likelihood,fit,snapshot

def main():
 pub=json.loads((G/'configs/public_contract.json').read_text());ev=json.loads((G/'configs/evaluator_contract.json').read_text());paths=json.loads((G/'results/G2_bound_profiles.json').read_text());rows=[]
 for path in paths:
  cell=path['cell'];rep=path['rep'];bound=path['bound'];label=['strong','weak','nonidentifying'][cell//4];ops=operators(pub['operators'][label]);theta=np.r_[ev['baseline'],ev['effects'][cell%4],ev['other_interaction']];p=snapshot(theta)[0];rng=np.random.default_rng(np.random.SeedSequence([20260923,2,207,cell,0,rep]));counts=[]
  for (a,f),phi in zip(ops,ev['true_phis'][label]):
   pp=p*np.exp(f@np.array(phi));pp/=pp.sum();counts.append(rng.multinomial(500,a@pp))
  lik=Likelihood(counts,ops);pts=path['points'];original_nll=path['fit_nll'];bestpoint=min(pts,key=lambda x:x['nll']);refit=fit(lik,bound,starts=[np.array(path['fit']),np.array(bestpoint['x'])]);path['fit_nll']=refit['nll'];path['fit']=refit['x'].tolist();cut=chi2.ppf(.95,1);val=np.array([2*(pt['nll']-path['fit_nll'])-cut for pt in pts]);success=True;refined=[];residual=[];calls=0
  for left,right in zip(range(len(pts)-1),range(1,len(pts))):
   if val[left]*val[right]>=0:continue
   starts=[np.array(pts[left]['x']),np.array(pts[right]['x']),np.array(path['fit'])]
   def func(v):
    nonlocal success,calls
    out=fit(lik,bound,starts=starts,fixed=(3,v),maxiter=400);success=success and out['success'];calls+=1;return 2*(out['nll']-path['fit_nll'])-cut
   try:
    root=brentq(func,pts[left]['c'],pts[right]['c'],xtol=1e-6);refined.append(root);residual.append(abs(func(root)))
   except ValueError:success=False
  lo=-np.inf if val[0]<=0 else (min(refined) if refined else np.nan);hi=np.inf if val[-1]<=0 else (max(refined) if refined else np.nan)
  rows.append({'cell_id':cell,'replicate_id':rep,'bound':bound,'ci_lower_refined':lo,'ci_upper_refined':hi,'root_success':success,'max_lr_crossing_error':max(residual) if residual else 0,'constrained_calls':calls,'nll':path['fit_nll'],'original_nll':original_nll,'joint_refit_success':refit['success'],'status':'NUMERICAL_AUDIT_ONLY_NOT_GLOBAL_CERTIFICATE'})
  pd.DataFrame(rows).to_csv(G/'results/G2_bound_refined_crossings.tsv',sep='\t',index=False)
 print('refined profiles',len(rows),flush=True)
if __name__=='__main__':main()
