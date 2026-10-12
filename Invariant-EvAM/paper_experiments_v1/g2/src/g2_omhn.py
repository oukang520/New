"""Separate matched clinical-observation endpoint. Never mixed with fixed-time core."""
from pathlib import Path
import sys,json,hashlib,time
R=Path(__file__).resolve().parents[3];G=R/'paper_experiments_v1/g2'
sys.path[:0]=[str(G/'vendor'),str(R/'src')]
import numpy as np,pandas as pd
from scipy.linalg import solve
from scipy.optimize import minimize as scipy_minimize
sys.path.insert(0,str(G/'src'))
from lbfgsb_direct import minimize_direct
def minimize(fun,x0,jac=True,method=None,bounds=None,options=None):
 opts=options or {}
 return minimize_direct(fun,x0,bounds,maxiter=opts.get('maxiter',15000),ftol=opts.get('ftol',2.2204460492503131e-9),gtol=opts.get('gtol',1e-5),maxls=opts.get('maxls',20))
from scipy.stats import chi2
from phase2.interaction_generator import SparseInteractingCTMC
from mhn import model as official_model
from mhn.full_state_space import Likelihood as official_likelihood
M=SparseInteractingCTMC(3,((1,0),(2,1)))
OMEGA=np.array([.4,-.3,.2]);PERM=np.array([0,4,2,6,1,5,3,7])
def matrix(t):
 out=np.zeros((3,3));np.fill_diagonal(out,t[:3]);out[1,0]=t[3];out[2,1]=t[4];return out

def local_reference(t):
 q,dq=M.generator_and_derivatives(t);rates=np.exp(M.states@OMEGA);a=np.diag(rates)-q.T;v=solve(a,np.eye(8)[0]);p=rates*v;j=np.column_stack([rates*solve(a,x.T@v) for x in dq]);return p,j

import ctypes
_ABS=ctypes.CDLL(str(G/'src/omhn_fast.dylib'))
_ARR=np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS')
_ABS.absorption.argtypes=[_ARR,_ARR,_ARR,_ARR]
def local(t):
 p=np.empty(8);j=np.empty((8,5));_ABS.absorption(np.ascontiguousarray(t),OMEGA,p,j);return p,j

def official(t):
 obj=official_model.oMHN(np.vstack([matrix(t),OMEGA]));return official_likelihood.generate_pTh(obj.get_equivalent_classical_mhn().log_theta)[PERM]

def official_fg(t,counts):
 obj=official_model.oMHN(np.vstack([matrix(t),OMEGA]));a=obj.get_equivalent_classical_mhn().log_theta;c=counts[PERM].astype(float);p=official_likelihood.generate_pTh(a);score=official_likelihood.score(a,c,p);grad=official_likelihood.grad(a,c,p)
 return -score,-np.r_[np.diag(grad),grad[1,0],grad[2,1]]

def main():
 contract={'version':'1.0','endpoint':'clinical_observation_event','observation_hazard':'exp(x dot omega)','omega_known':OMEGA.tolist(),'knowledge':'K0_KNOWN_OBSERVATION_RATE','topology':'D3_SPARSE5','n':1000,'effects':[-.8,0,.15,.8],'replicates':200,'seed_purpose':206,'methods':'official oMHN likelihood restricted to contract sparse topology vs exact absorption likelihood + M profile layer','parameter_bounds':[-3,3],'interpretation':'equivalence and calibration in matched subexperiment, no fixed-time performance comparison'}
 cp=G/'configs/B6_MATCHED_CONTRACT.json'
 if not cp.exists():cp.write_text(json.dumps(contract,indent=2)+'\n');(G/'configs/B6_MATCHED_FREEZE.sha256').write_text(hashlib.sha256(cp.read_bytes()).hexdigest()+'\n')
 rng=np.random.default_rng(np.random.SeedSequence([20260923,2,205]));checks=[]
 for i,t in enumerate(rng.uniform(-1.8,1,(40,5))):
  p,j=local(t);po=official(t);ct=np.arange(1,9)*17;f,g=official_fg(t,ct);locg=-j.T@(ct/p)
  checks.append({'draw':i,'max_probability_error':float(np.max(abs(p-po))),'max_gradient_error':float(np.max(abs(g-locg))),'sum_error':abs(float(po.sum())-1)})
 pd.DataFrame(checks).to_csv(G/'results/B6_probability_gradient_equivalence.tsv',sep='\t',index=False)
 assert max(r['max_probability_error'] for r in checks)<1e-10
 assert max(r['max_gradient_error'] for r in checks)<1e-6
 for ci,effect in enumerate(contract['effects']):
  dest=G/f'results/B6_matched_cell{ci}.tsv'
  rawdir=G/'runs/B6';rawdir.mkdir(exist_ok=True)
  rows=[];ttrue=np.array([-1,-1.2,-.8,effect,.5]);ptrue=local(ttrue)[0]
  for rep in range(contract['replicates']):
   rawpath=rawdir/f'cell{ci}_r{rep:04d}.json'
   if rawpath.exists():rows+=json.loads(rawpath.read_text())['results'];continue
   firstrow=len(rows)
   if rep%25==0:print('B6 raw archive',ci,rep,flush=True)
   rng=np.random.default_rng(np.random.SeedSequence([20260923,2,206,ci,0,rep]));c=rng.multinomial(1000,ptrue)
   def lf(t):
    p,j=local(t);return -float(c@np.log(p)),-j.T@(c/p)
   ofs=minimize(lambda t:official_fg(t,c),[-1,-1,-1,0,0],jac=True,method='L-BFGS-B',bounds=[(-3,3)]*5,options={'ftol':1e-11,'maxiter':400})
   lfs=minimize(lf,[-1,-1,-1,0,0],jac=True,method='L-BFGS-B',bounds=[(-3,3)]*5,options={'ftol':1e-11,'maxiter':400})
   p,j=local(ofs.x);cov=np.linalg.inv(1000*(j.T/p)@j);se=np.sqrt(cov[3,3]);lo=ofs.x[3]-1.95996398454*se;hi=ofs.x[3]+1.95996398454*se
   vals=[];warm=lfs.x.copy();grid=np.unique(np.r_[np.linspace(-3,3,49),lfs.x[3]])
   for v in grid:
    ix=[0,1,2,4]
    def ff(z):
     t=np.empty(5);t[3]=v;t[ix]=z;f,g=lf(t);return f,g[ix]
    fits=[minimize(ff,start[ix],jac=True,method='L-BFGS-B',bounds=[(-3,3)]*4,options={'ftol':1e-10,'maxiter':250}) for start in [warm,lfs.x]];best=min(fits,key=lambda x:x.fun);warm[ix]=best.x;warm[3]=v;vals.append({'c':float(v),'nll':float(best.fun),'success':bool(best.success),'theta':warm.tolist(),'attempts':[{'x':f.x.tolist(),'nll':float(f.fun),'success':bool(f.success),'nit':int(f.nit)} for f in fits]})
   sys.path.insert(0,str(G/'src'));from g2_engine import interval,sign
   ciobj=interval(vals,float(lfs.fun),chi2.ppf(.95,1),3)
   for meth,est,l,u,success in [('B6_OFFICIAL',ofs.x[3],lo,hi,ofs.success),('M_MATCHED',lfs.x[3],ciobj['lo'],ciobj['hi'],lfs.success and all(x['success'] for x in vals))]:
    call=sign(l,u,'COMPLETE' if success else 'FAILED');called=call in ['POSITIVE','NEGATIVE'];wrong=called and not ((call=='POSITIVE' and effect>0) or (call=='NEGATIVE' and effect<0))
    rows.append({'cell_id':ci,'replicate_id':rep,'method_id':meth,'truth':effect,'estimate':est,'ci_lower':l,'ci_upper':u,'covers_truth':bool(l<=effect<=u) if success else None,'success':bool(success),'called':called,'wrong':wrong,'endpoint':'clinical_observation_event','knowledge':'known_omega','estimate_difference_official_local':float(ofs.x[3]-lfs.x[3]),'loglik_difference':float(ofs.fun-lfs.fun)})
   rawpath.write_text(json.dumps({'seed':[20260923,2,206,ci,0,rep],'observed_counts':c.tolist(),'official_fit':{'theta':ofs.x.tolist(),'nll':float(ofs.fun),'success':bool(ofs.success)},'local_fit':{'theta':lfs.x.tolist(),'nll':float(lfs.fun),'success':bool(lfs.success)},'profile':vals,'results':rows[firstrow:]}))
  new=pd.DataFrame(rows)
  if dest.exists():
   old=pd.read_csv(dest,sep='\t');deltas={}
   for col in ['estimate','ci_lower','ci_upper']:
    a=old[col].to_numpy();b=new[col].to_numpy();finite=np.isfinite(a)&np.isfinite(b);assert np.array_equal(np.isinf(a),np.isinf(b));deltas[col]=float(np.max(abs(a[finite]-b[finite]))) if finite.any() else 0.;assert deltas[col]<1e-5,(ci,col,deltas[col])
   for col in ['called','wrong','success','covers_truth']:assert old[col].fillna(False).tolist()==new[col].fillna(False).tolist(),(ci,col)
   (G/f'runs/B6_archive_equivalence_cell{ci}.json').write_text(json.dumps(deltas,indent=2))
  else:new.to_csv(dest,sep='\t',index=False)
  print('B6 cell archive complete',ci,'rows',len(rows),flush=True)
if __name__=='__main__':main()
