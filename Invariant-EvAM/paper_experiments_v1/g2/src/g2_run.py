from pathlib import Path
import sys,json,time,argparse,os
import numpy as np
import pandas as pd
from scipy.stats import chi2,norm,beta
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
R=Path(__file__).resolve().parents[3];G=R/'paper_experiments_v1/g2';sys.path.insert(0,str(G/'vendor'))
from g2_engine import snapshot,operators,Likelihood,fit,wald,profile,interval,sign,STATES

def run_one(cell,rep,purpose=201,detail=True):
 # Only this generator/scorer reads evaluator contract. Estimators receive counts and stripped public operators.
 pub=json.loads((G/'configs/public_contract.json').read_text());private=json.loads((G/'configs/evaluator_contract.json').read_text())
 label=['strong','weak','nonidentifying'][cell//4];effect=private['effects'][cell%4];theta=np.r_[private['baseline'],effect,private['other_interaction']]
 ops=operators(pub['operators'][label]);rng=np.random.default_rng(np.random.SeedSequence([private['master_seed'],2,purpose,cell,0,rep]));p,j=snapshot(theta);sizes=[500,500];counts=[];accept=[]
 for (a,f),phi,n in zip(ops,private['true_phis'][label],sizes):
  w=np.exp(f@np.array(phi));q=p*w;q/=q.sum();counts.append(rng.multinomial(n,a@q));accept.append(float(p@(w/w.max())))
 observed={'counts':[x.tolist() for x in counts],'cell':cell,'replicate':rep,'seed':[private['master_seed'],2,purpose,cell,0,rep]}
 t0=time.time();lik=Likelihood(counts,ops);x0=np.r_[[-1,-1,-1,0,0],np.zeros(lik.dim-5)];x1=x0.copy();x1[3:5]=[.5,-.5];joint=fit(lik,starts=[x0,x1]);results=[]
 targets={'primary':np.array([0,0,0,1,0]),'interaction_difference':np.array([0,0,0,1,-1]),'interaction_mean':np.array([0,0,0,.5,.5])}
 # Evaluator-only geometry, never passed to method decisions.
 xt=np.concatenate([theta]+[np.array(x) for x in private['true_phis'][label]]);oracle=wald(lik,{'x':xt},np.r_[targets['primary'],np.zeros(lik.dim-5)])
 oracle_status='NONIDENTIFIED' if oracle['null_leakage']>1e-7 else 'LOCAL_IDENTIFIED'
 def row(method,est,lo,hi,target='primary',kind='WALD_EXPLORATORY',success=True,reason='',**more):
  truth=float(targets[target]@theta);call=sign(lo,hi,'COMPLETE' if success else 'FAILED');called=call in ['POSITIVE','NEGATIVE'];correct=called and ((call=='POSITIVE' and truth>0) or (call=='NEGATIVE' and truth<0))
  results.append({'protocol_version':'1.0','group_id':'G2','experiment_id':'CORE','cell_id':cell,'design':label,'method_id':method,'target_id':target,'family_id':'primary' if target=='primary' else 'secondary_no_simultaneous_claim','model_contract_hash':__import__('hashlib').sha256((R/'paper_experiments_v1/shared/MODEL_CONTRACT.json').read_bytes()).hexdigest(),'operator_id':label,'operator_knowledge':'K1_STRUCTURED','topology_id':'D3_SPARSE5','evidence_type':'NEW_RUN' if purpose==201 else 'DEVELOPMENT_ONLY','truth_available':True,'truth':truth,'estimate':est,'ci_lower':lo,'ci_upper':hi,'interval_type':kind,'coverage_defined':bool(success and not np.isnan(lo) and not np.isnan(hi)),'covers_truth':bool(lo<=truth<=hi) if success and not np.isnan(lo) and not np.isnan(hi) else None,'sign_call':call,'called':called,'correct_call':correct,'wrong_sign':called and not correct,'computational_status':'COMPLETE' if success else 'FAILED','structural_status':oracle_status,'null_leakage':oracle['null_leakage'],'n_unique_patients':1000,'n_observed':1000,'n_screened':float(sum(n/ac for n,ac in zip(sizes,accept))),'n_screened_basis':'expected_rejection_candidates','n_calibration':0,'budget_basis':'fixed_enrolled_patients','budget_value':1000,'seed':json.dumps(observed['seed']),'parameter_draw_id':0,'replicate_id':rep,'split_id':'confirmation' if purpose==201 else 'development','boundary_hit':bool(np.isinf(lo) or np.isinf(hi)),'compatibility_status':'MODEL_CORRECT_BY_GENERATOR','cutoff_source':'chi_square_asymptotic_candidate' if 'PROFILE' in kind else 'normal_asymptotic_candidate','runtime_seconds':time.time()-t0,'reason':reason,**more})
 # B5 and M share exact fit object; no estimator truth access.
 for target,vec in targets.items():
  a=np.r_[vec,np.zeros(lik.dim-5)];w=wald(lik,joint,a);row('B5',w['estimate'],w['lo'],w['hi'],target,success=joint['success'],fitted_null_leakage=w['null_leakage'])
  if target!='primary':row('M_SECONDARY_WALD',w['estimate'],w['lo'],w['hi'],target,success=joint['success'],reason='secondary diagnostic only; not main M confidence decision')
 points=profile(lik,joint,index=3,bound=3.,extra_restart=True)
 # Refit joint if constrained search improves it (same correction shared by B5 and M).
 bestp=min(points,key=lambda x:x['nll'])
 if bestp['nll']<joint['nll']-1e-5:
  joint=fit(lik,starts=[joint['x'],np.array(bestp['x'])]);
  for r in results:
   if r['method_id']=='B5':
    vec=targets[r['target_id']];w=wald(lik,joint,np.r_[vec,np.zeros(lik.dim-5)]);r.update(estimate=w['estimate'],ci_lower=w['lo'],ci_upper=w['hi'],covers_truth=bool(w['lo']<=r['truth']<=w['hi']));r['sign_call']=sign(w['lo'],w['hi']);r['called']=r['sign_call'] in ['POSITIVE','NEGATIVE'];r['correct_call']=r['called'] and ((r['sign_call']=='POSITIVE' and r['truth']>0) or (r['sign_call']=='NEGATIVE' and r['truth']<0));r['wrong_sign']=r['called'] and not r['correct_call']
 mci=interval(points,joint['nll'],chi2.ppf(.95,1),3.)
 good=joint['success'] and all(p['success'] for p in points)
 row('M',float(joint['x'][3]),mci['lo'],mci['hi'],kind='LOCAL_PROFILE_EXPLORATORY',success=good,reason=mci['status'],cutoff=float(chi2.ppf(.95,1)),profile_components=json.dumps(mci['components']))
 # Baseline B1 drops genotype ascertainment while retaining exact measurement marginalization.
 pooled=Likelihood(counts,[(a,np.empty((8,0))) for a,f in ops]);pf=fit(pooled,starts=[joint['x'][:5],x0[:5]])
 w=wald(pooled,pf,targets['primary']);row('B1',w['estimate'],w['lo'],w['hi'],success=pf['success'],reason='panel-aware selection-omission ablation')
 # B2 cannot inverse-variance combine target-unidentified per-environment representatives.
 ws=[];efits=[]
 for c,op in zip(counts,ops):
  el=Likelihood([c],[op]);ef=fit(el);ew=wald(el,ef,np.r_[targets['primary'],np.zeros(el.dim-5)]);efits.append(ef)
  if ef['success'] and ew['null_leakage']<=1e-7 and ew['se']>0:ws.append(ew)
 if len(ws)==len(ops):
  weights=np.array([1/w['se']**2 for w in ws]);est=float(weights@np.array([w['estimate'] for w in ws])/weights.sum());se=1/np.sqrt(weights.sum());row('B2',est,est-1.95996398454*se,est+1.95996398454*se,reason='inverse-variance meta of finite-information targets')
 else:row('B2',np.nan,-np.inf,np.inf,kind='NONIDENTIFIED_META_REFUSAL',success=all(x['success'] for x in efits),reason='insufficient per-environment target information; no forced meta')
 # B3 genuinely has separate environment theta and penalized deviations around their mean.
 hl=Likelihood(counts,ops,'hierarchical',pub['hierarchical_lambda']);hx=np.r_[np.tile(joint['x'][:5],2),joint['x'][5:]];hf=fit(hl,starts=[hx]);a=np.r_[targets['primary']/2,targets['primary']/2,np.zeros(hl.dim-10)];hw=wald(hl,hf,a)
 row('B3',hw['estimate'],hw['lo'],hw['hi'],kind='PENALIZED_APPROXIMATION',success=hf['success'],reason='shrinkage lambda=1 fixed; not data-identification evidence')
 # B4 first stage nonparametric common source + unknown selection, no true phi.
 fl=Likelihood(counts,ops,'fusion');pfit=snapshot(joint['x'][:5])[0];fx=np.r_[np.log(np.maximum(pfit[:-1],1e-100)/max(pfit[-1],1e-100)),joint['x'][5:]];ff=fit(fl,bound=6.,starts=[fx,np.zeros(fl.dim)])
 pi=__import__('scipy').special.softmax(np.r_[ff['x'][:7],0])
 def project(t):
  p,j=snapshot(t);return -float(pi@np.log(np.maximum(p,1e-300))),-j.T@(pi/np.maximum(p,1e-300))
 po=minimize(project,joint['x'][:5],jac=True,method='L-BFGS-B',bounds=[(-3,3)]*5,options={'maxiter':400,'ftol':1e-11})
 b4cut=float(chi2.ppf(.95,9));b4ci=interval(points,ff['nll'],b4cut,3.)
 row('B4',float(po.x[3]),b4ci['lo'],b4ci['hi'],kind='LOCAL_PROFILE_EXPLORATORY',success=bool(ff['success'] and po.success and good),reason='two-stage point + joint projection of first-stage LR compatible set; df9 unvalidated in singular cases',cutoff=b4cut,profile_components=json.dumps(b4ci['components']),first_stage_nll=ff['nll'])
 row('ALWAYS_UNKNOWN',np.nan,-np.inf,np.inf,kind='TRIVIAL_REFERENCE',reason='zero utility reference, not method success')
 row('REGULARIZED_POINT_ONLY',hw['estimate'],hw['estimate'],hw['estimate'],kind='POINT_STRATEGY_NOT_CI',success=hf['success'],reason='overconfidence reference; not a confidence interval')
 # Exact simultaneous binomial marginal bands; universally valid outer target projection R.
 alpha=.05/sum(len(c) for c in counts);bands=[]
 for c in counts:
  n=c.sum();bands.append([[float(beta.ppf(alpha/2,k,n-k+1)) if k else 0.,float(beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.] for k in c])
 row('CONSERVATIVE_OUTER',np.nan,-np.inf,np.inf,kind='CONSERVATIVE_REFERENCE',reason='certified but vacuous R outer projection of Bonferroni exact probability set; no efficiency claim')
 # Explicit masked-zero negative control, same observed counts, missing coordinates coded 0.
 masked=np.zeros(8,int)
 for c,(agg,f) in zip(counts,ops):
  for y,ct in enumerate(c):masked[np.flatnonzero(agg[y])[0]]+=ct
 ml=Likelihood([masked],[(np.eye(8),np.empty((8,0)))]);mf=fit(ml);mw=wald(ml,mf,targets['primary']);row('MASKED_ZERO_NEGATIVE_CONTROL',mw['estimate'],mw['lo'],mw['hi'],success=mf['success'],reason='negative control only')
 risk=[]
 for level in [.8,.9,.95,.975,.99]:
  for method,ref,df in [('M',joint['nll'],1),('B4',ff['nll'],9)]:
   ci=interval(points,ref,chi2.ppf(level,df),3.);call=sign(ci['lo'],ci['hi'],'COMPLETE' if good else 'FAILED');called=call in ['POSITIVE','NEGATIVE'];correct=called and ((call=='POSITIVE' and effect>0) or (call=='NEGATIVE' and effect<0));risk.append({'cell_id':cell,'replicate_id':rep,'design':label,'truth':effect,'method_id':method,'confidence':level,'called':called,'wrong':called and not correct,'success':good})
 artifact={'observed':observed,'joint':{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in joint.items()},'profile':points,'exact_probability_bands':bands,'fusion':{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in ff.items()},'runtime_seconds':time.time()-t0}
 return results,risk,artifact

def batch(task):
 cell,start,stop,purpose=task;directory=G/'runs'/('core' if purpose==201 else 'development');directory.mkdir(exist_ok=True,parents=True);path=directory/f'cell{cell:02d}_r{start:04d}.json'
 if path.exists():return str(path),'EXISTS'
 out=[];risk=[];details=[];t=time.time()
 for rep in range(start,stop):
  try:
   a,b,c=run_one(cell,rep,purpose);out+=a;risk+=b;details.append(c)
  except Exception as e:
   details.append({'cell':cell,'replicate':rep,'exception':repr(e)})
 payload={'results':out,'risk':risk,'details':details,'seconds':time.time()-t,'expected_replicates':stop-start};temp=path.with_suffix('.tmp');temp.write_text(json.dumps(payload));temp.replace(path)
 return str(path),round(time.time()-t,1)

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['probe','core']);p.add_argument('--workers',type=int,default=3);args=p.parse_args()
 tasks=[(c,0,1,202) for c in [0,4,8]] if args.command=='probe' else [(c,i,min(i+10,500),201) for c in range(12) for i in range(0,500,10)]
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  futures={pool.submit(batch,t):t for t in tasks}
  for f in as_completed(futures):print(f.result(),flush=True)
if __name__=='__main__':main()
