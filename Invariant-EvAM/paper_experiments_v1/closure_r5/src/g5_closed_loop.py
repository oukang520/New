"""Frozen source-allocation closure; public decision consumes baseline only."""
from pathlib import Path
import sys,json,time,argparse,hashlib,copy
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from scipy.stats import chi2
P=Path(__file__).resolve().parents[2];G=P/'g5/revision_r5';CFG=json.loads((P/'closure_r5/configs/PROTOCOL.json').read_text());C=CFG['g5']
sys.path[:0]=[str(P/'reviewer_audit/redesign_r3'),str(P/'g5/src')]
from g5_anchor_interval import WithX,spec,evaluator
from g5_model import X,fit,snapshot
from g5_design_strategy import AS
from g5_interval import endpoints,constrained
NAMES=C['actions']
def dump(path,z):
 path.write_text(json.dumps(z,default=lambda a:a.item() if isinstance(a,np.generic) else a.tolist()))
def menu(cost):
 # Common completion of 100 existing records costs100; remaining1400.
 return [dict(name='MORE_SELECTED',source=[0,0],selected=[175,175],cost=1500),dict(name='REFERENCE_X',source=[1400//cost,0],selected=[0,0],cost=1500),dict(name='BALANCED_X',source=[696//cost,176],selected=[0,0],cost=100+(696//cost)*cost+704),dict(name='MIXED_X_SELECTED',source=[696//cost,0],selected=[88,88],cost=100+(696//cost)*cost+704)]
def information(model,x):
 H=model.whitened(x);return H.T@H

def metrics(I):
 v,u=np.linalg.eigh((I+I.T)/2);keep=v>max(v.max(),1e-100)*1e-8;a=np.eye(len(v))[4];leak=float(np.linalg.norm(u[:,~keep].T@a));variance=float(np.sum((u[:,keep].T@a)**2/v[keep])) if leak<1e-7 else np.inf
 return dict(leak=leak,variance=variance,rank=int(keep.sum()),logdet=float(np.log(np.maximum(v,0)+1e-8).sum()))
def public_policy(baseline,cost,random_seed):
 model=WithX(baseline,spec());r=fit(model);x=np.array(r['x']);points=[x]
 # A frozen sensitivity stencil, NOT a confidence set or latent truth.
 for sign in [-1,1]:
  z=x.copy()
  for sl in model.phi:
   if sl is not None:z[sl]=sign*.8
  points.append(z)
  z=x.copy()
  for sl in model.delta:
   if sl is not None:z[sl]=sign*.2
  points.append(z)
 table=[]
 for action in menu(cost):
  scores=[]
  for z in points:
   groups=[]
   for e in range(2):
    q,_=model.selected(z,e);p,_=model.source(z,e)
    groups += [dict(env=e,A=np.eye(16).tolist(),counts=((50+action['selected'][e])*q).tolist()),dict(env=e,kind='X',counts=((10+action['source'][e])*p).tolist())]
   scores.append(metrics(information(WithX(groups,spec()),z)))
  table.append(dict(**action,point=scores[0],worst_leak=max(s['leak'] for s in scores),worst_var=max(s['variance'] for s in scores)))
 def key(z,robust=False):
  leak=z['worst_leak'] if robust else z['point']['leak'];v=z['worst_var'] if robust else z['point']['variance']
  return (leak>=1e-7,v if np.isfinite(v) else leak,z['name'])
 choices=dict(ROBUST_TARGET=min(table,key=lambda z:key(z,True))['name'],C_OPT=min(table,key=key)['name'],D_OPT=max(table,key=lambda z:z['point']['logdet'])['name'],RANDOM=table[np.random.default_rng(random_seed).integers(4)]['name'],COUNT_ONLY='MORE_SELECTED',MAX_SOURCE=max(table,key=lambda z:(sum(z['source']),z['source'][0]))['name'])
 return choices,table,r

def one(task):
 stage,scenario,cost,rep=task;path=G/f'runs/{stage}_s{scenario}_c{cost}_{rep:04d}.json'
 if path.exists():return path.name,'cached'
 start=time.time();rng=np.random.default_rng([CFG['seed'],5,0 if stage=='development' else 1,scenario,cost,rep]);theta=np.r_[rng.uniform(-1.8,-.2,4),rng.choice([-.5,-.15,0,.15,.5,1.],6)];theta[4]=rng.choice([-1.,-.5,-.15,0,.15,.5,1.]);source,selected,ss=evaluator.true_probs(theta,min(scenario,2))
 if scenario==3:
  source,selected,ss=evaluator.true_probs(theta,0)
  for e in range(2):
   weight=np.exp((.5 if e==0 else 1.)*X[:,e]+1.5*X[:,0]*X[:,1]);q=source[e]*weight;selected[e]=q/q.sum()
 base=[rng.multinomial(50,q) for q in selected];pilot=[rng.multinomial(10,q) for q in source];baseline=[]
 for e in range(2):baseline += [dict(env=e,A=AS[e].tolist(),counts=(AS[e]@base[e]).tolist()),dict(env=e,kind='X',counts=pilot[e].tolist())]
 choices,table,br=public_policy(baseline,cost,[CFG['seed'],55,scenario,cost,rep]);out=[];cut=None
 if stage=='evaluation':cut=json.loads((G/'configs/CUTOFF_FREEZE.json').read_text())['cutoff']
 for action in menu(cost):
  data=[]
  arng=np.random.default_rng([CFG['seed'],56,0 if stage=='development' else 1,scenario,cost,rep,NAMES.index(action['name'])])
  for e in range(2):
   data += [dict(env=e,A=np.eye(16).tolist(),counts=(base[e]+arng.multinomial(action['selected'][e],selected[e])).tolist()),dict(env=e,kind='X',counts=(pilot[e]+arng.multinomial(action['source'][e],source[e])).tolist())]
  lik=WithX(data,spec());r=fit(lik);z=dict(action=action['name'],cost=action['cost'],fit=r,data=data)
  if stage=='development':
   ct=constrained(lik,r,float(theta[4]));lr=2*(ct['nll']-r['nll']);z.update(lr=lr,valid=bool(r['success'] and ct['success'] and lr>=-1e-5))
  else:
   pr=endpoints(lik,r,cut);pr.update(covered=bool(pr['lo']<=theta[4]<=pr['hi']),called=bool(pr['lo']>0 or pr['hi']<0),wrong=bool((pr['lo']>0 and theta[4]<=0) or (pr['hi']<0 and theta[4]>=0)),finite=bool(np.isfinite([pr['lo'],pr['hi']]).all()));pr['valid']=bool(pr['status'] in ['COMPLETE','UNRESOLVED']);pr['correct']=bool(pr['called'] and not pr['wrong'] and pr['valid']);z['interval']=pr
  out.append(z)
 if stage=='evaluation':
  bp=endpoints(WithX(baseline,spec()),br,cut);bp.update(covered=bool(bp['lo']<=theta[4]<=bp['hi']),called=bool(bp['lo']>0 or bp['hi']<0),wrong=bool((bp['lo']>0 and theta[4]<=0) or (bp['hi']<0 and theta[4]>=0)),finite=bool(np.isfinite([bp['lo'],bp['hi']]).all()));bp['valid']=bool(bp['status'] in ['COMPLETE','UNRESOLVED']);bp['correct']=bool(bp['called'] and not bp['wrong'] and bp['valid']);out.append(dict(action='NO_ACTION',cost=0,fit=br,data=baseline,interval=bp));choices['NO_ACTION']='NO_ACTION'
 dump(path,dict(stage=stage,scenario=scenario,reference_source_cost=cost,replicate=rep,truth=float(theta[4]),truth_parameters=theta,baseline=baseline,choices=choices,diagnostics=table,outcomes=out,runtime=time.time()-start));return path.name,round(time.time()-start,2)

def finish(stage):
 docs=[json.loads(f.read_text()) for f in sorted((G/'runs').glob(stage+'_*.json'))]
 if stage=='development':
  rows=[dict(scenario=z['scenario'],cost=z['reference_source_cost'],action=o['action'],lr=o['lr'],valid=o['valid']) for z in docs for o in z['outcomes']];d=pd.DataFrame(rows);d.to_csv(G/'results/development_raw.tsv',sep='\t',index=False)
  q=d[d.scenario.lt(3)&d.valid].groupby(['scenario','cost','action']).lr.quantile(.95,interpolation='higher');cut=max(chi2.ppf(.95,1),q.max());dump(G/'configs/CUTOFF_FREEZE.json',dict(cutoff=float(cut),rule=C['cutoff_rule'],code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),development_files=len(docs),invalid=int((~d.valid).sum())));print('Frozen cutoff',cut,flush=True)
 else:
  rows=[];actions=[]
  for z in docs:
   for policy,action in z['choices'].items():
    o=next(o for o in z['outcomes'] if o['action']==action);rows.append(dict(scenario=z['scenario'],reference_source_cost=z['reference_source_cost'],replicate=z['replicate'],truth=z['truth'],policy=policy,action=action,cost=o['cost'],**o['interval']))
   for o in z['outcomes']:actions.append(dict(scenario=z['scenario'],reference_source_cost=z['reference_source_cost'],replicate=z['replicate'],action=o['action'],cost=o['cost'],truth=z['truth'],**o['interval']))
  pd.DataFrame(rows).to_csv(G/'results/policy_raw.tsv',sep='\t',index=False);pd.DataFrame(actions).to_csv(G/'results/all_actions_raw.tsv',sep='\t',index=False)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage');a.add_argument('--workers',type=int,default=2);args=a.parse_args();n=C['development_per_cell'] if args.stage=='development' else C['evaluation_per_cell'];tasks=[(args.stage,s,c,r) for s in C['scenarios'] for c in C['reference_source_costs'] for r in range(n)]
 with ProcessPoolExecutor(args.workers) as pool:
  for i,z in enumerate(pool.map(one,tasks),1):
   if i%10==0:print(i,len(tasks),z,flush=True)
 finish(args.stage)
