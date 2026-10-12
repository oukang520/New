"""Private simulation/reveal evaluator; strategy receives baseline counts and capacities only."""
from pathlib import Path
import sys,json,copy,argparse,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from g5_model import Model,fit,profile,snapshot,X
from g5_design_strategy import recommend,make_data,design_spec,AS
R=Path(__file__).resolve().parents[3];G=R/'paper_experiments_v1/g5';P=json.loads((G/'configs/PROTOCOL.json').read_text())['design'];MASTER=20260923
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x));q.replace(p)
def rng(purpose,scenario,rep,stream):return np.random.default_rng(np.random.SeedSequence([MASTER,5,purpose,scenario,rep,stream]))
def true_probs(theta,scenario):
 p=[];sel=[];selection=[]
 for e in range(2):
  t=theta.copy()
  if scenario==2 and e==1:t[4]+=.2
  pp=snapshot(t)[0];z=X[:,e]*(.5 if e==0 else 1.)
  if scenario==1:z=z+.5*X[:,0]*X[:,1]
  ss=np.exp(-2.2+z);sp=pp*ss;sp/=sp.sum();p.append(pp);sel.append(sp);selection.append(ss)
 return p,sel,selection

def generate_sim(scenario,rep,purpose=610):
 r=rng(purpose,scenario,rep,0);theta=np.r_[r.uniform(-1.8,-.2,4),r.choice([-.5,-.15,0,.15,.5,1.],6)];theta[4]=r.choice([-1.,-.5,-.15,0.,.15,.5,1.]);p,sp,selection=true_probs(theta,scenario);base=[r.choice(16,50,p=q) for q in sp];add=[r.choice(16,100,p=q) for q in sp];calx=[r.choice(16,100,p=q) for q in p];cals=[r.random(100)<selection[e][calx[e]] for e in range(2)];ev=[r.choice(16,500,p=q) for q in sp]
 return dict(base=base,add=add,calx=calx,cals=cals,evaluation=ev,source_evaluation=r.choice(16,1000,p=p[0]),true_target=float(theta[4]),truth=theta.tolist(),capacities=dict(add_pairs=100,calibration_pairs=100),base_unique=100,evidence_type='INDEPENDENT_KNOWN_TRUTH_SIMULATION',n_screened=None)

def generate_direct(rep):
 rows=json.loads((R/'paper_experiments_v1/g4/data_private/reference_records/LUAD.json').read_text());random=rng(620,0,rep,0);role={role:np.array([z['reference_state'] for z in rows if z['role']==role]) for role in ['train','calibration','evaluation']};out={};selected={};environments={};stats={}
 for key,x in role.items():
  env=random.integers(0,2,len(x));u=random.random(len(x));prob=np.exp(-2.2+np.where(env==0,.5*X[x,0],X[x,1]));s=u<prob;selected[key]=s;environments[key]=env;stats[key]=dict(screened=len(x),selected=int(s.sum()));out[key]=x
 base=[];add=[];calx=[];cals=[];ev=[];baseids=[];futureids=[];calids=[]
 for e in range(2):
  ids=np.flatnonzero((environments['train']==e)&selected['train']);random.shuffle(ids)
  if len(ids)<50:return dict(capacity_failure=True,reason='BASE_ENROLLED_CAPACITY',stats=stats)
  base.append(out['train'][ids[:50]]);baseids.append(ids[:50].tolist());future=np.flatnonzero(environments['calibration']==e);random.shuffle(future);fi=future[selected['calibration'][future]];add.append(out['calibration'][fi]);futureids.append(fi.tolist());calx.append(out['calibration'][future]);cals.append(selected['calibration'][future]);calids.append(future.tolist());ev.append(out['evaluation'][(environments['evaluation']==e)&selected['evaluation']])
 return dict(base=base,add=add,calx=calx,cals=cals,evaluation=ev,source_evaluation=out['evaluation'],true_target=None,capacities=dict(add_pairs=min(map(len,add)),calibration_pairs=min(map(len,calx))),base_unique=100,evidence_type='DIRECT_RECORD_CONTROLLED_REVEAL_PREVIOUSLY_EXPLORED',n_screened=stats,baseids=baseids,futureids=futureids,calids=calids)

def baseline_data(private):return make_data([np.bincount(np.argmax(AS[e][:,x],axis=0),minlength=8) for e,x in enumerate(private['base'])])
def reveal(private,action):
 partial=[];full=[];cal=[];a=action['add_pairs'];b=action['reveal_pairs'];c=action['calibration_pairs'];actualcalids=[]
 for e in range(2):
  base=private['base'][e];new=private['add'][e][:a];partial.append(np.bincount(np.argmax(AS[e][:,np.r_[base[b:],new]],axis=0),minlength=8));full.append(np.bincount(base[:b],minlength=16))
  # New selected and calibration patients must be distinct within each strategy.
  if 'futureids' in private:
   used=set(private['futureids'][e][:a]);pool=[i for i,k in enumerate(private['calids'][e]) if k not in used];take=np.array(pool[:c],int);actualcalids.append([private['calids'][e][i] for i in take]);assert len(take)==c
  else:take=np.arange(c)
  cx=private['calx'][e][take];cs=private['cals'][e][take];cal.append(np.bincount(cx+16*cs,minlength=32))
 data=make_data(partial,full,cal);return data,dict(new_unique_patients=2*a+2*c,remeasured_existing=2*b,n_unique=100+2*a+2*c,n_selected=100+2*a,n_calibration=2*c,cost=6*a+2*b+10*c,calibration_ids=actualcalids)

def scores(lik,r,private):
 if not r['success']:return dict(source_log_score=None,source_tv=None,hidden_brier=None)
 x=np.array(r['x']);p=lik.source(x)[0];src=private['source_evaluation'];freq=np.bincount(src,minlength=16)/len(src);bs=0;ll=0;nn=0
 for e,states in enumerate(private['evaluation']):
  sp=lik.selected(x,e)[0];post=AS[e]*sp[None,:]/np.maximum((AS[e]@sp)[:,None],1e-280);y=np.argmax(AS[e][:,states],axis=0);hidden=1 if e==0 else 0;pred=post[y]@X[:,hidden];bs+=np.sum((pred-X[states,hidden])**2);ll-=np.sum(np.log(np.maximum(post[y,states],1e-280)));nn+=len(states)
 return dict(source_log_score=float(-freq@np.log(np.maximum(p,1e-280))),source_tv=float(.5*np.abs(p-freq).sum()),hidden_brier=float(bs/nn) if nn else None,hidden_log_score=float(ll/nn) if nn else None,n_evaluation_selected=nn)

def one(task):
 kind,scenario,rep=task;path=G/f'runs/design_{kind}/{scenario}_{rep:03d}.json'
 if path.exists():return kind,scenario,rep,'CACHED'
 start=time.perf_counter();private=generate_direct(rep) if kind=='direct' else generate_sim(scenario,rep,600 if kind=='development' else 610)
 if private.get('capacity_failure'):put(path,private);return kind,scenario,rep,'CAPACITY_FAILURE'
 base=baseline_data(private);spec=design_spec();lik=Model(base,spec);br=fit(lik);bp=profile(lik,br,np.eye(10)[4]);results=[];decisions=[]
 for budget in P['budgets']:
  # Capacities are counts only; no future genotype/hidden response is supplied.
  caps=copy.deepcopy(private['capacities']);caps['calibration_pairs']=max(0,caps['calibration_pairs']-budget//6);decision=recommend(copy.deepcopy(base),budget,caps,MASTER+rep+scenario*1000);decisions.append(dict(budget=budget,decision=decision));cache={}
  for strategy,action in decision['choices'].items():
   if action['candidate_id'] not in cache:
    data,counts=reveal(private,action);model=Model(data,spec);r=fit(model);pr=profile(model,r,np.eye(10)[4]);sc=scores(model,r,private) if kind!='development' else {};truth=private['true_target'];pr['covers_truth']=pr['lo']<=truth<=pr['hi'] if truth is not None and pr['status']=='COMPLETE' else None;pr['wrong_sign']=bool((pr['sign']=='POSITIVE' and truth<=0) or (pr['sign']=='NEGATIVE' and truth>=0)) if truth is not None else None;cache[action['candidate_id']]=dict(fit=r,profile=pr,scores=sc,counts=counts)
    public=dict(data=data,spec=spec,action={k:action[k] for k in ['candidate_id','add_pairs','reveal_pairs','calibration_pairs','cost']});put(G/f'data_public/design_{kind}/{scenario}_{rep:03d}_{budget}_{action["candidate_id"]}.json',public)
   result=copy.deepcopy(cache[action['candidate_id']]);results.append(dict(strategy=strategy,budget=budget,action=action,result=result))
 # Private potential outcomes never enter strategy cache or public bundle.
 serial={k:([a.tolist() for a in v] if isinstance(v,list) and v and isinstance(v[0],np.ndarray) else v.tolist() if isinstance(v,np.ndarray) else v) for k,v in private.items()};put(G/f'data_private/design_{kind}/{scenario}_{rep:03d}.json',serial)
 put(path,dict(kind=kind,scenario=scenario,replicate=rep,evidence_type=private['evidence_type'],natural_theta_truth_available=False if kind=='direct' else True,baseline=br,baseline_profile=bp,baseline_scores=scores(lik,br,private) if kind!='development' else {},decisions=decisions,results=results,runtime_seconds=time.perf_counter()-start));return kind,scenario,rep,round(time.perf_counter()-start,2)

def main():
 a=argparse.ArgumentParser();a.add_argument('--kind',choices=['development','outer','direct'],required=True);a.add_argument('--workers',type=int,default=4);args=a.parse_args()
 if args.kind!='development':assert (G/'configs/EVALUATION_FREEZE.json').exists()
 reps=2 if args.kind=='development' else P['outer_replicates'] if args.kind=='outer' else P['direct_replicates'];scenarios=range(3) if args.kind=='outer' else [0];tasks=[(args.kind,s,i) for s in scenarios for i in range(reps)]
 with ProcessPoolExecutor(max_workers=args.workers) as ex:
  for i,f in enumerate(as_completed([ex.submit(one,t) for t in tasks]),1):print(i,'/',len(tasks),*f.result(),flush=True)
if __name__=='__main__':main()
