from pathlib import Path
import json,hashlib,time,argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from exact_target import probs,selected,confidence
from acquisition import choose
P=Path(__file__).resolve().parents[2];G=P/'g5/revision_r8';SEED=202609288;CASES=['SAMPLE_LIMITED','JOINT_VIEW_MISSING','ASCERTAINMENT_MARGIN','RARE_STRATA'];BUDGET=6000

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def design(case):
 if case==0:base={'SELECT_A':200,'SELECT_B':200};pilot={'SELECT_A':200,'SELECT_B':200}
 elif case==1:base={'MARGINAL_A':6400,'MARGINAL_B':6400};pilot={'MARGINAL_A':500,'MARGINAL_B':500}
 elif case==2:base={'SELECT_A':2000};pilot={'SELECT_A':500}
 else:base={'SELECT_A':2000,'SELECT_B':2000};pilot={'SELECT_A':500,'SELECT_B':500}
 actions={}
 same={'SELECT_A':1500,'SELECT_B':1500} if case in [0,3] else {'MARGINAL_A':3000,'MARGINAL_B':3000} if case==1 else {'SELECT_A':3000}
 for name,add in [('MORE_SAME',same),('REFERENCE_MARGIN',{'MARGINAL_A':6000}),('NEW_SOURCE',{'SOURCE':1000})]:
  actions[name]={k:base.get(k,0)+add.get(k,0) for k in set(base)|set(add)}
 if case==1:actions['JOINT_REVEAL']={'MARGINAL_A':400,'MARGINAL_B':6400,'SOURCE':6000}
 if case==2:actions['NEW_ENV']={'SELECT_A':2000,'SELECT_B':3000}
 return base,pilot,actions

def qdist(p,kind,phis):
 if kind=='SOURCE':return p
 if kind=='MARGINAL_A':return np.array([p[0]+p[1],p[2]+p[3]])
 if kind=='MARGINAL_B':return np.array([p[0]+p[2],p[1]+p[3]])
 return selected(p,kind,phis[0] if kind=='SELECT_A' else phis[1])

def outcome(groups,truth):
 z=confidence(groups);valid=z['status']=='COMPLETE';called=valid and (z['lo']>0 or z['hi']<0);wrong=called and ((z['lo']>0 and truth<=0)or(z['hi']<0 and truth>=0));z.update(covered=bool(valid and z['lo']<=truth<=z['hi']),correct=bool(called and not wrong),wrong=bool(wrong),valid=valid,width=z['hi']-z['lo'],restricted_width=max(0,min(z['hi'],4)-max(z['lo'],-4)))
 return z

def one(t):
 stage,case,rep=t;path=G/('development' if stage=='development' else 'runs')/f'{case}_{rep:03d}.json'
 if path.exists():return
 start=time.time();st=0 if stage=='development' else 1;rng=np.random.default_rng([SEED,5,st,case,rep]);effect=[-.8,0,.8][rep%3];a,b=rng.uniform(-1.4,-.6,2);p=probs(np.exp([a,b,b+effect]));phis=[0.,0.] if case==0 else [.7,-.7] if case<3 else [3.5,3.5];base,pilots,actions=design(case);pilot=[dict(kind=k,counts=rng.multinomial(n,qdist(p,k,phis)).tolist()) for k,n in sorted(pilots.items())];choice,diagnostics,fit=choose(pilot,actions)
 # Confirmation pool is independent of all pilot counts used in decisions.
 confirm={k:rng.multinomial(n,qdist(p,k,phis)) for k,n in sorted(base.items())};hidden_pool=None
 if case==1:
  hidden_pool=rng.multinomial(6400,p);confirm['MARGINAL_A']=np.array([hidden_pool[:2].sum(),hidden_pool[2:].sum()])
 raw=[]
 for action,ns in sorted(actions.items()):
  rr=np.random.default_rng([SEED,5,st,case,rep,90,sorted(actions).index(action)]);counts={k:c.copy() for k,c in confirm.items()}
  if action=='JOINT_REVEAL':
   sampled=rr.multivariate_hypergeometric(hidden_pool,6000);rest=hidden_pool-sampled;counts['MARGINAL_A']=np.array([rest[:2].sum(),rest[2:].sum()]);counts['SOURCE']=sampled;cost=6000
  else:
   cost=0
   for k,n in sorted(ns.items()):
    new=n-base.get(k,0)
    if new:counts[k]=counts.get(k,np.zeros(len(qdist(p,k,phis)),dtype=int))+rr.multinomial(new,qdist(p,k,phis));cost+=new*(6 if k=='SOURCE' else 1 if k.startswith('MARGINAL') else 2)
  assert cost==BUDGET
  groups=[dict(kind=k,counts=c.tolist()) for k,c in sorted(counts.items())];raw.append(dict(action=action,cost=cost,groups=groups,interval=outcome(groups,effect)))
 basegroups=[dict(kind=k,counts=c.tolist()) for k,c in sorted(confirm.items())]
 doc=dict(stage=stage,case=case,label=CASES[case],replicate=rep,truth=effect,rates=np.exp([a,b,b+effect]).tolist(),phis=phis,pilot=pilot,pilot_fit=fit,choice=choice,diagnostics=diagnostics,baseline=outcome(basegroups,effect),outcomes=raw,runtime=time.time()-start)
 path.write_text(json.dumps(doc,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['development','evaluation']);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args();code={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('exact_target.py'),Path(__file__).with_name('acquisition.py')]}
 if args.stage=='evaluation':assert json.loads((G/'configs/FREEZE.json').read_text())['code']==code
 n=6 if args.stage=='development' else 60
 with ProcessPoolExecutor(args.workers) as pool:
  for i,_ in enumerate(pool.map(one,[(args.stage,c,r) for c in range(4) for r in range(n)]),1):
   if i%12==0:print(args.stage,i,4*n,flush=True)
 docs=[json.loads(p.read_text()) for p in (G/('development' if args.stage=='development' else 'runs')).glob('*.json')];rows=[];choices=[]
 for z in docs:
  choices.append(dict(case=z['case'],label=z['label'],replicate=z['replicate'],action=z['choice']))
  for o in z['outcomes']:
   rows.append(dict(case=z['case'],label=z['label'],replicate=z['replicate'],truth=z['truth'],policy=o['action'],action=o['action'],cost=o['cost'],**o['interval']))
   if o['action']==z['choice']:rows.append(dict(case=z['case'],label=z['label'],replicate=z['replicate'],truth=z['truth'],policy='TARGET_C_OPT',action=o['action'],cost=o['cost'],**o['interval']))
  rows.append(dict(case=z['case'],label=z['label'],replicate=z['replicate'],truth=z['truth'],policy='NO_ACTION',action='NO_ACTION',cost=0,**z['baseline']))
 d=pd.DataFrame(rows);d.to_csv(G/f'results/{args.stage}_raw.tsv',sep='\t',index=False);s=d.groupby(['label','policy']).agg(n=('replicate','size'),coverage=('covered','mean'),finite=('finite','mean'),correct=('correct','mean'),wrong=('wrong','mean'),width=('restricted_width','mean'),valid=('valid','mean')).reset_index();s.to_csv(G/f'results/{args.stage}_summary.tsv',sep='\t',index=False);pd.DataFrame(choices).to_csv(G/f'results/{args.stage}_choices.tsv',sep='\t',index=False);print(s.to_string(index=False),flush=True);print(pd.DataFrame(choices).groupby(['label','action']).size().to_string(),flush=True)
 if args.stage=='development':(G/'configs/FREEZE.json').write_text(json.dumps(dict(seed=SEED,code=code,cases=CASES,budget=BUDGET,evaluation_per_case=60,rule='classical target c-opt at pilot-only fit, lexicographic estimability then variance',inference='conditional CP polytope; confirmation independent of pilot including retained base records',costs={'new_selected_full':2,'random_source_full_including_recruitment':6,'reference_single_bit':1,'reveal_missing_bit':1},width_metric='intersection with [-4,4] used only as bounded scoring window; actual intervals remain unbounded where appropriate',primary='gap-dependent action switching, narrower independent target intervals with valid coverage; no novel-algorithm superiority requirement',scope='closed sparse parent-child subsystem, declared sampling families, artificial cost menu'),indent=2))
