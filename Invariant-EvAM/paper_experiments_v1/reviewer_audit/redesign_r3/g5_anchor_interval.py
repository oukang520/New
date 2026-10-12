"""Source-anchor acquisition and an expanded nuisance family; higher-cost redesign."""
from pathlib import Path
import sys,json,copy,time,argparse,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from scipy.stats import chi2
P=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;G=P/'g5/revision_r3'
sys.path[:0]=[str(P/'g5/src'),str(D)]
import g5_design_evaluator as evaluator
from g5_model import Model,fit,X
from g5_design_strategy import AS
from g5_interval import constrained,endpoints
C=json.loads((D/'PROTOCOL.json').read_text());BUDGET=1500;NCAL=175
class WithX(Model):
 def outputs(self,x):
    if self.cache is not None and np.array_equal(x,self.cache[0]):return self.cache[1]
    out=list(super().outputs(x))
    for group in self.data:
     if group.get('kind')=='X':
      p,j=self.source(x,group['env']);out.append((np.array(group['counts']),p,j))
    self.cache=(x.copy(),out);return out
def spec():
    pair=X[:,0]*X[:,1]
    return dict(topology='SPARSE',features=[np.column_stack([X[:,e],pair]).tolist() for e in range(2)],known_phi=[None,None],phi_bound=1.,bound=3.,epsilon=.2,xs=False)
def one(task):
    stage,scenario,rep=task;path=G/f'runs/anchor_interval_{stage}_{scenario}_{rep:03d}.json'
    if path.exists():return path.name,'CACHED'
    t=time.time();rng=np.random.default_rng(np.random.SeedSequence([C['seed'],5,970 if stage=='development' else 971,scenario,rep]))
    theta=np.r_[rng.uniform(-1.8,-.2,4),rng.choice([-.5,-.15,0,.15,.5,1.],6)];theta[4]=rng.choice([-1.,-.5,-.15,0,.15,.5,1.]);source,selected,ss=evaluator.true_probs(theta,scenario)
    data=[]
    for e in range(2):
     # Original 50 selected records per environment are completed (100 assays).
     # Independent source calibration: 350 records x4 events =1400 assays.
     data.append(dict(env=e,A=np.eye(16).tolist(),counts=rng.multinomial(50,selected[e]).tolist()))
     data.append(dict(env=e,kind='X',counts=rng.multinomial(NCAL,source[e]).tolist()))
    lik=WithX(data,spec());r=fit(lik);truth=float(theta[4]);ct=constrained(lik,r,truth);lr=2*(ct['nll']-r['nll']);valid=r['success'] and ct['success'] and lr>=-1e-5
    out=dict(stage=stage,scenario=scenario,replicate=rep,truth=truth,truth_parameters=theta.tolist(),added_cost=BUDGET,base_cost=300,n_new_source=2*NCAL,n_remeasured=100,fit=r,constrained_true=ct,lr_true=lr,valid=bool(valid),public_data=data,public_spec=spec())
    if stage=='evaluation':
     freeze=json.loads((G/'configs/ANCHOR_INTERVAL_CUTOFF.json').read_text());out['intervals']={}
     for name,cut in [('CLASSIC',float(chi2.ppf(.95,1))),('CALIBRATED',freeze['cutoff'])]:
      pr=endpoints(lik,r,cut);pr.update(cutoff=cut,covered=bool(pr['lo']<=truth<=pr['hi']),called=bool(pr['lo']>0 or pr['hi']<0),wrong=bool((pr['lo']>0 and truth<=0) or (pr['hi']<0 and truth>=0)),width=pr['hi']-pr['lo']);out['intervals'][name]=pr
    out['runtime']=time.time()-t;path.write_text(json.dumps(out,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()));return path.name,round(out['runtime'],2)
def finish(stage):
 raw=[json.loads(f.read_text()) for f in sorted((G/'runs').glob('anchor_interval_'+stage+'_*.json'))]
 if stage=='development':
  audit=[]
  for s in range(3):
   q=[z for z in raw if z['scenario']==s];v=[z['lr_true'] for z in q if z['valid']];audit.append(dict(scenario=s,n=len(q),valid=len(v),q95=float(np.quantile(v,.95,method='higher'))))
  freeze=dict(cutoff=max(float(chi2.ppf(.95,1)),max(z['q95'] for z in audit)),development=audit,rule='Maximum 95% development LR quantile, at least chi-square 95% cutoff; empirical validation required',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(G/'configs/ANCHOR_INTERVAL_CUTOFF.json').write_text(json.dumps(freeze,indent=2));print('FREEZE',freeze,flush=True)
 else:
  rows=[]
  for z in raw:
   for method,pr in z['intervals'].items():rows.append(dict(scenario=z['scenario'],replicate=z['replicate'],method=method,truth=z['truth'],cost=z['added_cost'],fit_valid=z['valid'],**pr))
  d=pd.DataFrame(rows);d['finite']=np.isfinite(d.width);d.to_csv(G/'results/anchor_interval_evaluation_raw.tsv',sep='\t',index=False);d.groupby(['scenario','method']).agg(n=('replicate','size'),coverage=('covered','mean'),called=('called','mean'),wrong=('wrong','mean'),finite=('finite','mean'),fit_valid=('fit_valid','mean'),median_width=('width','median')).reset_index().to_csv(G/'results/anchor_interval_evaluation_summary.tsv',sep='\t',index=False)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['development','evaluation']);a.add_argument('--workers',type=int,default=4);a.add_argument('--probe',action='store_true');args=a.parse_args();f=G/'configs/ANCHOR_ACQUISITION_FREEZE.json'
 if not f.exists():f.write_text(json.dumps(dict(added_cost=BUDGET,n_new_source=350,n_completed=100,source_record_assays=4,selected_record_baseline=3,epsilon_log_rate=.2,selection_features='environment-specific main gene plus shared declared pair feature',development_replicates=100,evaluation_replicates=500,scenarios=[0,1,2],scope='Independent synthetic confirmation at larger actual budget, not same-cost advantage over cost74',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
 n=100 if args.stage=='development' else 500;tasks=[(args.stage,s,r) for s in range(3) for r in range(n)]
 if args.probe:tasks=tasks[:1]
 with ProcessPoolExecutor(args.workers) as ex:
  for i,r in enumerate(ex.map(one,tasks),1):print(i,len(tasks),r,flush=True)
 if not args.probe:finish(args.stage)
