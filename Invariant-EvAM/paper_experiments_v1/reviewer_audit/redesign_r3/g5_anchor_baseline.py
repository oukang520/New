"""Equal-assay-cost strong count-expansion control for the source-anchor design."""
from pathlib import Path
import json,argparse,hashlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from g5_anchor_interval import P,D,G,C,WithX,spec,evaluator
from g5_model import fit,geometry
from g5_interval import endpoints
def one(task):
 s,rep=task;path=G/f'runs/anchor_baseline_{s}_{rep:03d}.json'
 if path.exists():return path.name,'CACHED'
 z=json.loads((G/f'runs/anchor_interval_evaluation_{s}_{rep:03d}.json').read_text());theta=np.array(z['truth_parameters']);_,sp,_=evaluator.true_probs(theta,s);rng=np.random.default_rng(np.random.SeedSequence([C['seed'],5,972,s,rep]));data=[]
 for e in range(2):
  base=np.array(z['public_data'][2*e]['counts']);counts=base+rng.multinomial(175,sp[e]);data.append(dict(env=e,A=np.eye(16).tolist(),counts=counts.tolist()))
 lik=WithX(data,spec());r=fit(lik);pr=endpoints(lik,r,float(json.loads((G/'configs/ANCHOR_INTERVAL_CUTOFF.json').read_text())['cutoff']));truth=z['truth'];pr.update(covered=bool(pr['lo']<=truth<=pr['hi']),called=bool(pr['lo']>0 or pr['hi']<0),wrong=bool((pr['lo']>0 and truth<=0) or (pr['hi']<0 and truth>=0)),width=pr['hi']-pr['lo'])
 path.write_text(json.dumps(dict(scenario=s,replicate=rep,truth=truth,added_cost=1500,n_selected=450,n_new_source=0,public_data=data,fit=r,interval=pr),default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()));return path.name,pr['status']
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--freeze-only',action='store_true');a.add_argument('--workers',type=int,default=4);args=a.parse_args();f=G/'configs/ANCHOR_BASELINE_FREEZE.json'
 if not f.exists():f.write_text(json.dumps(dict(added_cost=1500,base_cost=300,action='Complete100 existing selected records; acquire350 additional full selected records',paired_to='same100 original selected records and same generator parameter draw',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
 if args.freeze_only:raise SystemExit(0)
 with ProcessPoolExecutor(args.workers) as ex:
  for i,r in enumerate(ex.map(one,[(s,r) for s in range(3) for r in range(500)]),1):print(i,1500,r,flush=True)
 rows=[]
 for f in sorted((G/'runs').glob('anchor_baseline_*.json')):
  z=json.loads(f.read_text());rows.append(dict(scenario=z['scenario'],replicate=z['replicate'],method='COUNT_EXPANSION',cost=z['added_cost'],truth=z['truth'],**z['interval']))
 d=pd.DataFrame(rows);d['finite']=np.isfinite(d.width);d.to_csv(G/'results/anchor_baseline_raw.tsv',sep='\t',index=False);d.groupby('scenario').agg(n=('replicate','size'),coverage=('covered','mean'),called=('called','mean'),wrong=('wrong','mean'),finite=('finite','mean')).reset_index().to_csv(G/'results/anchor_baseline_summary.tsv',sep='\t',index=False)
