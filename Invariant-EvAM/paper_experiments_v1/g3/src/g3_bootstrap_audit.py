"""Independent within-environment bootstrap audit of prespecified first 10 replicates."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json
import numpy as np,pandas as pd
from concurrent.futures import ProcessPoolExecutor,as_completed
from g3_model import Joint,fit
from g3_design import G

def task(cell,rep):
 p=G/'runs/core'/f'cell{cell:02d}_rep{rep:04d}.json'
 if not p.exists():return None
 o=json.loads(p.read_text());a=o['arms'][0];d=a['spec']['d'];out=[]
 for e,c in enumerate(a['data']['counts']):
  c=np.asarray(c);n=int(c.sum());spec={**a['spec'],'features':[a['spec']['features'][e]],'measurement':[a['spec']['measurement'][e]]};rng=np.random.default_rng(np.random.SeedSequence([20260923,3,303,cell,rep,e]));results=[]
  for b in range(50):
   counts=rng.multinomial(n,c/n);l=Joint(dict(counts=[counts.tolist()],calibration=None,calibration_kind=None),spec);r=fit(l);results.append(dict(bootstrap=b,estimate=r['x'][d],success=r['success'],nll=r['nll']))
  full=fit(Joint(dict(counts=[c.tolist()],calibration=None,calibration_kind=None),spec));good=[r for r in results if r['success']];truth=a['evaluator']['target'];wrong=sum((r['estimate']>0)!=(truth>0) for r in good)
  out.append(dict(cell=cell,replicate=rep,environment=e,n_bootstrap=50,n_success=len(good),wrong_sign_point_fraction=wrong/len(good) if good else None,original_estimate=full['x'][d],truth=truth,precision='BOOTSTRAP_DIAGNOSTIC_10_PRESPECIFIED_DATASETS_PER_CELL',source_attribution_allowed=False))
  (G/'runs/bootstrap'/f'cell{cell}_rep{rep}_env{e}.json').write_text(json.dumps(dict(original=full,bootstrap=results,seed=[20260923,3,303,cell,rep,e])))
 return out

def main():
 (G/'runs/bootstrap').mkdir(exist_ok=True);rows=[];missing=[]
 with ProcessPoolExecutor(max_workers=2) as ex:
  fs={ex.submit(task,c,r):(c,r) for c in [8,9,10,11] for r in range(10)}
  for f in as_completed(fs):
   out=f.result()
   if out is None:missing.append(fs[f])
   else:rows.extend(out)
 if rows:pd.DataFrame(rows).to_csv(G/'results/G3_bootstrap_stability.tsv',sep='\t',index=False)
 (G/'runs/bootstrap_STATUS.json').write_text(json.dumps(dict(complete=len(rows)==80,rows=len(rows),missing=missing),indent=2));print('bootstrap',len(rows),'missing',len(missing))
if __name__=='__main__':main()
