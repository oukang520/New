"""Predeclared sensitivity grid: 20 replicate IDs per cell, diagnostic precision only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,time,copy
import numpy as np,pandas as pd
from g3_design import G,cells,specifications
from g3_model import Joint,inference
from concurrent.futures import ProcessPoolExecutor,as_completed

def job(cell,rep):
 p=G/'runs/core'/f'cell{cell:02d}_rep{rep:04d}.json'
 if not p.exists():return None
 o=json.loads(p.read_text());data=o['arms'][0]['data'];truth=o['arms'][0]['evaluator']['target'];base,flex=specifications(cells()[cell]);rows=[];raw=[]
 for eps in [0,.2,.5]:
  for eta in [0,.5,1.]:
   spec={**(base if eta==0 else flex),'epsilon':eps};l=Joint(data,spec)
   if eta>0:
    for sl in l.phi:l.bounds[sl.stop-1]=(-eta,eta)
   result=inference(l);raw.append(dict(epsilon=eps,eta=eta,result=result));defined=result['status']=='COMPLETE';rows.append(dict(cell=cell,replicate=rep,epsilon_log_rate=eps,eta_selection_coefficient=eta,true_mechanism_in_allowed_set=bool((cell==3 and eps>=.5) or (cell==10 and eta>=1)),status=result['status'],lo=result.get('lo'),hi=result.get('hi'),sign=result.get('sign'),coverage_defined=defined,covers_truth=bool(result['lo']<=truth<=result['hi']) if defined else None,interval_type='LOCAL_FORWARD_SENSITIVITY_ENVELOPE',precision='20_PREDECLARED_REPLICATES_DIAGNOSTIC_ONLY',source_attribution_allowed=False))
 (G/'runs/sensitivity'/f'cell{cell}_rep{rep}.json').write_text(json.dumps(raw));return rows

def main():
 (G/'runs/sensitivity').mkdir(exist_ok=True);rows=[];missing=[]
 jobs=[(c,r) for c in [3,10] for r in range(20)]
 with ProcessPoolExecutor(max_workers=2) as ex:
  fs={ex.submit(job,c,r):(c,r) for c,r in jobs}
  for f in as_completed(fs):
   out=f.result()
   if out is None:missing.append(fs[f])
   else:rows.extend(out)
 if rows:pd.DataFrame(rows).to_csv(G/'results/G3_sensitivity_regions.tsv',sep='\t',index=False)
 (G/'runs/sensitivity_STATUS.json').write_text(json.dumps(dict(complete=len(rows)==360,rows=len(rows),missing=missing),indent=2));print('sensitivity',len(rows),'missing',len(missing))
if __name__=='__main__':main()
