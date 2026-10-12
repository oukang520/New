from pathlib import Path
import sys,json,time,argparse,functools
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[2];G=P/'g4/revision_r5';CFG=json.loads((P/'closure_r5/configs/PROTOCOL.json').read_text());C=CFG['g4'];sys.path.insert(0,str(P/'g4/revision_r4/src'));import bridge as b
ORIGINAL_RECORDS=b.gp.records
@functools.lru_cache(None)
def boot_records(cancer,outer):
 rows=ORIGINAL_RECORDS(cancer);rng=np.random.default_rng([CFG['seed'],4,10,C['cancers'].index(cancer),outer]);out=[]
 for role in ['train','calibration','evaluation']:
  r=[z for z in rows if z['role']==role];out += [r[i] for i in rng.integers(0,len(r),len(r))]
 return out

def one(task):
 cancer,outer,inner,design=task;path=G/f'runs/nested_{cancer}_{outer:03d}_{inner:02d}_{design}.json'
 if path.exists():return 'cached'
 start=time.time();b.gp.records=lambda c:boot_records(c,outer);b.SEED=CFG['seed']+10000+outer;ds,ev,stats=b.generate(cancer,design,inner,'evaluation');z=dict(cancer=cancer,outer=outer,inner=inner,design=design,stats=stats,status='CAPACITY_FAILURE',methods={})
 if ds is not None:
  spec=b.gp.DESIGNS[design];truth=json.loads((P/'g4/data_private/selection_truth/operators.json').read_text())[design];z['status']='COMPLETE';z['public_inputs']=ds
  for arm in ['BASE','MORE','BRIDGE']:
   r=b.estimate(ds[arm],spec,'CTMC');pred=b.pred(r,spec);score,_=b.score(pred,r['source'],ev,spec,truth);z['methods'][arm+'_CTMC']=dict(fit_success=r['success'],scores=score)
   if arm=='BASE':base=r;basep=pred
  pred=b.adjust(basep,ds['BRIDGE']['bridge'],spec,'CELL',16.);score,_=b.score(pred,base['source'],ev,spec,truth);z['methods']['AUDIT_CELL_16']=dict(fit_success=base['success'],scores=score)
 z['runtime']=time.time()-start;path.write_text(json.dumps(z));return z['status']

def finish():
 docs=[json.loads(f.read_text()) for f in sorted((G/'runs').glob('nested_*.json'))];rows=[];ledger=[]
 for z in docs:
  key={k:z[k] for k in ['cancer','outer','inner','design']};ledger.append(dict(**key,status=z['status']))
  for m,r in z['methods'].items():rows.append(dict(**key,method=m,fit_success=r['fit_success'],**r['scores']))
 pd.DataFrame(rows).to_csv(G/'results/nested_raw.tsv',sep='\t',index=False);pd.DataFrame(ledger).to_csv(G/'results/nested_ledger.tsv',sep='\t',index=False)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=2);args=a.parse_args();tasks=[(c,o,i,d) for c in C['cancers'] for o in range(C['outer_patient_bootstraps']) for i in range(C['inner_mechanism_seeds']) for d in C['designs']]
 with ProcessPoolExecutor(args.workers) as pool:
  for i,status in enumerate(pool.map(one,tasks),1):
   if i%50==0:print(i,len(tasks),status,flush=True)
 finish()
