import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,time,sys,argparse,hashlib,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from g3_model import Joint,inference
from g3_design import cells,arms,G,R
from g3_generator import generate

def task(cell,rep,purpose,folder):
 target=G/'runs'/folder/f"cell{cell['cell_id']:02d}_rep{rep:04d}.json"
 if target.exists():return {'cached':True,'cell':cell['cell_id'],'rep':rep}
 start=time.time();results=[]
 for arm in arms(cell):
  t=time.time();data,meta=generate(cell,arm,rep,purpose)
  try:
   lik=Joint(data,arm['spec']);o=inference(lik);o['prediction_KL']=float(np.mean([sum(np.asarray(p)*(np.log(np.maximum(p,1e-280))-np.log(np.maximum(out[1],1e-280)))) for p,out in zip(meta['population_selected'],[z for i,z in enumerate(lik.outputs(np.array(o['x']))) if not data['calibration_kind'] or i%2==0])]))
   targetvalue=meta['target'];defined=targetvalue is not None and o['status']=='COMPLETE';covered=bool(o['lo']<=targetvalue<=o['hi']) if defined else None
   called=o['sign'] in ['POSITIVE','NEGATIVE'];correct=bool(called and ((o['sign']=='POSITIVE' and targetvalue>0) or (o['sign']=='NEGATIVE' and targetvalue<0))) if targetvalue is not None else None
   phi_error=[]
   for e,sl in enumerate(lik.phi):
    if sl is not None:
     q=min(len(meta['phi_true'][e]),sl.stop-sl.start);phi_error.extend((np.array(o['x'])[sl][:q]-np.array(meta['phi_true'][e])[:q]).tolist())
   o.update(coverage_defined=defined,covers_truth=covered,called=called,correct_call=correct,wrong_call=bool(called and not correct) if correct is not None else None,nuisance_rmse=float(np.sqrt(np.mean(np.square(phi_error)))) if phi_error else None)
  except Exception as exc:o=dict(status='EXCEPTION',error=repr(exc),traceback=traceback.format_exc(),source_attribution_allowed=False)
  results.append(dict(method=arm['name'],spec=arm['spec'],data=data,evaluator=meta,result=o,runtime_seconds=time.time()-t))
 obj=dict(protocol='G3_1.0',evidence_type='NEW_RUN' if purpose==302 else 'DEVELOPMENT_ONLY',cell=cell,replicate=rep,purpose=purpose,arms=results,runtime_seconds=time.time()-start)
 tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(obj));tmp.replace(target)
 return dict(cell=cell['cell_id'],rep=rep,seconds=obj['runtime_seconds'],statuses=[a['result']['status'] for a in results])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--development',action='store_true');ap.add_argument('--workers',type=int,default=6);args=ap.parse_args();folder='development' if args.development else 'core';purpose=301 if args.development else 302;(G/'runs'/folder).mkdir(exist_ok=True)
 if not args.development:
  contract=json.loads((G/'configs/G3_PROTOCOL.json').read_text());assert contract['status']=='FROZEN'
  for path,digest in contract['source_hashes'].items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==digest,path
 jobs=[(c,r) for c in cells() for r in range(1 if args.development else c['replicates'])];done=0;start=time.time()
 with ProcessPoolExecutor(max_workers=args.workers) as ex:
  futures=[ex.submit(task,c,r,purpose,folder) for c,r in jobs]
  for f in as_completed(futures):
   result=f.result();done+=1
   if args.development or done%50==0 or done==len(jobs):
    status=dict(run_status='COMPLETE' if done==len(jobs) else 'RUNNING',mode=folder,completed=done,total=len(jobs),elapsed_seconds=time.time()-start,last=result);(G/'runs'/f'{folder}_STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
if __name__=='__main__':main()
