"""MODEL_ANCHORED only: source-bootstrap uncertainty, separate from real-record evidence."""
from pathlib import Path
import json,argparse,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from g4_pipeline import G,P,DESIGNS,records,rng,put
from g4_estimator import full_reference,train,snapshot

def one(task):
 cancer,draw=task;path=G/f'runs/auxiliary/{cancer}_{draw:02d}.json'
 if path.exists():return cancer,draw,'CACHED'
 rows=[z for z in records(cancer) if z['role']!='evaluation'];random=rng('auxiliary',cancer,draw,1000);x=np.array([z['reference_state'] for z in rows]);idx=random.integers(0,len(x),len(x));cnt=np.bincount(x[idx],minlength=16);sources={str(b):full_reference(cnt,profile=False,bound=b) for b in [3,6]};out=[];spec=DESIGNS['HETEROGENEOUS'];truth=json.loads((G/'data_private/selection_truth/operators.json').read_text())['HETEROGENEOUS'];ntrain=sum(z['role']=='train' for z in records(cancer))
 for b,sf in sources.items():
  if not sf['success']:continue
  p=np.array(sf['source']);theta=sf['x'][:10]
  for rep in range(P['auxiliary_replicates_per_draw']):
   rand=rng('auxiliary',cancer,draw,2000+rep);xx=rand.choice(16,ntrain,p=p);env=rand.integers(0,2,ntrain);u=rand.random(ntrain);data=dict(counts=[],calibration=[],calibration_kind='XS')
   for e in range(2):
    f=np.array(spec['features'][e]);s=np.exp(truth['alpha'][e]+f@truth['phi'][e]);use=(env==e)&(u<s[xx]);yy=np.argmax(np.array(spec['measurement'][e])[:,xx[use]],axis=0);data['counts'].append(np.bincount(yy,minlength=8).tolist());cx=rand.choice(16,40,p=p);cs=rand.random(40)<s[cx];data['calibration'].append(np.bincount(cx+16*cs,minlength=32).tolist())
   for method in ['M','B3','B4','K0']:
    ms={**spec}
    if method=='K0':ms['known_phi']=truth['phi']
    r=train(data,ms,method,profile=method in ['M','K0']);r['true_target']=float(theta[4]);r['covers_truth']=r['lo']<=theta[4]<=r['hi'] if r.get('interval_status')=='COMPLETE' else None;r['source_tv']=float(.5*np.abs(np.array(r['source'])-p).sum()) if r['success'] else None;out.append(dict(bound=int(b),replicate=rep,method=method,result=r))
 put(path,dict(cancer=cancer,draw=draw,evidence_type='MODEL_ANCHORED_SOURCE_BOOTSTRAP',source_fits=sources,source_training_n=len(x),evaluation_patients_used=False,expanded_source_distribution_tv=float(.5*np.abs(np.array(sources['3']['source'])-np.array(sources['6']['source'])).sum()),results=out));return cancer,draw,len(out)
def main():
 a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=4);args=a.parse_args();assert (G/'configs/EVALUATION_FREEZE.json').exists()
 with ProcessPoolExecutor(max_workers=args.workers) as ex:
  for i,f in enumerate(as_completed([ex.submit(one,(c,d)) for c in ['LUAD','COAD'] for d in range(P['auxiliary_source_draws'])]),1):print(i,'/20',*f.result(),flush=True)
if __name__=='__main__':main()
