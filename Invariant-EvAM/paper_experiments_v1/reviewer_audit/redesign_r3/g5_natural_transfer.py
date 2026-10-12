"""Predictive transfer with training-panel validation; no natural theta claim."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;G=P/'g5/revision_r3'
C=json.loads((D/'PROTOCOL.json').read_text());rows=json.loads((P/'g5/data_private/natural_records.json').read_text())
panels=['MSK-IMPACT468','MSK-IMPACT410','MSK-IMPACT505'];predictions=[];records=[]
def counts(rr):return np.bincount([r['reference_state'] for r in rr],minlength=16)
def pred(cal,prior,k):return (cal+.5+k*prior)/(cal.sum()+8+k)
for hi,held in enumerate(panels[1:]):
    tr=[r for r in rows if r['environment']!=held and r['natural_training_eligible']]
    cal=[r for r in rows if r['environment']==held and r['heldout_role']=='calibration']
    candidates=[0,4,16,64,256,1024];loss={k:0. for k in candidates};folds=[]
    # Ten fixed training-only pseudo-transfer tasks. Target calibration and
    # evaluation never participate in selecting the prior mass.
    for rep in range(10):
      for env in [v for v in panels if v!=held]:
        rr=[r for r in tr if r['environment']==env];other=[r for r in tr if r['environment']!=env]
        rng=np.random.default_rng(np.random.SeedSequence([C['seed'],5,hi,rep,panels.index(env)]));ids=rng.permutation(len(rr));n=min(len(cal),len(rr)//2)
        a=counts([rr[i] for i in ids[:n]]);b=counts([rr[i] for i in ids[n:]])
        prior=(counts(other)+.5)/(len(other)+8)
        for k in candidates:loss[k]+=float(-b@np.log(pred(a,prior,k)))/max(1,b.sum())
        folds.append(dict(rep=rep,training_panel=env,n_calibration=int(a.sum()),n_validation=int(b.sum())))
    k=min(candidates,key=lambda x:loss[x]);prior=(counts(tr)+.5)/(len(tr)+8);a=counts(cal)
    methods={'TRANSFER_SHRINKAGE':pred(a,prior,k),'TARGET_ONLY_DIRICHLET':pred(a,prior,0),'POOLED_TRAINING':prior}
    for m,p in methods.items():predictions.append(dict(heldout=held,method=m,probability=p.tolist(),kappa=k if m=='TRANSFER_SHRINKAGE' else 0,training_n=len(tr),calibration_n=len(cal),validation_losses=loss,folds=folds))
f=G/'configs/NATURAL_PREDICTION_FREEZE.json';f.write_text(json.dumps(dict(predictions=predictions,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='PREVIOUSLY_EXPLORED_INTERNAL_LEAVE_PANEL_OUT'),indent=2))
# The scoring section begins only after the prediction bundle is on disk.
for z in predictions:
    held=z['heldout'];ev=[r for r in rows if r['environment']==held and r['heldout_role']=='evaluation'];p=np.array(z['probability'])
    for r in ev:records.append(dict(heldout=held,method=z['method'],patient_key=r['patient_key'],state=r['reference_state'],log_score=-np.log(p[r['reference_state']])))
    print(held,z['method'],len(ev),flush=True)
# Prior matched CTMC predictions: copied with explicit legacy provenance.
for held in panels[1:]:
    for f in sorted((P/'g5/runs/heldout').glob(held+'_*.json')):
      z=json.loads(f.read_text());p=np.array(z['prediction']);ev=[r for r in rows if r['environment']==held and r['heldout_role']=='evaluation']
      for r in ev:records.append(dict(heldout=held,method='LEGACY_'+z['spec_id'],patient_key=r['patient_key'],state=r['reference_state'],log_score=-np.log(max(p[r['reference_state']],1e-280))))
d=pd.DataFrame(records);d.to_csv(G/'results/natural_patient_scores.tsv',sep='\t',index=False)
summary=[]
for held,q in d.groupby('heldout'):
    w=q.pivot(index='patient_key',columns='method',values='log_score');a=w.TRANSFER_SHRINKAGE
    for m in w:
      diff=w[m]-a;se=diff.std(ddof=1)/np.sqrt(len(diff));summary.append(dict(heldout=held,reference=m,n=len(diff),reference_log_score=w[m].mean(),transfer_log_score=a.mean(),gain=diff.mean(),lo=diff.mean()-1.96*se,hi=diff.mean()+1.96*se))
pd.DataFrame(summary).to_csv(G/'results/natural_paired_summary.tsv',sep='\t',index=False)
