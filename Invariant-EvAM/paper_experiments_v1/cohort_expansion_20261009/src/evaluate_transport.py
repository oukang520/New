"""New patients, frozen model ensembles, outcome-independent perturbation laws."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np,pandas as pd
O=Path(__file__).resolve().parents[1];P=O.parent
C=json.loads((O/'configs/PROTOCOL.json').read_text())
sys.path.insert(0,str(P/'g4/revision_r4/src'));import bridge as b
TR=json.loads((P/'g4/data_private/selection_truth/operators.json').read_text())
METHODS=['MORE','ALIGNED','DEASSOCIATED','TWO_STEP','POOLED','BASE','CALIBRATED','SHUFFLED']
CONTRASTS=[('total','MORE','ALIGNED'),('margin','MORE','DEASSOCIATED'),('pairing','DEASSOCIATED','ALIGNED'),('optimizer','TWO_STEP','ALIGNED'),('model','POOLED','ALIGNED'),('calibration','BASE','CALIBRATED'),('shuffle','SHUFFLED','ALIGNED')]
def models(cancer,des):
 arr={k:[] for k in METHODS};excluded=[]
 for rep in range(50):
  z=json.loads((P/f'g4/revision_r4/runs/evaluation_{cancer}_{des}_{rep:03d}.json').read_text())
  if z['status']!='COMPLETE':excluded.append(dict(rep=rep,reason=z['status']));continue
  rz=json.loads((P/f'g4/revision_r6/runs/{cancer}_{des}_{rep:03d}.json').read_text())
  for k,m in [('MORE','MORE_CTMC'),('ALIGNED','BRIDGE_CTMC'),('TWO_STEP','BRIDGE_TWO_STEP'),('POOLED','BRIDGE_POOLED'),('BASE','BASE_CTMC'),('CALIBRATED','AUDIT_CELL_16'),('SHUFFLED','SHUFFLED_CTMC')]:arr[k].append(z['methods'][m]['probabilities'])
  arr['DEASSOCIATED'].append(rz['results']['0.0']['probabilities'])
 return {k:np.array(v) for k,v in arr.items()},excluded
def evaluate(study,cancer,stage):
 data=pd.read_csv(O/'records'/f'{study}.tsv',sep='\t');data=data[data.split==stage];n=len(data);freq=np.bincount(data.state,minlength=16)/n
 rng=np.random.default_rng([C['seed'],int(hashlib.sha256(study.encode()).hexdigest()[:8],16),stage=='confirmation']);B=C['bootstrap_replicates'];counts=rng.multinomial(n,freq,size=B)/n
 totals=[];boot=[];ledger=[];designrows=[]
 for des in C['designs']:
  spec=b.gp.DESIGNS[des];arr,excluded=models(cancer,des);ledger.append(dict(study=study,design=des,completed=len(arr['MORE']),planned=50,excluded=excluded));weight=np.array([.5*np.exp(TR[des]['alpha'][e]+np.array(spec['features'][e])@TR[des]['phi'][e]) for e in range(2)]);den=weight.sum(axis=0);numer={}
  for k,pred in arr.items():
   v=np.zeros((len(pred),16))
   for e in range(2):
    yi=np.argmax(np.array(spec['measurement'][e]),axis=0);hidden=np.flatnonzero(1-np.array(spec['masks'][e]))[0];y=b.X[:,hidden];pr=np.clip(pred[:,e,yi],1e-8,1-1e-8);v+=weight[e]*(-y*np.log(pr)-(1-y)*np.log1p(-pr))
   numer[k]=v
  si=rng.integers(0,len(arr['MORE']),(B,len(arr['MORE'])))
  point=[];bootdes=[]
  for name,a,c in CONTRASTS:
   dif=numer[a]-numer[c];estimate=float(dif.mean(axis=0)@freq/(den@freq));draw=dif[si].mean(axis=1);boots=(draw*counts).sum(axis=1)/(counts@den);point.append(estimate);bootdes.append(boots);designrows.append(dict(study=study,cancer=cancer,stage=stage,design=des,contrast=name,n_patients=n,gain=estimate))
  totals.append(point);boot.append(bootdes)
  np.savez_compressed(O/f'research_traceability/{stage}_{study}_{des}_patient_losses.npz',PATIENT_ID=data.PATIENT_ID.to_numpy(dtype=str),state=data.state.to_numpy(),selection_mass=den,**numer)
 mean=np.mean(totals,axis=0);draws=np.mean(boot,axis=0);assert np.allclose(mean[0],mean[1]+mean[2]);assert np.allclose(draws[0],draws[1]+draws[2]);rows=[]
 for j,(name,ref,cand) in enumerate(CONTRASTS):
  lo,hi=np.quantile(draws[j],[.025,.975]);rows.append(dict(study=study,cancer=cancer,stage=stage,n_patients=n,contrast=name,reference=ref,candidate=cand,gain=mean[j],lo=lo,hi=hi,positive=bool(mean[j]>0),interval_positive=bool(lo>0),bootstrap_unit='patients and original frozen fit draws, paired across methods',cost_contract='560 initial plus 78 new assays; same registered GENIE acquisition fits',evidence_type='NEW_PATIENT_FROZEN_MODEL_CONTROLLED_RECORD_PREDICTION'))
 return rows,designrows,ledger
def main(stage):
 for f,h in C['model_hashes'].items():assert hashlib.sha256((P/f).read_bytes()).hexdigest()==h
 studies=[(s,a) for s,a in C['studies'].items() if a in ['LUAD','COAD']]
 if stage=='confirmation':
  freeze=json.loads((O/'configs/ADOPTION_FREEZE.json').read_text());studies=[(s,C['studies'][s]) for s in freeze['selected_for_confirmation']]
 rows=[];cells=[];ledger=[];ineligible=[]
 for study,cancer in studies:
  df=pd.read_csv(O/'records'/f'{study}.tsv',sep='\t')
  if len(df)<C['minimum_unique_patients'] or (df.split==stage).sum()<15:ineligible.append(dict(study=study,reason='prespecified patient capacity',n=len(df)));continue
  r,c,l=evaluate(study,cancer,stage);rows+=r;cells+=c;ledger+=l;print(stage,study,[(x['contrast'],round(x['gain'],5)) for x in r],flush=True)
 pd.DataFrame(rows).to_csv(O/f'research_traceability/{stage}_transport.tsv',sep='\t',index=False);pd.DataFrame(cells).to_csv(O/f'research_traceability/{stage}_transport_by_design.tsv',sep='\t',index=False)
 (O/f'research_traceability/{stage}_ledger.json').write_text(json.dumps(dict(fits=ledger,ineligible=ineligible),indent=2))
 if stage=='pilot':
  tab=pd.DataFrame(rows);selected=tab[(tab.contrast=='total')&(tab.gain>0)].study.tolist()
  (O/'configs/ADOPTION_FREEZE.json').write_text(json.dumps(dict(selected_for_confirmation=selected,selection_rule=C['selection_rule'],pilot_results_sha256=hashlib.sha256((O/'research_traceability/pilot_transport.tsv').read_bytes()).hexdigest(),confirmation_results_inspected=False),indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['pilot','confirmation']);args=a.parse_args();main(args.stage)
