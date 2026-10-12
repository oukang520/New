from pathlib import Path
import numpy as np,pandas as pd,json
P=Path(__file__).resolve().parents[2];G=P/'g4/revision_r6';des=['REDUNDANT','HETEROGENEOUS','CROSSVIEW_FAILURE'];rows=[]
for ci,ca in enumerate(['LUAD','COAD']):
 ins=[np.load(G/f'results/internal_losses_{ca}_{d}.npz') for d in des];outs=[np.load(G/f'results/patient_losses_{ca}_{d}.npz') for d in des];ni=len(ins[0]['weights']);no=len(outs[0]['patient_id']);rng=np.random.default_rng([2026092764,98,ci]);boot=[]
 for b in range(2000):
  ix=rng.integers(0,ni,ni);ox=rng.integers(0,no,no);v=[]
  for ii,oo in zip(ins,outs):
   assert len(ii['ALIGNED'])==len(oo['ALIGNED']);si=rng.integers(0,len(ii['ALIGNED']),len(ii['ALIGNED']));inside=((ii['DEASSOCIATED']-ii['ALIGNED'])[si][:,ix].sum(axis=1)/ii['weights'][ix].sum()).mean();outside=np.nanmean((oo['DEASSOCIATED']-oo['ALIGNED'])[si][:,ox],axis=1).mean();v.append(outside-inside)
  boot.append(np.mean(v))
 inside=pd.read_csv(G/'results/internal_patient_summary.tsv',sep='\t');outside=pd.read_csv(G/'results/attribution_summary.tsv',sep='\t');delta=float(outside[(outside.cancer==ca)&(outside.contrast=='pairing')].gain.iloc[0]-inside[(inside.cancer==ca)&(inside.contrast=='pairing')].gain.iloc[0]);lo,hi=np.quantile(boot,[.025,.975]);rows.append(dict(cancer=ca,external_minus_internal_pairing=delta,lo=lo,hi=hi,n_genie=ni,n_tcga=no,scope='Shared fitted-model seed resampling; independent patient resampling. Interaction in predictive contribution, not proof of a unique biological cause.'))
pd.DataFrame(rows).to_csv(G/'results/transport_interaction.tsv',sep='\t',index=False);print(pd.DataFrame(rows).to_string(index=False))
