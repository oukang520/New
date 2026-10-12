from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[2];G=P/'g4/revision_r5';D=G/'data_external';C=json.loads((P/'closure_r5/configs/PROTOCOL.json').read_text());F=json.loads((P/'g4/configs/DATA_FREEZE.json').read_text());sys.path.insert(0,str(P/'g4/revision_r4/src'));import bridge as b
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
STUDIES={'LUAD':'luad_tcga_pan_can_atlas_2018','COAD':'coadread_tcga_pan_can_atlas_2018'}
METHODS=['BASE_CTMC','MORE_CTMC','BRIDGE_CTMC','AUDIT_CELL_16','BASE_POOLED','MORE_POOLED','BRIDGE_POOLED','BRIDGE_TWO_STEP','SHUFFLED_CTMC','AUDIT_MARGINAL_1']
files=list((P/'g4/revision_r4/runs').glob('evaluation_*.json'));freeze={str(f.relative_to(P)):sha(f) for f in files};(G/'configs/EXTERNAL_MODEL_FREEZE.json').write_text(json.dumps(freeze,indent=2))
AUDIT=[];RAW=[];PAT=[];matrices={}
for cancer,study in STUDIES.items():
 folder=D/study;sample=pd.read_csv(folder/'data_clinical_sample.txt',sep='\t',comment='#',dtype=str);panel=pd.read_csv(folder/'data_gene_panel_matrix.txt',sep='\t',dtype=str);text=(folder/'case_lists/cases_sequenced.txt').read_text();sequenced=set(next(x.split(':',1)[1].strip() for x in text.splitlines() if x.startswith('case_list_ids:')).split())
 eligible=sample[sample.SAMPLE_TYPE.eq('Primary')&sample.SAMPLE_ID.isin(sequenced)&sample.SAMPLE_ID.isin(panel[panel.mutations.eq('WXS')].SAMPLE_ID)].sort_values('SAMPLE_ID').drop_duplicates('PATIENT_ID');ids=eligible.SAMPLE_ID.tolist();index={s:i for i,s in enumerate(ids)};genes=F['modules'][cancer];X=np.zeros((len(ids),4),int);mut=json.loads((folder/'target_mutations_api.json').read_text())
 for v in mut:
  gene=v['gene']['hugoGeneSymbol']
  if v['sampleId'] in index and gene in genes and v['mutationType'] in F['variant_classes']:X[index[v['sampleId']],genes.index(gene)]=1
 states=X@np.array([8,4,2,1]);records=eligible[['PATIENT_ID','SAMPLE_ID','ONCOTREE_CODE']].copy()
 for j,gene in enumerate(genes):records[gene]=X[:,j]
 records['state']=states;records['negative_contract']='WXS_ELIGIBLE_NO_RECORDED_VARIANT_NOT_SITE_CALLABILITY';records.to_csv(D/f'{cancer}_records.tsv',sep='\t',index=False)
 AUDIT.append(dict(cancer=cancer,study=study,clinical_samples=len(sample),sequenced_samples=len(sequenced),eligible_unique_patients=len(ids),all_module_zero=int((states==0).sum()),genes=genes,oncotree=eligible.ONCOTREE_CODE.value_counts().to_dict(),site_callability_available=False,evolution_parameter_truth_available=False,scope='External dataset to GENIE; published recorded events only; TCGA CRC includes colon/rectal adenocarcinoma',source_hashes={f.name:sha(f) for f in folder.iterdir() if f.is_file()}))
 for design in C['g4']['designs']:
  spec=b.gp.DESIGNS[design];truth=json.loads((P/'g4/data_private/selection_truth/operators.json').read_text())[design];arr={m:[] for m in METHODS};valid_ids=[]
  for rep in range(50):
   z=json.loads((P/f'g4/revision_r4/runs/evaluation_{cancer}_{design}_{rep:03d}.json').read_text())
   if z['status']!='COMPLETE':continue
   rng=np.random.default_rng([C['seed'],7,STUDIES.keys().__iter__().__next__()=='dummy',list(STUDIES).index(cancer),rep]);env=rng.integers(0,2,len(states));sel=rng.random(len(states))<np.array([np.exp(truth['alpha'][e]+np.array(spec['features'][e])[s]@truth['phi'][e]) for e,s in zip(env,states)]);y=np.array([b.X[s,np.flatnonzero(1-np.array(spec['masks'][e]))[0]] for e,s in zip(env,states)]);visible=np.array([np.argmax(np.array(spec['measurement'][e]),axis=0)[s] for e,s in zip(env,states)])
   for m in METHODS:
    pred=np.array(z['methods'][m]['probabilities']);p=np.clip(pred[env,visible],1e-8,1-1e-8);loss=-(y*np.log(p)+(1-y)*np.log1p(-p));loss[~sel]=np.nan;arr[m].append(loss)
    RAW.append(dict(cancer=cancer,design=design,replicate=rep,method=m,n_evaluation=int(sel.sum()),log_score=float(np.nanmean(loss)),brier=float(np.mean((p[sel]-y[sel])**2))))
   valid_ids.append(rep)
  matrices[cancer,design]={m:np.array(a) for m,a in arr.items()}
  np.savez_compressed(G/f'results/external_patient_loss_{cancer}_{design}.npz',patient_id=eligible.PATIENT_ID.to_numpy(dtype=str),replicate=np.array(valid_ids),**matrices[cancer,design])
  print(cancer,design,'n',len(ids),'fits',len(valid_ids),flush=True)
(G/'results/EXTERNAL_DATA_AUDIT.json').write_text(json.dumps(AUDIT,indent=2));pd.DataFrame(RAW).to_csv(G/'results/external_raw.tsv',sep='\t',index=False)
# Fixed fitted-model ensemble; patient bootstrap handles shared patients across masks.
# Seed variability is resampled separately, with all paired method predictions kept together.
rows=[];bootrows=[]
for cancer in STUDIES:
 n=next(x['eligible_unique_patients'] for x in AUDIT if x['cancer']==cancer)
 for comparison,ref,cand in [('B','MORE_CTMC','BRIDGE_CTMC'),('C','BASE_CTMC','AUDIT_CELL_16'),('JOINT_VS_TWO_STEP','BRIDGE_TWO_STEP','BRIDGE_CTMC'),('JOINT_VS_SHUFFLE','SHUFFLED_CTMC','BRIDGE_CTMC'),('JOINT_VS_POOLED','BRIDGE_POOLED','BRIDGE_CTMC')]:
  diffs=[matrices[cancer,des][ref]-matrices[cancer,des][cand] for des in C['g4']['designs']];mean=float(np.mean([np.nanmean(v,axis=1).mean() for v in diffs]));boot=[];rng=np.random.default_rng([C['seed'],8,list(STUDIES).index(cancer)])
  for r in range(2000):
   ix=rng.integers(0,n,n);means=[]
   for v in diffs:
    si=rng.integers(0,len(v),len(v));means.append(float(np.nanmean(v[si][:,ix],axis=1).mean()))
   gain=float(np.mean(means));boot.append(gain);bootrows.append(dict(cancer=cancer,comparison=comparison,replicate=r,gain=gain))
  lo,hi=np.quantile(boot,[.025,.975]);rows.append(dict(cancer=cancer,comparison=comparison,reference=ref,candidate=cand,n_patients=n,gain=mean,lo=lo,hi=hi,scope='conditional frozen GENIE ensemble; TCGA patients and mechanism seeds resampled; equal design mean'))
pd.DataFrame(rows).to_csv(G/'results/external_paired_summary.tsv',sep='\t',index=False);pd.DataFrame(bootrows).to_csv(G/'results/external_bootstrap.tsv',sep='\t',index=False)
assert all(sha(P/f)==h for f,h in freeze.items());print('External validation complete; source fits unchanged',flush=True)
