"""Download registered public cohorts; study selection never reads outcomes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests, json, hashlib, time
import pandas as pd
import numpy as np
O=Path(__file__).resolve().parents[1]; C=json.loads((O/'configs/PROTOCOL.json').read_text())
ENTREZ={'TP53':7157,'KRAS':3845,'STK11':6794,'KEAP1':9817,'APC':324,'PIK3CA':5290,'GATA3':2625,'MAP3K1':4214,'CDKN2A':1029,'SMAD4':4089}
API='https://www.cbioportal.org/api'
FILES=['meta_study.txt','meta_mutations.txt','data_clinical_sample.txt','data_clinical_patient.txt','data_gene_panel_matrix.txt','case_lists/cases_sequenced.txt']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def request(method,url,body=None):
 for attempt in range(3):
  try:
   r=requests.request(method,url,json=body,timeout=90)
   if r.status_code==429 or r.status_code>=500:time.sleep(2*(attempt+1));continue
   return r
  except requests.RequestException:
   if attempt==2:raise
 return r
def one(item):
 study,cancer=item;D=O/'data_external'/study;D.mkdir(exist_ok=True); receipts=[]
 for name in FILES:
  p=D/name;p.parent.mkdir(exist_ok=True)
  if p.exists():receipts.append(dict(file=name,status='cached',bytes=p.stat().st_size,sha256=sha(p)));continue
  u=f'https://raw.githubusercontent.com/cBioPortal/datahub/master/public/{study}/{name}';r=request('GET',u)
  if r.ok and r.content.startswith(b'version https://git-lfs'):
   u=f'https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/{study}/{name}';r=request('GET',u)
  row=dict(file=name,url=u,status=r.status_code,bytes=len(r.content))
  if r.ok:p.write_bytes(r.content);row['sha256']=sha(p)
  receipts.append(row)
 profiles=D/'molecular_profiles.json'
 if not profiles.exists():
  r=request('GET',f'{API}/studies/{study}/molecular-profiles');r.raise_for_status();profiles.write_bytes(r.content)
 profile=next(x['molecularProfileId'] for x in json.loads(profiles.read_text()) if x['molecularAlterationType']=='MUTATION_EXTENDED')
 samplelists=D/'sample_lists.json'
 if not samplelists.exists():
  r=request('GET',f'{API}/studies/{study}/sample-lists');r.raise_for_status();samplelists.write_bytes(r.content)
 lists=json.loads(samplelists.read_text());sl=next((x['sampleListId'] for x in lists if 'sequenced' in x['sampleListId']),None)
 if sl is None:raise RuntimeError(study+' has no sequenced sample list')
 seq=D/'sequenced_ids.json'
 if not seq.exists():
  r=request('GET',f'{API}/sample-lists/{sl}/sample-ids');r.raise_for_status();seq.write_bytes(r.content)
 genes=C['modules'][cancer];p=D/'target_mutations.json';body=dict(entrezGeneIds=[ENTREZ[g] for g in genes],sampleListId=sl)
 if not p.exists():
  u=f'{API}/molecular-profiles/{profile}/mutations/fetch?projection=DETAILED';r=request('POST',u,body);r.raise_for_status();p.write_bytes(r.content)
  receipts.append(dict(file=p.name,url=u,body=body,bytes=len(r.content),status=r.status_code,sha256=sha(p)))
 if not (D/'data_clinical_sample.txt').exists():
  u=f'{API}/studies/{study}/clinical-data?clinicalDataType=SAMPLE&projection=DETAILED';r=request('GET',u);r.raise_for_status();(D/'clinical_sample_api.json').write_bytes(r.content)
  a=pd.DataFrame(r.json());samples=a.pivot_table(index=['patientId','sampleId'],columns='clinicalAttributeId',values='value',aggfunc='first').reset_index().rename(columns={'patientId':'PATIENT_ID','sampleId':'SAMPLE_ID'})
  samples.to_csv(D/'data_clinical_sample.txt',sep='\t',index=False);receipts.append(dict(file='clinical_sample_api.json',url=u,status=r.status_code,sha256=sha(D/'clinical_sample_api.json')))
 samples=pd.read_csv(D/'data_clinical_sample.txt',sep='\t',comment='#',dtype=str)
 ids=set(json.loads(seq.read_text()));s=samples[samples.SAMPLE_ID.isin(ids)].copy()
 # Keep the whole eligible sequenced study; only pick one record per patient.
 for col,allowed in C.get('eligibility_filters',{}).get(study,{}).items():
  if col not in s:raise RuntimeError(study+' missing eligibility field '+col)
  s=s[s[col].isin(allowed)]
 if study in C.get('patient_eligibility_filters',{}):
  patients=pd.read_csv(D/'data_clinical_patient.txt',sep='\t',comment='#',dtype=str)
  for col,allowed in C['patient_eligibility_filters'][study].items():
   if col not in patients:raise RuntimeError(study+' missing patient eligibility field '+col)
   patients=patients[patients[col].isin(allowed)]
  s=s[s.PATIENT_ID.isin(patients.PATIENT_ID)]
 s=s.sort_values('SAMPLE_ID').drop_duplicates('PATIENT_ID');index={v:i for i,v in enumerate(s.SAMPLE_ID)};x=np.zeros((len(s),4),int)
 muts=json.loads(p.read_text())
 for m in muts:
  g=m['gene']['hugoGeneSymbol']
  if m['sampleId'] in index and g in genes and m['mutationType'] in C['variant_classes']:x[index[m['sampleId']],genes.index(g)]=1
 out=s[['PATIENT_ID','SAMPLE_ID']].copy()
 for col in ['SAMPLE_TYPE','ONCOTREE_CODE','CANCER_TYPE_DETAILED']:
  if col in s:out[col]=s[col]
 for j,g in enumerate(genes):out[g]=x[:,j]
 out['state']=x@np.array([8,4,2,1]);out['study']=study;out['cancer']=cancer
 out['split']=['pilot' if int(hashlib.sha256(f'{C["split_seed"]}|{study}|{p}'.encode()).hexdigest()[:16],16)/2**64<C['pilot_fraction'] else 'confirmation' for p in out.PATIENT_ID]
 out['negative_contract']=C['record_negative_contract'];out.to_csv(O/'records'/f'{study}.tsv',sep='\t',index=False)
 audit=dict(study=study,cancer=cancer,genes=genes,n_clinical=len(samples),n_sequenced=len(ids),n_unique_patients=len(out),n_pilot=int(out.split.eq('pilot').sum()),n_confirmation=int(out.split.eq('confirmation').sum()),all_zero=int((out.state==0).sum()),n_module_states=int(out.state.nunique()),has_panel_matrix=(D/'data_gene_panel_matrix.txt').exists(),sample_type=s.SAMPLE_TYPE.value_counts().to_dict() if 'SAMPLE_TYPE' in s else {},source_hashes={str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file() and p.name not in ['DATA_AUDIT.json','DOWNLOAD_RECEIPTS.json']})
 (D/'DOWNLOAD_RECEIPTS.json').write_text(json.dumps(receipts,indent=2));(D/'DATA_AUDIT.json').write_text(json.dumps(audit,indent=2));print(study,len(out),'patients',flush=True);return audit
if __name__=='__main__':
 rows=[]
 with ThreadPoolExecutor(4) as ex:
  for study,f in zip(C['studies'],ex.map(one,C['studies'].items())):
   rows.append(f);(O/'research_traceability/DATA_AUDIT.json').write_text(json.dumps(rows,indent=2))
