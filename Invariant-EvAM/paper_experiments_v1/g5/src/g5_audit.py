from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[3];G=R/'paper_experiments_v1/g5';S=R/'paper_experiments_v1/shared';D=Path('/Users/oukang/Downloads/project/Rel_ObsHN/Data/AACR/AACR');GENES=['TP53','KRAS','STK11','KEAP1'];PANELS=['MSK-IMPACT468','MSK-IMPACT410','MSK-IMPACT505']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 if (G/'configs/DATA_FREEZE.json').exists():print('Audit frozen; no overwrite');return
 checks=[]
 for fn in [S/'G1_SHARED_ASSET_MANIFEST.json',R/'paper_experiments_v1/g2/configs/FINAL_DELIVERY_MANIFEST.json',R/'paper_experiments_v1/g3/DELIVERY_MANIFEST.json',R/'paper_experiments_v1/g4/DELIVERY_MANIFEST.json']:
  fs=json.loads(fn.read_text())['files'];items=fs.items() if isinstance(fs,dict) else [(z['path'],z) for z in fs]
  for path,v in items:
   if '/archive/' in path or '__pycache__' in path:continue
   if any(t in path for t in ['/results/','/runs/','/shared/']) or ('/src/' in path and not any(t in path.lower() for t in ['plot','figure','render'])):
    p=R/path;checks.append(dict(path=path,match=p.exists() and sha(p)==v['sha256']))
 pd.DataFrame(checks).to_csv(G/'results/DEPENDENCY_HASH_AUDIT.tsv',sep='\t',index=False);assert all(z['match'] for z in checks),[z for z in checks if not z['match']][:4]
 f4=json.loads((R/'paper_experiments_v1/g4/configs/DATA_FREEZE.json').read_text());s=pd.read_csv(D/'data_clinical_sample.txt',sep='\t',comment='#',low_memory=False);pat=pd.read_csv(D/'data_clinical_patient.txt',sep='\t',comment='#',low_memory=False);mat=pd.read_csv(D/'data_gene_matrix.txt',sep='\t');panel=pd.read_csv(D/'genomic_information.txt',sep='\t',low_memory=False);assay=pd.read_csv(D/'assay_information.txt',sep='\t')
 q=s[s.ONCOTREE_CODE.eq('LUAD')&s.SAMPLE_TYPE.eq('Primary')&s.SEQ_ASSAY_ID.isin(PANELS)].copy();legacy=q.sort_values(['PATIENT_ID','SAMPLE_ID']).drop_duplicates('PATIENT_ID');q['priority']=q.SEQ_ASSAY_ID.map(dict(zip(PANELS,range(3))));q=q.sort_values(['PATIENT_ID','priority','SAMPLE_ID']).drop_duplicates('PATIENT_ID');q=q.merge(pat,on='PATIENT_ID',validate='one_to_one').merge(mat[['SAMPLE_ID','mutations']],on='SAMPLE_ID',validate='one_to_one')
 ids=set(q.SAMPLE_ID)|set(legacy.SAMPLE_ID);hits={g:set() for g in GENES};anyrow=set()
 for part in pd.read_csv(D.parent/'data_mutations_extended.txt',sep='\t',usecols=['Hugo_Symbol','Tumor_Sample_Barcode','Variant_Classification'],chunksize=400000,low_memory=False):
  part=part[part.Tumor_Sample_Barcode.isin(ids)];anyrow.update(part.Tumor_Sample_Barcode);part=part[part.Variant_Classification.isin(f4['variant_classes'])]
  for gene,z in part[part.Hugo_Symbol.isin(GENES)].groupby('Hugo_Symbol'):hits[gene].update(z.Tumor_Sample_Barcode)
 old4={z['patient_key']:z for z in json.loads((R/'paper_experiments_v1/g4/data_private/reference_records/LUAD.json').read_text())};rows=[];audit=[];summary=[];composition=[]
 for e in PANELS:
  covered=set(panel.loc[panel.SEQ_ASSAY_ID.eq(e)&panel.includeInPanel.astype(str).str.lower().eq('true'),'Hugo_Symbol']);z=q[q.SEQ_ASSAY_ID.eq(e)];n=len(z);rr=[]
  for row in z.itertuples():
   key=hashlib.sha256(('G4_20260924|'+str(row.PATIENT_ID)).encode()).hexdigest();x=[int(row.SAMPLE_ID in hits[g]) if g in covered and row.mutations==e else -1 for g in GENES];state=int(np.array(x)@np.array([8,4,2,1])) if min(x)>=0 else None;old=old4.get(key)
   if e==PANELS[0]:assert old is not None and old['reference_state']==state
   rr.append(dict(patient_key=key,environment=e,mutation=x,measurement_status=['RECORDED_MUTATION' if a==1 else 'PANEL_ONLY_PROXY_NEGATIVE' if a==0 else 'UNMEASURED_OR_UNKNOWN' for a in x],reference_state=state,g4_role=old['role'] if old else None,any_mutation_row=row.SAMPLE_ID in anyrow))
  # Non-reference holdout split fixed by patient hash, independent of outcomes.
  rr.sort(key=lambda z:hashlib.sha256(('G5_HOLDOUT|'+z['patient_key']).encode()).digest())
  for i,row in enumerate(rr):row['heldout_role']='calibration' if i<int(.2*n) else 'evaluation';row['natural_training_eligible']=e!=PANELS[0] or row['g4_role']=='train'
  rows+=rr;xx=np.array([r['reference_state'] for r in rr if r['reference_state'] is not None]);counts=np.bincount(xx,minlength=16);summary.append(dict(environment=e,n_samples_before_dedup=int(sum(s.ONCOTREE_CODE.eq('LUAD')&s.SAMPLE_TYPE.eq('Primary')&s.SEQ_ASSAY_ID.eq(e))),n_patients=n,complete_proxy_n=len(xx),no_mutation_rows=sum(not r['any_mutation_row'] for r in rr),supported_states=int(sum(counts>0)),natural_training_n=sum(r['natural_training_eligible'] for r in rr),heldout_calibration_n=int(.2*n),heldout_evaluation_n=n-int(.2*n),strict_callable_negative_n=0))
  for gene in GENES:audit.append(dict(environment=e,event=gene,nominal_measurement=int(gene in covered),operator_on_record='IDENTITY_ON_FOUR_PROXY_EVENTS' if all(g in covered for g in GENES) else 'PARTIAL',sample_callable_proof=False,selection='UNKNOWN_NATURAL_SELECTION',treatment_available=False,sequencing_indication_available=False,sampling_date_available=False,eligibility='CONDITIONAL_PROXY_RECORD' if len(xx)>=100 and sum(counts>0)>=8 else 'INELIGIBLE'))
  for field in ['SEX','PRIMARY_RACE','ETHNICITY','AGE_AT_SEQ_REPORT']:
   for val,count in z[field].fillna('MISSING').astype(str).value_counts().items():composition.append(dict(environment=e,field=field,value=val,n=int(count),denominator=n))
 pd.DataFrame(summary).to_csv(G/'results/G5_natural_denominator.tsv',sep='\t',index=False);pd.DataFrame(audit).to_csv(G/'results/NATURAL_OPERATOR_AUDIT.tsv',sep='\t',index=False);pd.DataFrame(composition).to_csv(G/'results/G5_clinical_composition.tsv',sep='\t',index=False)
 for e in PANELS:assert next(z for z in summary if z['environment']==e)['supported_states']>=8
 assert len(set(r['patient_key'] for r in rows))==len(rows);assert sum(r['environment']==PANELS[0] for r in rows)==2242
 G.joinpath('data_private').chmod(0o700);put(G/'data_private/natural_records.json',rows);G.joinpath('data_private/natural_records.json').chmod(0o600)
 for c in ['LUAD','COAD']:assert all(sha(Path(p))==d['sha256'] for p,d in f4['source_files'].items())
 metadata=dict(source='GENIE18.0-public',source_status='PREVIOUSLY_EXPLORED',genome='GRCh37',modules=GENES,panels=PANELS,measurement='Panel-only proxy negatives; four-target A identical; not sample callability',dedup='Prefer468 to preserve frozen G4 patient roles, then410 then505, then sample lexical order; no genotype dependence',g4_data_freeze_sha256=sha(R/'paper_experiments_v1/g4/configs/DATA_FREEZE.json'),private_sha256=sha(G/'data_private/natural_records.json'),historical_lexical_dedup_n=len(legacy),historical_lexical_anyrow_n=int(sum(legacy.SAMPLE_ID.isin(anyrow))),current_n=len(rows),current_no_mutation_rows=sum(not r['any_mutation_row'] for r in rows),g4_reference_roles_preserved=True,natural_reference_training='Only G4 train1345; calibration/evaluation remain reserved for controlled design; no natural reference holdout target recentering',clinical_fields_available=['SEX','PRIMARY_RACE','ETHNICITY','AGE_AT_SEQ_REPORT'],sampling_time_missing=True,no_reweighting='Descriptive measured composition only; no fabricated adjustment',dependency_hashes=len(checks))
 put(G/'configs/DATA_FREEZE.json',metadata);print(pd.DataFrame(summary).to_string(index=False));print('Previous4737 reconstruction:',metadata['historical_lexical_anyrow_n'],'dependency hashes',len(checks))
if __name__=='__main__':main()
