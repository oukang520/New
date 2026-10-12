# coding: utf-8
from pathlib import Path
import json,hashlib,os,sys,platform
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[3];G=R/'paper_experiments_v1/g4';SRC=Path('/Users/oukang/Downloads/project/Rel_ObsHN/Data/AACR');D=SRC/'AACR'
MODULES={'LUAD':['TP53','KRAS','STK11','KEAP1'],'COAD':['APC','TP53','KRAS','PIK3CA'],'IDC':['PIK3CA','TP53','GATA3','CDH1'],'PAAD':['KRAS','TP53','SMAD4','CDKN2A']}
NONSYN=['Missense_Mutation','Nonsense_Mutation','Frame_Shift_Del','Frame_Shift_Ins','In_Frame_Del','In_Frame_Ins','Splice_Site','Translation_Start_Site','Nonstop_Mutation']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def writej(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def main():
 if (G/'configs/DATA_FREEZE.json').exists():print('Data audit already frozen; no overwrite');return
 for p in [G/'data_private',G/'data_private/reference_records',G/'data_private/selection_truth']:p.chmod(0o700)
 patients=pd.read_csv(D/'data_clinical_patient.txt',sep='\t',comment='#',low_memory=False);samples=pd.read_csv(D/'data_clinical_sample.txt',sep='\t',comment='#',low_memory=False);matrix=pd.read_csv(D/'data_gene_matrix.txt',sep='\t',low_memory=False)
 assert patients.PATIENT_ID.is_unique and samples.SAMPLE_ID.is_unique and matrix.SAMPLE_ID.is_unique
 assert set(samples.PATIENT_ID)<=set(patients.PATIENT_ID)
 s=samples.merge(matrix[['SAMPLE_ID','mutations']],on='SAMPLE_ID',how='left',validate='one_to_one')
 panel=pd.read_csv(D/'genomic_information.txt',sep='\t',low_memory=False);panel=panel[(panel.SEQ_ASSAY_ID=='MSK-IMPACT468')&panel.includeInPanel.astype(str).str.lower().eq('true')];covered=set(panel.Hugo_Symbol)
 parts={};flow=[];allids=set()
 for cancer,genes in MODULES.items():
  q=s[s.ONCOTREE_CODE==cancer].copy();steps=[('cancer_manifest',len(q),q.PATIENT_ID.nunique())]
  q=q[q.SAMPLE_TYPE=='Primary'];steps.append(('primary',len(q),q.PATIENT_ID.nunique()))
  q=q[q.SEQ_ASSAY_ID=='MSK-IMPACT468'];steps.append(('uniform_panel',len(q),q.PATIENT_ID.nunique()))
  q=q.sort_values(['PATIENT_ID','SAMPLE_ID']).drop_duplicates('PATIENT_ID');steps.append(('one_lexicographic_sample_per_patient',len(q),q.PATIENT_ID.nunique()))
  # Missing/discordant mutation assay assignment is unknown, never silently negative.
  q['assay_link_ok']=q.mutations.eq(q.SEQ_ASSAY_ID);parts[cancer]=q;allids.update(q.SAMPLE_ID)
  for stage,ns,npats in steps:flow.append(dict(cancer=cancer,stage=stage,n_samples=int(ns),n_patients=int(npats)))
 anyrow=set();hits={g:set() for genes in MODULES.values() for g in genes};oldhits={g:set() for g in hits};classes={};builds={};varianttypes={};selected_mut_rows=0
 for chunk in pd.read_csv(SRC/'data_mutations_extended.txt',sep='\t',usecols=['Hugo_Symbol','Tumor_Sample_Barcode','Variant_Classification','Variant_Type','NCBI_Build'],chunksize=400000,low_memory=False):
  q=chunk[chunk.Tumor_Sample_Barcode.isin(allids)];anyrow.update(q.Tumor_Sample_Barcode);selected_mut_rows+=len(q)
  for col,dst in [('Variant_Classification',classes),('NCBI_Build',builds),('Variant_Type',varianttypes)]:
   for k,v in q[col].fillna('MISSING').astype(str).value_counts().items():dst[k]=dst.get(k,0)+int(v)
  for gene,z in q[q.Hugo_Symbol.isin(hits)].groupby('Hugo_Symbol'):
   oldhits[gene].update(z.Tumor_Sample_Barcode);hits[gene].update(z.loc[z.Variant_Classification.isin(NONSYN),'Tumor_Sample_Barcode'])
 ledger=[];states=[];measure=[];splits={};legacy=[];privatehash={};exclusions=[]
 for cancer,q in parts.items():
  genes=MODULES[cancer];rows=[]
  for row in q.itertuples():
   nom=[g in covered and row.assay_link_ok for g in genes];bits=[int(row.SAMPLE_ID in hits[g]) if nom[j] else -1 for j,g in enumerate(genes)];status=['RECORDED_MUTATION' if bits[j]==1 else 'PANEL_ONLY_PROXY_NEGATIVE' if nom[j] else 'UNMEASURED_OR_UNKNOWN' for j in range(4)];reason=['' if v=='RECORDED_MUTATION' else 'NO_SAMPLE_CALLABILITY_PROOF' if v=='PANEL_ONLY_PROXY_NEGATIVE' else 'ASSAY_LINK_OR_PANEL_MISSING' for v in status];pid=hashlib.sha256(('G4_20260924|'+str(row.PATIENT_ID)).encode()).hexdigest()
   rows.append(dict(patient_key=pid,mutation=bits,measurement_status=status,missing_reason=reason,any_mutation_row=row.SAMPLE_ID in anyrow,reference_state=int(np.array(bits)@np.array([8,4,2,1])) if min(bits)>=0 else None))
  eligible=[z for z in rows if z['reference_state'] is not None];missing=len(rows)-len(eligible);oldids=[z.SAMPLE_ID for z in q.itertuples() if z.SAMPLE_ID in anyrow];oldmatrix=np.array([[int(sid in oldhits[g]) for g in genes] for sid in oldids]);legacy.append(dict(cancer=cancer,legacy_any_row_count=len(oldids),legacy_all_annotation_prevalence=oldmatrix.mean(0).tolist(),current_manifest_count=len(q),no_mutation_rows_included=sum(not z['any_mutation_row'] for z in eligible)))
  x=np.array([z['mutation'] for z in eligible]);n=len(x);counts=np.bincount([z['reference_state'] for z in eligible],minlength=16);prev=x.mean(0);qualified=n>=700 and np.all((prev>=.02)&(prev<=.85)) and np.count_nonzero(counts)>=8
  ledger.append(dict(cancer=cancer,n_manifest_deduplicated=len(q),n_proxy_complete=n,n_partial_or_unknown=missing,n_strict_callable_complete=0,nominal_panel_covered=all(g in covered for g in genes),n_no_mutation_rows=sum(not z['any_mutation_row'] for z in eligible),n_zero_target_records=int(counts[0]),state_support=int(np.count_nonzero(counts)),prevalence=json.dumps(dict(zip(genes,prev))),eligible='CONDITIONAL_PROXY_RECORD' if qualified else 'INELIGIBLE',evidence_type='PREVIOUSLY_EXPLORED_SOURCE_NEW_DIRECT_RECORD_RUN',reason='Sample-level callability unavailable; reference is recorded nonsynonymous event pattern' if qualified else 'Prespecified frequency/sample/support eligibility failed; no replacement module'))
  for j,gene in enumerate(genes):
   for status in ['RECORDED_MUTATION','VERIFIED_CALLABLE_NEGATIVE','PANEL_ONLY_PROXY_NEGATIVE','UNMEASURED_OR_UNKNOWN']:measure.append(dict(cancer=cancer,gene=gene,status=status,n=sum(z['measurement_status'][j]==status for z in rows)))
  for k,v in enumerate(counts):states.append(dict(cancer=cancer,state=k,n=int(v),denominator=n))
  for z in rows:
   if z['reference_state'] is None:exclusions.append(dict(patient_key=z['patient_key'],cancer=cancer,reason='UNKNOWN_MEASUREMENT_REFERENCE'))
  if cancer not in ['LUAD','COAD']:continue
  if not qualified:continue
  eligible.sort(key=lambda z:hashlib.sha256(('ROLE_G4_V1|'+z['patient_key']).encode()).digest());nt=int(.6*n);nc=int(.2*n)
  for i,z in enumerate(eligible):z['role']='train' if i<nt else 'calibration' if i<nt+nc else 'evaluation'
  fp=G/f'data_private/reference_records/{cancer}.json';writej(fp,eligible);fp.chmod(0o600);privatehash[cancer]=sha(fp);splits[cancer]={role:sum(z['role']==role for z in eligible) for role in ['train','calibration','evaluation']};splits[cancer]['patient_key_manifest_sha256']=hashlib.sha256(json.dumps([(z['patient_key'],z['role']) for z in eligible]).encode()).hexdigest()
 pd.DataFrame(flow).to_csv(G/'results/G4_patient_flow.tsv',sep='\t',index=False);pd.DataFrame(ledger).to_csv(G/'results/DATASET_ELIGIBILITY_LEDGER.tsv',sep='\t',index=False);pd.DataFrame(states).to_csv(G/'results/G4_state_counts.tsv',sep='\t',index=False);pd.DataFrame(measure).to_csv(G/'results/G4_measurement_qualification.tsv',sep='\t',index=False);pd.DataFrame(legacy).to_csv(G/'results/G4_legacy_denominator_reconciliation.tsv',sep='\t',index=False);writej(G/'data_private/reference_records/exclusions.json',exclusions)
 files=[D/f for f in ['data_clinical_patient.txt','data_clinical_sample.txt','data_gene_matrix.txt','assay_information.txt','genomic_information.txt','manifest.csv','meta_study.txt','meta_mutations_extended.txt']]+[SRC/'data_mutations_extended.txt']
 provenance={str(f):{'sha256':sha(f),'bytes':f.stat().st_size} for f in files}
 writej(G/'configs/DATA_FREEZE.json',dict(status='FROZEN_BEFORE_METHOD_DEVELOPMENT',source='GENIE18.0-public',source_files=provenance,patient_manifest_n=len(patients),sample_manifest_n=len(samples),modules=MODULES,event_definition='Reported nonsynonymous mutation row; SNV/indel only; no CNA/SV; no driver inference',variant_classes=NONSYN,observed_build_counts=builds,observed_variant_classes=classes,observed_variant_types=varianttypes,n_selected_mutation_rows=selected_mut_rows,negative_contract='PANEL_ONLY_PROXY_NEGATIVE_NOT_VERIFIED',eligibility=dict(min_n=700,min_prevalence=.02,max_prevalence=.85,min_states=8),split=splits,private_records_sha256=privatehash,selection_rules='Primary + IMPACT468 + lexicographic sample per patient; include no-mutation-record samples; exclude only unknown assay/reference',changed_from_legacy=['retain no-mutation-row samples','nonsynonymous definition explicit rather than any reported gene row'],evaluation_use='Only qualification summaries; no fitting, tuning or scoring'))
 print(pd.DataFrame(ledger)[['cancer','n_manifest_deduplicated','n_proxy_complete','n_no_mutation_rows','eligible']].to_string(index=False));print('Frozen split counts',splits)
if __name__=='__main__':main()
