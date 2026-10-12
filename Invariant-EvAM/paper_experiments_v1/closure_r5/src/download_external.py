from pathlib import Path
import requests,json,hashlib,time
P=Path(__file__).resolve().parents[2];G=P/'g4/revision_r5/data_external';receipts=[]
files=['data_clinical_sample.txt','data_clinical_patient.txt','data_gene_panel_matrix.txt','case_lists/cases_sequenced.txt','data_mutations.txt','meta_study.txt','meta_mutations.txt','README.md']
for study in ['luad_tcga_pan_can_atlas_2018','coadread_tcga_pan_can_atlas_2018']:
 for name in files:
  path=G/study/name;path.parent.mkdir(parents=True,exist_ok=True);url=f'https://raw.githubusercontent.com/cBioPortal/datahub/master/public/{study}/{name}'
  r=requests.get(url,timeout=90)
  if r.ok and r.content.startswith(b'version https://git-lfs'):
   url=f'https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/{study}/{name}';r=requests.get(url,timeout=180)
  row=dict(study=study,file=name,url=url,status=r.status_code,bytes=len(r.content))
  if r.ok:path.write_bytes(r.content);row['sha256']=hashlib.sha256(r.content).hexdigest()
  receipts.append(row);(G/'DOWNLOAD_RECEIPTS.json').write_text(json.dumps(receipts,indent=2));print(row,flush=True)
