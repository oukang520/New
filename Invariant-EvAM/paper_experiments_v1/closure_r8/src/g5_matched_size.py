"""Prespecified independent sensitivity: equal initial sample sizes, different selection intensity."""
from pathlib import Path
import json,hashlib
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import run_g5 as m
BASE=m.G;OUT=BASE/'matched_size';OUT.mkdir(exist_ok=True);(OUT/'runs').mkdir(exist_ok=True);original=m.design

def matched(case):
 base,pilot,actions=original(case)
 if case==0:
  base={'SELECT_A':2000,'SELECT_B':2000};pilot={'SELECT_A':500,'SELECT_B':500};actions={'MORE_SAME':{'SELECT_A':3500,'SELECT_B':3500},'REFERENCE_MARGIN':{'SELECT_A':2000,'SELECT_B':2000,'MARGINAL_A':6000},'NEW_SOURCE':{'SELECT_A':2000,'SELECT_B':2000,'SOURCE':1000}}
 return base,pilot,actions
m.design=matched;m.G=OUT
if __name__=='__main__':
 f=BASE/'configs/MATCHED_SIZE_FREEZE.json';h=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 if f.exists():assert json.loads(f.read_text())['code']==h
 else:f.write_text(json.dumps(dict(code=h,replicates=list(range(1002,1032)),cases=[0,3],baseline_per_environment=2000,pilot_per_environment=500,budget=6000,reason='separate the effect of strong ascertainment from differences in initial sample size; independent sensitivity, not a replacement for primary'),indent=2))
 with ProcessPoolExecutor(2) as ex:
  for i,_ in enumerate(ex.map(m.one,[('evaluation',c,r) for c in [0,3] for r in range(1002,1032)]),1):
   if i%10==0:print(i,60,flush=True)
 rows=[]
 for p in (OUT/'runs').glob('*.json'):
  z=json.loads(p.read_text())
  for a in z['outcomes']:rows.append(dict(case=z['case'],replicate=z['replicate'],choice=z['choice'],action=a['action'],chosen=a['action']==z['choice'],**a['interval']))
 d=pd.DataFrame(rows);d.to_csv(OUT/'results.tsv',sep='\t',index=False);print(d[d.chosen].groupby(['case','choice']).size().to_string());print(d[d.chosen].groupby('case').agg(coverage=('covered','mean'),finite=('finite','mean'),width=('restricted_width','mean')).to_string())
