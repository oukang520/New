from pathlib import Path
import json
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[2]
def out(d,g,name):d.to_csv(P/f'g{g}/revision_r2/results/{name}.tsv',sep='\t',index=False)
def main():
 rows=[];status=[]
 files=list((P/'g4/revision_r2/runs').glob('*.json'));assert len(files)==700, len(files)
 for p in files:
  r=json.loads(p.read_text());common={k:r[k] for k in ['cancer','design','replicate','purpose']};status.append(dict(**common,status=r['status']))
  if r['status']!='COMPLETE':continue
  for method,m in r['methods'].items():rows.append(dict(**common,method=method,success=m['success'],chosen_ctmc_weight=r['chosen_ctmc_weight'],**m['scores']))
 d=pd.DataFrame(rows);out(d,4,'G4_R2_raw');out(pd.DataFrame(status),4,'G4_R2_status');summary=[]
 for key,q in d.groupby(['purpose','cancer','design','method']):
  z=dict(zip(['purpose','cancer','design','method'],key),n_attempted=len(q),n_success=int(q.success.sum()),mean_ctmc_weight=q.chosen_ctmc_weight.mean())
  for metric in ['source_log_score','source_tv','hidden_brier']:
   v=q[metric].dropna();z[metric]=v.mean();z[metric+'_q025']=v.quantile(.025);z[metric+'_q975']=v.quantile(.975)
  summary.append(z)
 out(pd.DataFrame(summary),4,'G4_R2_summary');pairs=[]
 for key,q in d.groupby(['purpose','cancer','design']):
  for metric in ['source_log_score','source_tv','hidden_brier']:
   wide=q.pivot(index='replicate',columns='method',values=metric)
   for reference in ['CTMC_M','UNPROJECTED_FUSION','POOLED']:
    v=(wide[reference]-wide.TRAIN_ONLY_MIXTURE).dropna()
    for rep,gain in v.items():pairs.append(dict(zip(['purpose','cancer','design'],key),reference=reference,metric=metric,replicate=rep,gain=gain))
 pair=pd.DataFrame(pairs);out(pair,4,'G4_R2_paired');ps=pair.groupby(['purpose','cancer','design','reference','metric']).gain.agg(n='size',mean='mean',low=lambda x:x.quantile(.025),high=lambda x:x.quantile(.975)).reset_index();out(ps,4,'G4_R2_paired_summary')
 rows=[];status=[];files=list((P/'g5/revision_r2/runs').glob('*.json'));assert len(files)==100,len(files)
 for p in files:
  r=json.loads(p.read_text());base=dict(cancer=r['cancer'],replicate=r['replicate']);status.append(dict(**base,status=r['status']))
  if r['status']!='COMPLETE':continue
  for z in r['results']:
   if 'strategy' not in z:continue
   v=z['result'];sc=v['scores'];b=r['baseline_scores'];rows.append(dict(**base,cap=z['cap'],budget=z['budget'],strategy=z['strategy'],candidate=z['action']['candidate_id'],success=v['fit']['success'],cost=v['counts']['cost'],new_patients=v['counts']['new_unique_patients'],remeasured=v['counts']['remeasured_existing'],target_information=v['geometry']['target_information'],null_leakage=v['geometry']['null_leakage'],source_gain=(b['source_log_score']-sc['source_log_score']) if b['source_log_score'] is not None and sc['source_log_score'] is not None else np.nan,hidden_gain=(b['hidden_brier']-sc['hidden_brier']) if b['hidden_brier'] is not None and sc['hidden_brier'] is not None else np.nan))
 d=pd.DataFrame(rows);out(d,5,'G5_R2_raw');out(pd.DataFrame(status),5,'G5_R2_status');assert (d.cost==d.budget).all()
 summaries=[]
 for key,q in d.groupby(['cancer','cap','strategy']):
  v=q.source_gain.dropna();mean=v.mean();err=1.96*v.std(ddof=1)/np.sqrt(len(v));summaries.append(dict(zip(['cancer','cap','strategy'],key),n_attempted=len(q),n_success=int(q.success.sum()),mean_cost=q.cost.mean(),min_cost=q.cost.min(),max_cost=q.cost.max(),gain_mean=mean,gain_lo=mean-err,gain_hi=mean+err,finite_information_fraction=(q.target_information>0).mean()))
 out(pd.DataFrame(summaries),5,'G5_R2_summary');pairs=[];aliases=[]
 for key,q in d.groupby(['cancer','cap']):
  wide=q.pivot(index='replicate',columns='strategy',values='source_gain');actions=q.pivot(index='replicate',columns='strategy',values='candidate')
  for ref in ['COUNT_ONLY','RANDOM_FEASIBLE','MAX_PANEL','D_OPT','C_OPT']:
   v=(wide.INVARIANT_ROBUST-wide[ref]).dropna()
   for rep,gain in v.items():pairs.append(dict(cancer=key[0],cap=key[1],reference=ref,replicate=rep,gain=gain))
  for rep,row in actions.iterrows():aliases.append(dict(cancer=key[0],cap=key[1],replicate=rep,all_four_equal=len(set(row[['MAX_PANEL','D_OPT','C_OPT','INVARIANT_ROBUST']]))==1))
 pair=pd.DataFrame(pairs);out(pair,5,'G5_R2_paired');out(pd.DataFrame(aliases),5,'G5_R2_aliases');summary=[]
 for key,q in pair.groupby(['cancer','cap','reference']):
  v=q.gain;mean=v.mean();err=1.96*v.std(ddof=1)/np.sqrt(len(v));summary.append(dict(zip(['cancer','cap','reference'],key),n=len(v),mean=mean,lo=mean-err,hi=mean+err))
 out(pd.DataFrame(summary),5,'G5_R2_paired_summary');print('G4/G5 complete: 700 and 100 prespecified runs; exact cost equality passed')
if __name__=='__main__':main()
