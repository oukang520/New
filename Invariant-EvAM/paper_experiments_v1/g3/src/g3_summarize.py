from pathlib import Path
import sys,json,math,hashlib
import numpy as np,pandas as pd
from g3_design import G,R,cells,arms

def wilson(k,n):
 if not n:return (np.nan,np.nan)
 z=1.95996398454;p=k/n;mid=(p+z*z/(2*n))/(1+z*z/n);h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return mid-h,mid+h

def main():
 rows=[];failures=[];corrections=[]
 model_hash=hashlib.sha256((R/'paper_experiments_v1/shared/MODEL_CONTRACT.json').read_bytes()).hexdigest()
 primary_paths=sorted(p for p in (G/'runs/core').glob('*.json') if not p.name.startswith('cell04_'))+sorted((G/'runs/alignment_revision').glob('*.json'))
 for p in primary_paths:
  obj=json.loads(p.read_text());c=obj['cell']
  for a in obj['arms']:
   o=a['result'];ev=a['evaluator'];truth=ev['target'];est=o.get('estimate');complete=o['status']=='COMPLETE';finite=complete and np.isfinite([o.get('lo'),o.get('hi')]).all()
   collected=sum(int(np.asarray(z).sum()) for z in ev.get('raw_subgroup_counts',[])) if ev.get('raw_subgroup_counts') else ev['n_observed']
   corrected_kl=o.get('prediction_KL')
   if a['method']=='COARSE_SUBGROUP' and o.get('x') is not None:
    from g3_model import Joint,snapshot
    from g3_design import theta
    t=theta(c['d']);t2=t.copy();t2[c['d']]+=.8;p0=snapshot(t,c['d'])[0];p1=snapshot(t2,c['d'])[0]
    lik=Joint(a['data'],a['spec']);values=[]
    for e,(_,pred,_) in enumerate(lik.outputs(np.asarray(o['x']))):
     pi=.3 if e==0 else .7;pop=.8*pi*p0+.2*(1-pi)*p1;f=np.asarray(a['spec']['features'][e]);w=np.exp(f@np.array([.5]));pop=pop*w/(pop@w)
     values.append(float(sum(pop*(np.log(np.maximum(pop,1e-280))-np.log(np.maximum(pred,1e-280))))))
    corrected_kl=float(np.mean(values));corrections.append(dict(cell=c['cell_id'],replicate=obj['replicate'],method=a['method'],field='population_prediction_KL',old=o.get('prediction_KL'),new=corrected_kl,reason='Evaluate coarse-label mixture against its actual conditional observable distribution; raw generator/fits unchanged'))
   row=dict(protocol_version=obj['protocol'],group_id='G3',experiment_id=c['group'],cell_id=c['cell_id'],mechanism=c['kind'],model='D'+str(c['d'])+'_SPARSE'+str(c['d']+2 if c['d']==3 else 10),method_id=a['method'],model_contract_hash=model_hash,operator_id='G3_FROZEN_SENTINEL_'+str(c['cell_id']),operator_knowledge='K0_KNOWN' if 'known_phi' in a['spec'] else 'K1_STRUCTURED_OR_DECLARED_MISSPECIFICATION',target_id='interaction_0_to_1',evidence_type='NEW_RUN',truth_available=truth is not None,truth=truth,n_unique_patients=collected+ev['n_calibration'],n_collected_records=collected+ev['n_calibration'],n_observed=ev['n_observed'],n_screened=ev['n_screened_calibration'],n_calibration=ev['n_calibration'],budget_basis='PROXY_ACQUISITION_AND_MEASUREMENT_UNITS_NOT_SCREENED_CANDIDATES',budget_value=ev['n_budget_used'],seed=json.dumps(ev['seed_sequence']),parameter_draw_id=0,replicate_id=obj['replicate'],split_id='INDEPENDENT_PURPOSE_'+str(obj['purpose']),estimate=est,estimate_error=est-truth if truth is not None and est is not None and o.get('success',False) else None,ci_lower=o.get('lo'),ci_upper=o.get('hi'),interval_type=o.get('interval_type'),coverage_defined=o.get('coverage_defined',False),covers_truth=o.get('covers_truth'),structural_status='LOCAL_TARGET_NULL_LEAKAGE' if o.get('null_leakage',0)>1e-7 else 'LOCAL_GEOMETRY_ONLY',null_leakage=o.get('null_leakage'),rank=o.get('rank'),target_effective_information=o.get('target_effective_information'),sigma_min_per_sample=None,sign_call=o.get('sign','NUMERICAL_FAILURE'),called=o.get('called',False),correct_call=o.get('correct_call'),wrong_call=o.get('wrong_call'),computational_status=o['status'],fit_success=o.get('success',False),compatibility_status=o.get('compatibility_status'),deviance=o.get('deviance'),population_prediction_KL=corrected_kl,boundary_hit=o.get('boundary_hit',False),finite_width=o['hi']-o['lo'] if finite else None,infinite_interval=complete and not finite,nuisance_rmse=o.get('nuisance_rmse'),runtime_seconds=a['runtime_seconds'],peak_memory_mb=None,reason=o.get('error') or ('No natural shared CTMC target' if truth is None else ''),source_attribution_allowed=False,raw_file=str(p.relative_to(R)))
   rows.append(row)
   if not complete:failures.append(row)
 (G/'results/EVALUATOR_CORRECTIONS.json').write_text(json.dumps(corrections,indent=2))
 df=pd.DataFrame(rows);df.to_csv(G/'results/G3_replicate_results.tsv.gz',sep='\t',index=False,compression='gzip')
 # Arrow comes from the independently isolated local G2 wheel directory.
 sys.path.insert(0,str(R/'paper_experiments_v1/g2/vendor'))
 import pyarrow as pa,pyarrow.parquet as pq
 pq.write_table(pa.Table.from_pandas(df,preserve_index=False),G/'results/G3_replicate_results.parquet')
 pq.write_table(pa.Table.from_pandas(df[df.experiment_id=='G3D'],preserve_index=False),G/'results/G3_misspecification_results.parquet')
 summaries=[]
 for (cell,method),f in df.groupby(['cell_id','method_id'],sort=True):
  n=len(f);done=f.computational_status.eq('COMPLETE');defined=f.coverage_defined.eq(True);nc=int(defined.sum());covered=int(f.covers_truth.eq(True).sum());called=int(f.called.eq(True).sum());validcall=f.called.eq(True)&f.truth_available;nv=int(validcall.sum());wrong=int(f.wrong_call.eq(True).sum());correct=int(f.correct_call.eq(True).sum());lo,hi=wilson(covered,nc);cl,ch=wilson(correct,n);rl,rh=wilson(wrong,nv)
  errors=f.estimate_error.dropna();width=f.finite_width.dropna()
  summaries.append(dict(cell_id=int(cell),experiment_id=f.experiment_id.iloc[0],mechanism=f.mechanism.iloc[0],model=f.model.iloc[0],method_id=method,truth=f.truth.iloc[0],n_tasks=n,n_complete=int(done.sum()),n_fit_success=int(f.fit_success.sum()),numerical_failures=int((~done).sum()),n_coverage_defined=nc,n_covered=covered,coverage=covered/nc if nc else np.nan,coverage_low=lo,coverage_high=hi,end_to_end_covered_fraction=covered/n if nc else np.nan,coverage_mcse=np.sqrt(covered/nc*(1-covered/nc)/nc) if nc else np.nan,called=called,call_rate=called/n,correct_calls=correct,correct_call_fraction=correct/n if f.truth_available.all() else np.nan,correct_low=cl if f.truth_available.all() else np.nan,correct_high=ch if f.truth_available.all() else np.nan,wrong_calls=wrong,wrong_sign_among_called=wrong/nv if nv else np.nan,risk_low=rl,risk_high=rh,bias=errors.mean(),bias_mcse=errors.std(ddof=1)/np.sqrt(len(errors)) if len(errors)>1 else np.nan,rmse=np.sqrt(np.mean(errors**2)) if len(errors) else np.nan,n_estimate_errors=len(errors),n_finite_width=len(width),median_finite_width=width.median(),infinite_interval_fraction=float(f.infinite_interval.mean()),prediction_KL_mean=f.population_prediction_KL.mean(),deviance_mean=f.deviance.mean(),nuisance_rmse_mean=f.nuisance_rmse.mean(),n_calibration=f.n_calibration.mean(),n_observed=f.n_observed.mean(),budget_value=f.budget_value.mean(),source_attribution_allowed=False))
 summary=pd.DataFrame(summaries);summary.to_csv(G/'results/G3_summary.tsv',sep='\t',index=False)
 for group,name in [('G3A','G3_crossed_mechanisms'),('G3B','G3_calibration_value'),('G3C','G3_common_bias')]:summary[summary.experiment_id==group].to_csv(G/'results'/f'{name}.tsv',sep='\t',index=False)
 pd.DataFrame(failures,columns=df.columns).to_csv(G/'results/FAILURE_LEDGER.tsv',sep='\t',index=False)
 expected=sum(c['replicates']*len(arms(c)) for c in cells());status=dict(run_status='COMPLETE' if len(df)==expected else 'PARTIAL',independent_replicates=len(primary_paths),total_executed_independent_replicates=len(list((G/'runs/core').glob('*.json')))+len(list((G/'runs/alignment_revision').glob('*.json'))),method_evaluations=len(df),expected_method_evaluations=expected,numerical_failures=len(failures),source_attribution_allowed=False)
 (G/'runs/SUMMARY_STATUS.json').write_text(json.dumps(status,indent=2));print(status)
if __name__=='__main__':main()
