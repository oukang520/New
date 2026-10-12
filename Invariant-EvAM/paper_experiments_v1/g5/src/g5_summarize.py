"""Derived tables only. No fitting, selection, filtering, or mutation of frozen runs."""
from pathlib import Path
import json,math
import numpy as np,pandas as pd
from scipy.stats import norm
G=Path(__file__).resolve().parents[1];P=json.loads((G/'configs/PROTOCOL.json').read_text());OUT=G/'results'
def files(folder):
 for p in sorted((G/'runs'/folder).glob('*.json')):yield p,json.loads(p.read_text())
def save(name,rows):
 df=pd.DataFrame(rows);df.to_csv(OUT/(name+'.tsv'),sep='\t',index=False);return df
def interval(v):
 v=np.asarray(v,float);v=v[np.isfinite(v)]
 return (float(v.mean()),float(v.mean()-1.96*v.std(ddof=1)/np.sqrt(len(v))),float(v.mean()+1.96*v.std(ddof=1)/np.sqrt(len(v)))) if len(v)>1 else (float(v[0]) if len(v) else np.nan,np.nan,np.nan)
def wilson(k,n):
 if not n:return np.nan,np.nan
 z=1.95996398454;p=k/n;c=(p+z*z/(2*n))/(1+z*z/n);w=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return c-w,c+w
FAIL=[]
def definition(target_id):
 genes=['TP53','KRAS','STK11','KEAP1'];s=target_id.split('__')[1]
 if 'contrast' in s:
  pair='KRAS←TP53 minus STK11←KRAS' if target_id.startswith('D4_SPARSE') else 'TP53←KRAS minus TP53←STK11'
  return 'Conditional CTMC log-hazard interaction contrast: '+pair+'; fixed time 1, reference468; not causal/ancestry'
 u,v=s.replace('interaction_','').split('_to_');return 'Conditional CTMC log-hazard interaction '+genes[int(v)]+'←'+genes[int(u)]+'; fixed time 1, reference468; not causal/ancestry'
def addfail(area,key,r):
 if not r.get('success',True):FAIL.append(dict(area=area,key=key,stage='FIT',status='NUMERICAL_FAILURE',message=r.get('fit',r).get('message','')))
 for t,v in r.get('targets',{}).items():
  if v.get('status') in ['NUMERICAL_FAILURE','FIT_FAILED','PROFILE_FAILED']:FAIL.append(dict(area=area,key=key,stage=t,status='NUMERICAL_FAILURE',message=json.dumps(v.get('attempts'))))

def natural():
 boot=[];compat=[];main=[];targets=[];sens=[];pred=[];predboot=[];resid=[]
 for p,o in files('bootstrap'):
  r=o['result'];addfail('natural_bootstrap',p.name,r)
  for t,z in r['targets'].items():boot.append(dict(spec_id=o['spec_id'],replicate=o['replicate'],target_id=t,fit_success=r['success'],training_deviance=r['deviance'],**{k:v for k,v in z.items() if k not in ['attempts','tolerance']}))
 b=save('G5_patient_bootstrap',boot)
 for p,o in files('compatibility'):
  r=o['result'];addfail('compatibility',p.name,r);compat.append(dict(spec_id=o['spec_id'],replicate=o['replicate'],success=r['success'],deviance=r['deviance']))
 co=save('G5_compatibility_bootstrap',compat)
 for p,o in files('sensitivity'):
  addfail('sensitivity',p.name,o['result'])
  for t,z in o['result']['targets'].items():sens.append(dict(epsilon=o['epsilon'],eta=o['eta'],target_id=t,**{k:v for k,v in z.items() if k not in ['attempts','tolerance']}))
 se=save('G5_sensitivity_all_targets',sens)
 for p,o in files('heldout'):
  addfail('heldout_train',p.name,o['train']);addfail('heldout_calibration',p.name,o['calibration_fit'])
  pred.append({k:v for k,v in o.items() if k not in ['train','calibration_fit','prediction','evaluation_counts','evaluation_bootstrap_scores']})
  for i,v in enumerate(o['evaluation_bootstrap_scores']):predboot.append(dict(heldout=o['heldout'],spec_id=o['spec_id'],replicate=i,log_score=v))
  freq=np.array(o['evaluation_counts'])/o['n_evaluation']
  for state,(q,f) in enumerate(zip(o['prediction'],freq)):resid.append(dict(analysis='HELDOUT_EVALUATION',spec_id=o['spec_id'],environment=o['heldout'],state=state,observed=f,predicted=q,residual=f-q))
 pr=save('G5_heldout_predictions',pred);pb=save('G5_heldout_score_bootstrap',predboot)
 for p,o in files('natural_main'):
  r=o['result'];sid=o['spec_id'];addfail('natural_main',p.name,r);geo=r.get('geometry') or {};cc=co[co.spec_id.eq(sid)] if len(co) else pd.DataFrame();good=cc[cc.success.eq(True)] if len(cc) else cc;pv=(1+sum(good.deviance>=r['deviance']))/(1+len(good)) if len(good) and o['topology']=='SPARSE' else np.nan
  sat=sum(-np.sum(np.array(c)[np.array(c)>0]*np.log(np.array(c)[np.array(c)>0]/sum(c))) for c in o['counts']);projected_nll=sum(-np.array(c)@np.log(np.maximum(q,1e-280)) for c,q in zip(o['counts'],r['probabilities']))
  main.append(dict(spec_id=sid,topology=o['topology'],n=sum(map(sum,o['counts'])),success=r['success'],training_nll_raw_likelihood=r['nll'],training_deviance_raw=r['deviance'],post_projection_nll=projected_nll,post_projection_deviance=2*(projected_nll-sat),n_compatibility_attempted=len(cc),n_compatibility_success=len(good),parametric_bootstrap_p=pv,compatibility_scope='plugin parametric refit; conditional family, not biology truth',**{k:v for k,v in geo.items() if k!='tolerance'}))
  for panel,c,q in zip(P['panels'],o['counts'],r['probabilities']):
   for state,(obs,pre) in enumerate(zip(np.array(c)/sum(c),q)):resid.append(dict(analysis='TRAINING_POST_PROJECTION',spec_id=sid,topology=o['topology'],environment=panel,state=state,observed=obs,predicted=pre,residual=obs-pre))
  registry=pd.read_csv(G/'configs/target_registry.tsv',sep='\t');registry=registry[registry.model_id.eq('D4_SPARSE10' if o['topology']=='SPARSE' else 'D4_FULL16')]
  for row in registry.itertuples():
   z=r['targets'].get(row.target_id,{});bs=b[b.spec_id.eq(sid)&b.target_id.eq(row.target_id)] if len(b) and o['topology']=='SPARSE' else pd.DataFrame();ss=se[se.target_id.eq(row.target_id)] if len(se) else pd.DataFrame();hp=pr[pr.spec_id.eq(sid)] if len(pr) else pd.DataFrame();raw=r['fit'].get('projection',r['fit']);estimate=z.get('estimate',float(np.array(json.loads(row.vector))@np.array(raw['x'])[:len(json.loads(row.vector))]));status=z.get('status','POINT_ONLY_NO_INTERVAL');lo=z.get('lo') if z.get('lo') is not None else np.nan;hi=z.get('hi') if z.get('hi') is not None else np.nan;leak=z.get('null_leakage',np.nan)
   if status in ['NUMERICAL_FAILURE','FIT_FAILED','PROFILE_FAILED']:level='COMPUTATION_UNRESOLVED'
   elif status=='POINT_ONLY_NO_INTERVAL':level='NOT_EVALUATED_POINT_BASELINE'
   elif leak>=1e-7:level='STRUCTURALLY_UNRESOLVED'
   elif np.isfinite(pv) and pv<.05:level='MODEL_INCOMPATIBLE'
   elif not np.isfinite([lo,hi]).all():level='PARTIALLY_BOUNDED'
   elif z.get('sign')=='UNKNOWN':level='WEAK_OR_WIDE'
   elif len(ss) and (ss.sign.ne(z.get('sign')).any() or ss.status.ne('COMPLETE').any()):level='SENSITIVITY_DEPENDENT'
   else:level='CONDITIONALLY_SUPPORTED'
   targets.append(dict(spec_id=sid,target_id=row.target_id,parameter_and_event_definition=row.interpretation,reference_population='MSK468 primary LUAD G4 development 1345; reference anchor phi=0',model_topology=o['topology'],topology_assumed=True,operator_specification=sid,estimate=estimate,profile_lo=lo,profile_hi=hi,profile_status=status,sign=z.get('sign','NOT_EVALUATED'),interval_type=z.get('interval_type','NONE'),structural_status='TARGET_CONFOUNDED' if leak>=1e-7 else 'LOCALLY_SEPARABLE' if np.isfinite(leak) else 'NOT_EVALUATED',null_leakage=leak,tolerance_stable=z.get('tolerance_stable'),tolerance_details=json.dumps(z.get('tolerance')),sigma_min_per_sample=z.get('sigma_min_per_sample'),target_information=z.get('target_information'),boundary_status=z.get('boundary_hit'),compatibility_bootstrap_p=pv,heldout_compatibility=json.dumps(dict(zip(hp.heldout,hp.conditional_multinomial_p))) if len(hp) else '{}',bootstrap_attempts=len(bs),bootstrap_positive=int(bs.sign.eq('POSITIVE').sum()) if len(bs) else 0,bootstrap_negative=int(bs.sign.eq('NEGATIVE').sum()) if len(bs) else 0,bootstrap_unknown=int((bs.sign.eq('UNKNOWN')&bs.status.eq('COMPLETE')).sum()) if len(bs) else 0,bootstrap_numerical_failure=int(bs.status.ne('COMPLETE').sum()) if len(bs) else 0,biology_sensitivity='epsilon 0,0.2,0.5; all targets in sensitivity table' if len(ss) else 'NOT_RUN_FULL_TOPOLOGY',operator_sensitivity='reference-gene eta 0,0.5,1; no unit addition' if len(ss) else 'NOT_RUN_FULL_TOPOLOGY',sensitivity_signs='|'.join(sorted(set(ss.sign))) if len(ss) else '',shared_bias_limitation='No exclusion of arbitrary common selection; reference anchor conditional',interpretation_level=level,biological_claim_allowed=False))
 for z in targets:z['parameter_and_event_definition']=definition(z['target_id'])
 m=save('G5_natural_model_comparison',main);t=save('G5_all_target_interpretations',targets);save('G5_distribution_residuals',resid)
 part=[]
 for p,o in files('partitions'):
  r=o['result'];addfail('partitions',p.name,r);part.append(dict(replicate=o['replicate'],n_unique=1345,success=r['success'],rank_one=r['same_operator_rank_one'],rank_split=r['same_operator_rank_split'],shared_deviance=r['deviance'],independent_deviance=r['independent']['deviance']))
 save('G5_within_panel_random_partitions',part)
 paired=[]
 for e,z in pb.groupby('heldout'):
  base=z[z.spec_id.eq('HIERARCHICAL')].set_index('replicate').log_score
  for sid,w in z.groupby('spec_id'):
   dif=w.set_index('replicate').log_score-base;paired.append(dict(heldout=e,spec_id=sid,reference='HIERARCHICAL',mean_difference=dif.mean(),conditional_patient_bootstrap_lo=dif.quantile(.025),conditional_patient_bootstrap_hi=dif.quantile(.975),n_resamples=len(dif)))
 save('G5_heldout_paired_differences',paired)
 return m,t

def design(kind):
 rows=[];dec=[];capacity=[]
 for p,o in files('design_'+kind):
  if o.get('capacity_failure'):capacity.append(dict(kind=kind,file=p.name,reason=o['reason']));FAIL.append(dict(area='design_'+kind,key=p.name,stage='CAPACITY',status='INFEASIBLE',message=o['reason']));continue
  base=o['baseline_scores'];bp=o['baseline_profile'];addfail('design_'+kind,p.name,o['baseline'])
  if bp['status']!='COMPLETE':FAIL.append(dict(area='design_'+kind,key=p.name,stage='BASELINE_PROFILE',status='NUMERICAL_FAILURE',message=json.dumps(bp.get('attempts'))))
  for d in o['decisions']:
   v=d['decision'];dec.append(dict(kind=kind,scenario=o['scenario'],replicate=o['replicate'],budget=d['budget'],robust_equals_c=v['robust_equals_c'],representative_count=v['representative_count'],ridge_choices=json.dumps(v['ridge_choices']),ridge_choice_stable=len(set(v['ridge_choices'].values()))==1))
  for z in o['results']:
   r=z['result'];a=z['action'];pr=r['profile'];complete=pr['status']=='COMPLETE';sc=r['scores'];truth=pr.get('covers_truth');called=complete and pr['sign'] in ['POSITIVE','NEGATIVE'];wrong=pr.get('wrong_sign') if complete else None
   if not r['fit']['success'] or not complete:FAIL.append(dict(area='design_'+kind,key=p.name,stage=f"{z['strategy']}:{z['budget']}",status='NUMERICAL_FAILURE',message=json.dumps(pr.get('attempts'))))
   row=dict(kind=kind,scenario=o['scenario'],replicate=o['replicate'],strategy=z['strategy'],budget=z['budget'],budget_fraction=z['budget']/300,evidence_type=o['evidence_type'],natural_theta_truth_available=o['natural_theta_truth_available'],fit_success=r['fit']['success'],profile_status=pr['status'],estimate=pr['estimate'],lo=pr['lo'],hi=pr['hi'],interval_type=pr['interval_type'],finite_interval=complete and np.isfinite([pr['lo'],pr['hi']]).all(),width=pr['hi']-pr['lo'] if complete else np.nan,baseline_width=bp['hi']-bp['lo'] if bp['status']=='COMPLETE' else np.nan,baseline_finite=bp['status']=='COMPLETE' and np.isfinite([bp['lo'],bp['hi']]).all(),sign=pr['sign'],called=called,covers_truth=truth,wrong_sign=wrong,correct_call=called and wrong==False if truth is not None else None,predicted_rank=a['rank'],predicted_target_variance=a['variance'],predicted_null_leakage=a['null_leakage'],**{k:a[k] for k in ['candidate_id','add_pairs','reveal_pairs','calibration_pairs','unused']},**{k:v for k,v in r['counts'].items() if k!='calibration_ids'},**sc)
   for name in ['source_log_score','source_tv','hidden_brier','hidden_log_score']:
    row['baseline_'+name]=base.get(name);row[name+'_gain']=base[name]-sc[name] if base.get(name) is not None and sc.get(name) is not None else np.nan
   rows.append(row)
 df=save('G5_design_'+('outer_evaluation' if kind=='outer' else 'direct_evaluation'),rows);save('G5_design_'+kind+'_decisions',dec);save('G5_design_'+kind+'_capacity_failures',capacity)
 if not len(df):return df,pd.DataFrame()
 summary=[]
 for key,z in df.groupby(['scenario','budget','strategy']):
  good=z[z.profile_status.eq('COMPLETE')];finite=z[z.finite_interval];s=dict(zip(['scenario','budget','strategy'],key));s.update(n_attempted=len(z),n_complete=len(good),n_numerical_failure=len(z)-len(good),n_finite=len(finite),finite_fraction=len(finite)/len(z),n_called=int(z.called.sum()),call_fraction=float(z.called.mean()),finite_width_mean=finite.width.mean(),mean_actual_cost=z.cost.mean(),mean_new_patients=z.new_unique_patients.mean(),mean_remeasured=z.remeasured_existing.mean())
  for metric in ['source_log_score_gain','source_tv_gain','hidden_brier_gain','hidden_log_score_gain']:
   mean,lo,hi=interval(z[metric]);s[metric]=mean;s[metric+'_mc_lo']=lo;s[metric+'_mc_hi']=hi
  if kind=='outer':
   n=int(good.covers_truth.notna().sum());k=int(good.covers_truth.sum());lo,hi=wilson(k,n);wrong=int(good.wrong_sign.sum());called=int(good.called.sum());s.update(n_truth_intervals=n,coverage=k/n if n else np.nan,coverage_wilson_lo=lo,coverage_wilson_hi=hi,end_to_end_coverage=k/len(z),n_correct=int(z.correct_call.sum()),correct_fraction=float(z.correct_call.sum()/len(z)),n_wrong=wrong,wrong_fraction=wrong/len(z),conditional_wrong=wrong/called if called else np.nan)
  summary.append(s)
 su=save('G5_design_'+kind+'_summary',summary)
 pair=[]
 for (scenario,budget),group in df.groupby(['scenario','budget']):
  for comparator in ['COUNT_ONLY','D_OPT','C_OPT','RANDOM_FEASIBLE','MAX_PANEL']:
   a=group[group.strategy.eq('INVARIANT_ROBUST')].set_index('replicate');b=group[group.strategy.eq(comparator)].set_index('replicate')
   for metric in ['source_log_score_gain','hidden_brier_gain','finite_interval']+(['correct_call','wrong_sign','covers_truth'] if kind=='outer' else []):
    aa=pd.to_numeric(a[metric],errors='coerce');bb=pd.to_numeric(b[metric],errors='coerce');dif=aa.astype(float)-bb.astype(float);mean,lo,hi=interval(dif);pair.append(dict(scenario=scenario,budget=budget,method='INVARIANT_ROBUST',comparator=comparator,metric=metric,n_paired=int(dif.notna().sum()),difference=mean,mc_lo=lo,mc_hi=hi))
 save('G5_design_'+kind+'_paired_comparisons',pair);return df,su

def main():
 original=OUT/'audit_original/G5_natural_denominator.tsv'
 if not original.exists():
  original.parent.mkdir(exist_ok=True);original.write_bytes((OUT/'G5_natural_denominator.tsv').read_bytes())
 denom=pd.read_csv(original,sep='\t');denom['heldout_split_used']=denom.environment.ne('MSK-IMPACT468')
 denom.loc[~denom.heldout_split_used,['heldout_calibration_n','heldout_evaluation_n']]=np.nan
 for col,val in [('g4_train_n',1345),('g4_calibration_n',448),('g4_evaluation_n',449)]:denom[col]=np.where(denom.environment.eq('MSK-IMPACT468'),val,np.nan)
 denom.to_csv(OUT/'G5_natural_denominator.tsv',sep='\t',index=False)
 m,t=natural();outer,os=design('outer');direct,ds=design('direct')
 boundary=[]
 for path,o in files('boundary_diagnostic'):
  r=o['result'];addfail('boundary_diagnostic',path.name,r)
  for target,z in r['targets'].items():boundary.append(dict(bound=o['bound'],spec_id=o['spec_id'],topology=o['topology'],target_id=target,training_deviance=r['deviance'],**{k:v for k,v in z.items() if k not in ['attempts','tolerance']}))
 save('G5_boundary_diagnostic',boundary)
 save('FAILURE_LEDGER',FAIL or [dict(area='ALL',key='ALL',stage='ALL',status='NONE',message='No failures')])
 reg=[dict(action='ADD_SAME_PATIENT_TYPE',unit='one new selected patient, 3 measured genes',unit_cost=3,adds_independent_patient=True,source='Controlled simulated future or G4 calibration record pool',requirements='Recruit and obtain current measurement; screening acquisition cost excluded'),dict(action='REMEASURE_HIDDEN_EVENT',unit='one existing selected patient, 1 hidden gene',unit_cost=1,adds_independent_patient=False,source='Existing baseline patients',requirements='Stored material and hidden-event assay; replaces partial record likelihood'),dict(action='CALIBRATION_XS',unit='one new source patient, four genes and selection indicator',unit_cost=5,adds_independent_patient=True,source='Controlled experiment only; S is generated, not available as natural metadata',requirements='Prospective source sampling and joint X,S collection; no invented natural S')]
 save('G5_design_candidate_registry',reg)
 expected={'natural_main':9,'sensitivity':9,'heldout':14,'bootstrap':800,'compatibility':800,'partitions':100,'design_outer':600,'design_direct':50};counts={k:len(list((G/'runs'/k).glob('*.json'))) for k in expected};(OUT/'RUN_STATUS.json').write_text(json.dumps(dict(expected=expected,actual=counts,all_attempted=counts==expected,outer_method_rows=len(outer),direct_method_rows=len(direct),all_target_rows=len(t),failure_rows=len(FAIL)),indent=2));print(json.dumps(counts));print(m[['spec_id','topology','training_deviance_raw','parametric_bootstrap_p']].to_string(index=False));print('Targets',t.interpretation_level.value_counts().to_dict())
if __name__=='__main__':main()
