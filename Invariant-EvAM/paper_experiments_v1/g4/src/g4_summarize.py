from pathlib import Path
import json,sys,hashlib
R=Path(__file__).resolve().parents[3];G=R/'paper_experiments_v1/g4';sys.path.insert(0,str(R/'paper_experiments_v1/g2/vendor'))
import numpy as np,pandas as pd
MET=['hidden_brier','hidden_log_score','hidden_joint_log_score','source_log_score','source_tv']
def save(df,name):df.to_csv(G/f'results/{name}.tsv',sep='\t',index=False)
def main():
 rows=[];refs=[];flows=[];failures=[];bins=[];capacities=[];fullrefs={c:json.loads((G/f'runs/reference/{c}.json').read_text()) for c in ['LUAD','COAD']}
 for folder in ['perturbation','bootstrap']:
  for f in sorted((G/'runs'/folder).glob('*.json')):
   j=json.loads(f.read_text());base={k:j[k] for k in ['cancer','design','replicate','purpose','fixed_n']};base.update(group_id='G4',protocol_version=j.get('protocol_version','G4_v1.0'),evidence_type='DIRECT_RECORD_PROXY_REFERENCE',truth_available=False,interval_coverage_defined=False,split_id='HASH60_20_20_G4_V1',source_status='PREVIOUSLY_EXPLORED')
   if j['status']=='DESIGN_CAPACITY_FAILURE':capacities.append({**base,**j['stats']});failures.append({**base,'method':'ALL','failure':'DESIGN_CAPACITY_FAILURE','reason':'Insufficient distinct enrolled patients; no seed replacement'});continue
   for role,s in j['stats'].items():
    if isinstance(s,dict) and 'n_screened' in s:flows.append({**base,'role':role,**{k:v for k,v in s.items() if k!='by_environment'},'actual_gene_assays':j['stats']['actual_gene_assays'],'budget_cap':j['stats']['budget_cap']})
   for method,r in j['methods'].items():
    row={**base,'method':method,'success':r['success'],'computational_status':r.get('status','ALIAS_M'),'n_screened':j['stats']['train']['n_screened'],'n_observed':j['stats']['train']['n_selected'],'n_unique_patients':j['stats']['train']['n_unique'],'n_unique_observed':j['stats']['train']['selected_unique'],'n_unique_calibration':j['stats']['calibration']['n_unique'],'n_calibration':80 if method!='M_NO_CAL' else 0,'budget_basis':'FIXED_CANDIDATE_AND_MAXIMUM_MEASUREMENT_BUDGET','budget_value':j['stats']['budget_cap'],'actual_gene_assays':j['stats']['actual_gene_assays'],'estimate':r.get('estimate'),'ci_lower':r.get('lo'),'ci_upper':r.get('hi'),'interval_type':r.get('interval_type'),'interval_status':r.get('interval_status','NOT_REQUESTED'),'rank':r.get('rank'),'null_leakage':r.get('null_leakage'),'target_effective_information':r.get('target_effective_information'),'sign_call':r.get('sign'),'boundary_hit':r.get('boundary_hit'),'runtime_seconds':r.get('runtime_seconds'),'peak_memory_mb':None,'reason':'Peak memory not instrumented; no natural theta truth'}
    scores=r.get('scores') or {};row.update({k:scores.get(k) for k in MET+['score_status','n_source_evaluation','n_unique_source_evaluation','n_hidden_evaluation']});rows.append(row)
    for z in scores.get('calibration_bins',[]):bins.append({**base,'method':method,**z})
    if not r['success']:failures.append({**base,'method':method,'failure':'FIT_FAILED','reason':r.get('message','Baseline subfit or projection failed')})
    if r.get('interval_status')=='INTERVAL_FAILED':failures.append({**base,'method':method,'failure':'INTERVAL_FAILED','reason':str(r.get('interval_attempts'))})
    if r.get('interval_status')=='COMPLETE':
     fr=fullrefs[j['cancer']];reference_ready=fr.get('interval_status')=='COMPLETE';overlap=max(r['lo'],fr['lo'])<=min(r['hi'],fr['hi']) if reference_ready else None;refs.append({**base,'method':method,'estimate':r['estimate'],'reference_estimate':fr['estimate'],'estimate_difference':r['estimate']-fr['estimate'],'ci_lower':r['lo'],'ci_upper':r['hi'],'reference_lower':fr.get('lo'),'reference_upper':fr.get('hi'),'profile_overlap':overlap,'sign':r['sign'],'reference_sign':fr.get('sign'),'sign_agreement':r['sign']==fr.get('sign') if reference_ready else None,'both_definite_signs':r['sign'] in ['POSITIVE','NEGATIVE'] and fr.get('sign') in ['POSITIVE','NEGATIVE'],'interval_type':r['interval_type'],'analysis':'reference_analysis_agreement_NOT_biology_recovery'})
 d=pd.DataFrame(rows);save(d,'G4_all_method_results');primary=d[(d.purpose=='perturbation')&~d.fixed_n];fixed=d[(d.purpose=='perturbation')&d.fixed_n];boot=d[d.purpose=='bootstrap'];save(primary.drop(columns=['source_log_score','source_tv']),'G4_hidden_record_scores');save(primary.drop(columns=['hidden_brier','hidden_log_score','hidden_joint_log_score']),'G4_source_distribution_scores');save(fixed,'G4_fixed_enrolled_n_secondary');save(pd.DataFrame(refs),'G4_reference_agreement');save(pd.DataFrame(flows),'G4_perturbation_patient_flow');save(pd.DataFrame(bins),'G4_prediction_calibration_bins');save(pd.DataFrame(capacities),'G4_design_capacity_failures');save(pd.DataFrame(failures,columns=list(dict.fromkeys([k for z in failures for k in z])) or ['cancer','design','replicate','method','failure','reason']),'FAILURE_LEDGER');boot.to_parquet(G/'results/G4_patient_bootstrap_results.parquet',index=False)
 summary=[]
 for key,q in d.groupby(['purpose','fixed_n','cancer','design','method']):
  row=dict(zip(['purpose','fixed_n','cancer','design','method'],key));row.update(n_planned=50 if key[0]=='perturbation' else 200,n_runs=len(q),n_fit_success=int(q.success.sum()),n_interval_complete=int(q.interval_status.eq('COMPLETE').sum()),n_interval_failed=int(q.interval_status.eq('INTERVAL_FAILED').sum()),n_unknown=int(q.sign_call.eq('UNKNOWN').sum()),mean_selected=q.n_observed.mean())
  for metric in MET:
   vv=q.loc[q.success,metric].dropna();row.update({metric+'_mean':vv.mean(),metric+'_sd':vv.std(),metric+'_q025':vv.quantile(.025),metric+'_q975':vv.quantile(.975),metric+'_n':len(vv)})
  summary.append(row)
 save(pd.DataFrame(summary),'G4_summary');pair=[];pairsummary=[]
 # Positive gain always means the comparator has a larger (worse) loss.
 for cancer in ['LUAD','COAD']:
  for purpose,dd in [('perturbation',primary),('bootstrap',boot)]:
   for metric in MET:
    tasks=[('HETEROGENEOUS','M','HETEROGENEOUS',m) for m in ['B1','B2','B3','B4','K0','M_NO_CAL'] if m in dd.method.unique()]
    if metric.startswith('source_'):tasks += [('HETEROGENEOUS','M','REDUNDANT','M'),('HETEROGENEOUS','M','CROSSVIEW_FAILURE','M')]
    for des,m,refdes,refm in tasks:
     a=dd[(dd.cancer==cancer)&(dd.design==des)&(dd.method==m)].set_index('replicate');b=dd[(dd.cancer==cancer)&(dd.design==refdes)&(dd.method==refm)].set_index('replicate');ids=a.index.intersection(b.index);ids=[i for i in ids if a.loc[i,'success'] and b.loc[i,'success'] and pd.notna(a.loc[i,metric]) and pd.notna(b.loc[i,metric])];vals=[]
     for i in ids:
      gain=b.loc[i,metric]-a.loc[i,metric];vals.append(gain);pair.append(dict(cancer=cancer,purpose=purpose,metric=metric,design=des,method=m,reference_design=refdes,reference_method=refm,replicate=i,gain=gain))
     if len(vals):pairsummary.append(dict(cancer=cancer,purpose=purpose,metric=metric,design=des,method=m,reference_design=refdes,reference_method=refm,n_pairs=len(vals),n_planned=50 if purpose=='perturbation' else 200,gain_mean=np.mean(vals),gain_median=np.median(vals),low=np.quantile(vals,.025),high=np.quantile(vals,.975),fraction_gain_positive=np.mean(np.array(vals)>0),interval_semantics='Patient bootstrap plus one randomized perturbation percentile range; not certified nominal confidence' if purpose=='bootstrap' else 'Mechanism stability range; not independent-cohort confidence'))
 save(pd.DataFrame(pair),'G4_paired_losses');save(pd.DataFrame(pairsummary),'G4_paired_summary')
 aux=[];sources=[]
 for f in sorted((G/'runs/auxiliary').glob('*.json')):
  j=json.loads(f.read_text())
  for bound,r in j['source_fits'].items():sources.append(dict(cancer=j['cancer'],draw=j['draw'],bound=int(bound),success=r['success'],target=r['estimate'],nll=r['nll'],target_boundary=abs(r['estimate'])>=float(bound)-.001,any_theta_boundary=any(abs(v)>=float(bound)-.001 for v in r['x'][:10]),expanded_source_distribution_tv=j['expanded_source_distribution_tv']))
  for z in j['results']:
   r=z['result'];aux.append(dict(cancer=j['cancer'],source_draw=j['draw'],bound=z['bound'],replicate=z['replicate'],method=z['method'],evidence_type='MODEL_ANCHORED_SOURCE_BOOTSTRAP',true_target=r['true_target'],estimate=r['estimate'],success=r['success'],interval_status=r.get('interval_status','NOT_REQUESTED'),interval_type=r.get('interval_type'),ci_lower=r.get('lo'),ci_upper=r.get('hi'),covers_truth=r['covers_truth'],sign=r.get('sign'),source_tv=r['source_tv'],source_truth_outside_fit_box=abs(r['true_target'])>3,runtime_seconds=r['runtime_seconds']))
 save(pd.DataFrame(aux),'G4_model_anchored_auxiliary');save(pd.DataFrame(sources),'G4_auxiliary_source_fits')
 status=dict(primary_runs=len(list((G/'runs/perturbation').glob('*[0-9].json'))),primary_expected=300,fixed_n_runs=len(list((G/'runs/perturbation').glob('*_fixedN.json'))),fixed_n_expected=300,bootstrap_runs=len(list((G/'runs/bootstrap').glob('*.json'))),bootstrap_expected=800,auxiliary_source_draws=len(list((G/'runs/auxiliary').glob('*.json'))),auxiliary_expected=20,auxiliary_method_evaluations=len(aux),failures=len(failures),capacity_failures=len(capacities));status['run_status']='COMPLETE' if all(status[a]==status[b] for a,b in [('primary_runs','primary_expected'),('fixed_n_runs','fixed_n_expected'),('bootstrap_runs','bootstrap_expected'),('auxiliary_source_draws','auxiliary_expected')]) else 'PARTIAL';(G/'results/RUN_STATUS.json').write_text(json.dumps(status,indent=2));print(status)
if __name__=='__main__':main()
