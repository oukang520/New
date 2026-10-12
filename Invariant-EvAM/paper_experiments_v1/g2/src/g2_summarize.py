from pathlib import Path
import sys,json,math
R=Path(__file__).resolve().parents[3];G=R/'paper_experiments_v1/g2';sys.path.insert(0,str(G/'vendor'))
import numpy as np,pandas as pd
from scipy.stats import norm

def wilson(k,n):
 if not n:return (np.nan,np.nan)
 z=norm.ppf(.975);p=k/n;den=1+z*z/n;centre=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return max(0,centre-half),min(1,centre+half)
def summary_frame(d,keys):
 rows=[]
 for key,f in d.groupby(keys,dropna=False):
  if not isinstance(key,tuple):key=(key,)
  n=len(f);success=f.computational_status.eq('COMPLETE');defined=f.coverage_defined.fillna(False);covered=f.covers_truth.fillna(False).astype(bool);called=f.called.astype(bool);wrong=f.wrong_sign.astype(bool);correct=f.correct_call.astype(bool)
  k=int(covered.sum());nc=int(called.sum());nw=int(wrong.sum());nf=int((~success).sum());valid=int(defined.sum());width=f.ci_upper-f.ci_lower;finite=width[np.isfinite(width)&defined]
  cl,cu=wilson(k,valid);wl,wu=wilson(nw,nc);ecl,ecu=wilson(k,n)
  rows.append({**dict(zip(keys,key)),'n_tasks':n,'n_interval_completed':valid,'n_covered':k,'conditional_coverage':k/valid if valid else np.nan,'coverage_wilson_low':cl,'coverage_wilson_high':cu,'coverage_mcse':np.sqrt((k/valid)*(1-k/valid)/valid) if valid else np.nan,'end_to_end_covered_fraction':k/n,'end_to_end_wilson_low':ecl,'end_to_end_wilson_high':ecu,'called':nc,'call_rate':nc/n,'correct_calls':int(correct.sum()),'correct_call_fraction':correct.mean(),'wrong_calls':nw,'wrong_sign_among_called':nw/nc if nc else np.nan,'wrong_sign_wilson_low':wl,'wrong_sign_wilson_high':wu,'solver_failures':nf,'workflow_failure_rate':nf/n,'infinite_interval_fraction_all_tasks':float((np.isinf(width)&defined).sum()/n),'finite_width_count':len(finite),'median_finite_width':float(finite.median()) if len(finite) else np.nan,'median_absolute_error_local_identified_only':float((f.estimate-f.truth).abs()[f.structural_status.eq('LOCAL_IDENTIFIED')&success].median()),'scientific_unknown_count':int(f.sign_call.eq('UNKNOWN').sum()),'validated_reliable_call_fraction':np.nan,'reliability_label':'EMPIRICAL_CALIBRATION_ONLY_NO_AUTOMATIC_VALIDATION'})
 return pd.DataFrame(rows)

def main():
 rows=[];risk=[];exceptions=[];complete=0;corrections=[]
 from g2_derive import normalize_batch
 public=json.loads((G/'configs/public_contract.json').read_text())
 for p in sorted((G/'runs/core').glob('*.json')):
  x=json.loads(p.read_text());a,b,c=normalize_batch(x,public);rows+=a;risk+=b;corrections+=c;complete+=x['expected_replicates']
  for y in x['details']:
   if 'exception' in y:exceptions.append(y)
 (G/'results/G2_derived_field_corrections.json').write_text(json.dumps(corrections,indent=2)+'\n')
 d=pd.DataFrame(rows);d.to_parquet(G/'results/G2_replicate_results.parquet',index=False)
 d.to_csv(G/'results/G2_replicate_results.tsv.gz',sep='\t',index=False,compression='gzip')
 primary=d[d.target_id=='primary'].copy();summary=summary_frame(primary,['cell_id','design','truth','method_id']);summary.to_csv(G/'results/G2_core_summary.tsv',sep='\t',index=False)
 summary.to_csv(R/'paper_experiments_v1/shared/calibration_summary.tsv',sep='\t',index=False)
 pairs=[]
 m=primary[primary.method_id=='M'].set_index(['cell_id','replicate_id'])
 for method in ['B1','B2','B3','B4','B5']:
  b=primary[primary.method_id==method].set_index(['cell_id','replicate_id']);both=m.join(b,lsuffix='_M',rsuffix='_B')
  for cell,g in both.groupby(level=0):
   for metric in ['covers_truth','correct_call','wrong_sign','called']:
    diff=g[metric+'_M'].fillna(False).astype(float)-g[metric+'_B'].fillna(False).astype(float);mean=float(diff.mean());se=float(diff.std(ddof=1)/np.sqrt(len(diff)))
    pairs.append({'cell_id':cell,'comparison':'M_vs_'+method,'metric':metric,'n_pairs':len(diff),'paired_difference':mean,'paired_mcse':se,'paired_normal95_low':mean-1.96*se,'paired_normal95_high':mean+1.96*se})
 pd.DataFrame(pairs).to_csv(G/'results/G2_paired_comparisons.tsv',sep='\t',index=False)
 risk=pd.DataFrame(risk);risk.to_parquet(G/'results/G2_risk_replicates.parquet',index=False);rr=[]
 for key,f in risk.groupby(['design','method_id','confidence']):
  n=len(f);called=int(f.called.sum());wrong=int(f.wrong.sum());lo,hi=wilson(wrong,called)
  rr.append(dict(zip(['design','method_id','confidence'],key))|{'n_tasks':n,'called':called,'wrong':wrong,'call_rate':called/n,'wrong_sign_among_called':wrong/called if called else np.nan,'wrong_sign_wilson_low':lo,'wrong_sign_wilson_high':hi,'failures':int((~f.success).sum())})
 pd.DataFrame(rr).to_csv(G/'results/G2_risk_answerability.tsv',sep='\t',index=False)
 d[d.computational_status!='COMPLETE'].to_csv(G/'results/G2_solver_failures.tsv',sep='\t',index=False)
 pd.DataFrame(exceptions,columns=['cell','replicate','exception']).to_csv(G/'results/G2_unhandled_failures.tsv',sep='\t',index=False)
 state={'expected_replicates':6000,'written_replicates':complete,'result_rows':len(d),'uncaught_replicate_failures':len(exceptions),'run_status':'COMPLETE' if complete==6000 and not exceptions else 'PARTIAL','science_status':'UNRESOLVED_PENDING_BOUND_AND_CALIBRATION_AUDIT'}
 (G/'runs/STATUS.json').write_text(json.dumps(state,indent=2)+'\n');print(state)
 if len(m):
  b5=primary[primary.method_id=='B5'].set_index(['cell_id','replicate_id']);delta=(m.estimate-b5.estimate).abs();print('M/B5 estimate max gap',delta.max());assert delta.max()<1e-12
if __name__=='__main__':main()
