from pathlib import Path
import json
import numpy as np
import pandas as pd
G=Path(__file__).resolve().parents[1]
F=json.loads((G/'configs/EVALUATION_FREEZE.json').read_text());CAL=F['primary_calibration']
METRICS=['log_score','brier','ece','source_log_score','integrated_ece','conditional_probability_mse','integrated_log_score']
def read(stage):return pd.read_csv(G/f'results/{stage}_raw.tsv',sep='\t')
def compare(d,a,b,metric):
    x=d.pivot(index=['cancer','design','replicate'],columns='method',values=metric)
    return (x[a]-x[b]).rename('gain').reset_index() if a in x and b in x else pd.DataFrame()
def main():
    d=read('evaluation');boot=read('bootstrap') if (G/'results/bootstrap_raw.tsv').exists() else None
    pairs=[('B_'+m,'MORE_'+m,'BRIDGE_'+m) for m in ['CTMC','POOLED','HIERARCHICAL','FUSION','TWO_STEP']]+[('C_base','BASE_CTMC',CAL),('C_samecost','MORE_CTMC',CAL),('C_joint_refit','BASE_CTMC','BRIDGE_CTMC'),('C_vs_joint','BRIDGE_CTMC',CAL),('C_vs_direct','AUDIT_DIRECT_1',CAL),('C_vs_marginal','AUDIT_MARGINAL_1',CAL),('B_label_control','SHUFFLED_CTMC','BRIDGE_CTMC')]
    out=[];raw=[]
    for name,a,b in pairs:
        for metric in METRICS:
            t=compare(d,a,b,metric)
            if t.empty:continue
            t['comparison']=name;t['metric']=metric;raw.append(t)
            q=compare(boot,a,b,metric) if boot is not None else pd.DataFrame()
            for cancer in ['LUAD','COAD']:
                for design in ['REDUNDANT','HETEROGENEOUS','CROSSVIEW_FAILURE','ALL_EQUAL_WEIGHT']:
                    v=t[t.cancer.eq(cancer)]
                    if design!='ALL_EQUAL_WEIGHT':v=v[v.design.eq(design)];value=v.gain.mean()
                    else:value=v.groupby('design').gain.mean().mean()
                    rr=dict(comparison=name,reference=a,candidate=b,cancer=cancer,design=design,metric=metric,gain=value,n_successful_seed_cells=len(v),seed_sd=v.gain.std(),bootstrap_n=0,bootstrap_lo=np.nan,bootstrap_hi=np.nan,interval_type='PATIENT_PLUS_SINGLE_PERTURBATION_PERCENTILE_RANGE')
                    if not q.empty:
                        qb=q[q.cancer.eq(cancer)]
                        if design!='ALL_EQUAL_WEIGHT':z=qb[qb.design.eq(design)].gain.dropna()
                        else:
                            mat=qb.pivot(index='replicate',columns='design',values='gain');z=mat.dropna().mean(axis=1)
                        if len(z):rr.update(bootstrap_n=len(z),bootstrap_lo=z.quantile(.025),bootstrap_hi=z.quantile(.975))
                    out.append(rr)
    pd.concat(raw).to_csv(G/'results/paired_seed_gains.tsv',sep='\t',index=False);pd.DataFrame(out).to_csv(G/'results/paired_comparisons.tsv',sep='\t',index=False)
    rows=[];fail=[]
    for stage in ['development','evaluation','bootstrap']:
        for f in sorted((G/'runs').glob(stage+'_*.json')):
            z=json.loads(f.read_text());row=dict(stage=stage,cancer=z['cancer'],design=z['design'],replicate=z['replicate'],status=z['status']);rows.append(row)
            for m,r in z.get('methods',{}).items():
                if not r.get('fit',{}).get('success',True):fail.append(dict(**row,method=m))
    pd.DataFrame(rows).to_csv(G/'results/capacity_ledger.tsv',sep='\t',index=False);pd.DataFrame(fail,columns=['stage','cancer','design','replicate','status','method']).to_csv(G/'results/fit_failure_ledger.tsv',sep='\t',index=False)
    mean=d.groupby(['cancer','design','method'])[METRICS].mean().reset_index();mean.to_csv(G/'results/all_methods_means.tsv',sep='\t',index=False)
    means=mean.groupby(['cancer','method'])[METRICS].mean().reset_index();means.to_csv(G/'results/cancer_equal_design_means.tsv',sep='\t',index=False)
    print(means[means.method.isin(['BASE_CTMC','MORE_CTMC','BRIDGE_CTMC',CAL,'AUDIT_DIRECT_1','AUDIT_MARGINAL_1'])].to_string(index=False))
if __name__=='__main__':main()
