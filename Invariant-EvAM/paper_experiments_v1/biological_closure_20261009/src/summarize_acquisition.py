"""Patient-paired inference for the frozen biological acquisition experiment."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
O=Path(__file__).resolve().parents[1]
def holm(p):
    p=np.asarray(p,float);i=np.argsort(p);r=np.empty(len(p));r[i]=np.minimum(1,np.maximum.accumulate(p[i]*(len(p)-np.arange(len(p)))));return r
def interval(v,seed):
    v=np.asarray(v,float);rng=np.random.default_rng(seed)
    b=v[rng.integers(0,len(v),(2000,len(v)))].mean(1)
    return float(v.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))
def paired_test(v,seed):
    v=np.asarray(v,float);rng=np.random.default_rng(seed);hit=0;B=49999
    for k in range(0,B,1000):
        s=rng.integers(0,2,(min(1000,B-k),len(v)),dtype=np.int8)*2-1
        hit+=int((np.abs((s*v).mean(1))>=abs(v.mean())-1e-14).sum())
    return (hit+1)/(B+1)
def main():
    R=O/'results/acquisition';d=pd.read_csv(R/'confirmation_patient_summary.tsv',sep='\t')
    protocol=json.loads((O/'configs/PROTOCOL.json').read_text());a=protocol['acquisition']
    assert d.split.eq('confirmation').all()
    s=pd.read_csv(O/'configs/PATIENT_SPLIT.tsv',sep='\t');assert set(d.patient_id)<=set(s.loc[s.split.eq('confirmation'),'patient_id'])
    metrics=['branch_recall','all_branch_recall','hidden_driver_recall','all_driver_recall','four_gene_log_loss','four_gene_marginal_brier','cost']
    rows=[];contrasts=[]
    for (pop,b,o),q in d.groupby(['population','budget','overhead'],sort=True):
        for method,m in q.groupby('method',sort=True):
            assert not m.patient_id.duplicated().any()
            for metric in metrics:
                u,lo,hi=interval(m[metric],[20261009,1,b,o,len(rows)])
                rows.append(dict(population=pop,budget=b,overhead=o,method=method,metric=metric,n_patients=len(m),mean=u,lo=lo,hi=hi))
        for metric in metrics:
            w=q.pivot(index='patient_id',columns='method',values=metric)
            assert not w.isna().any().any()
            for method in ['CTMC_BRIDGE','EMPIRICAL_BRIDGE','DIRECT_RIDGE','RANDOM']:
                v=w[method]-w.BEST_FIXED
                if metric in ['four_gene_log_loss','four_gene_marginal_brier','cost']:v=-v
                u,lo,hi=interval(v,[20261009,2,b,o,len(contrasts)])
                contrasts.append(dict(population=pop,budget=b,overhead=o,method=method,comparator='BEST_FIXED',metric=metric,n_patients=len(v),gain=u,lo=lo,hi=hi))
    pd.DataFrame(rows).to_csv(R/'METHOD_SUMMARY.tsv',sep='\t',index=False)
    c=pd.DataFrame(contrasts);c['p_raw']=np.nan;c['p_holm']=np.nan;c['family_size']=np.nan
    ix=(c.population.eq('LUAD') & c.budget.eq(a['primary_budget']) & c.overhead.eq(a['primary_overhead']) & c.metric.eq('branch_recall') & c.method.isin(['CTMC_BRIDGE','EMPIRICAL_BRIDGE','DIRECT_RIDGE']))
    w=d[d.population.eq('LUAD')&d.budget.eq(a['primary_budget'])&d.overhead.eq(a['primary_overhead'])].pivot(index='patient_id',columns='method',values='branch_recall')
    p=[paired_test(w[m]-w.BEST_FIXED,[20261009,3,j]) for j,m in enumerate(c.loc[ix,'method'])]
    c.loc[ix,'p_raw']=p;c.loc[ix,'p_holm']=holm(p);c.loc[ix,'family_size']=3
    c.to_csv(R/'PAIRED_CONTRASTS.tsv',sep='\t',index=False)
    decision={'new_run':True,'independent_unit':'patient after averaging five initial-region perturbations','primary':c[ix].to_dict('records'),
      'g5_diagnostic_value_supported':bool(((c.loc[ix,'lo']>0)&(c.loc[ix,'p_holm']<.05)).any()),
      'CTMC_specific_improvement_supported':bool(((c.loc[ix&c.method.eq('CTMC_BRIDGE'),'lo']>0)&(c.loc[ix&c.method.eq('CTMC_BRIDGE'),'p_holm']<.05)).any()),
      'inference':'Paired patient bootstrap conditional on frozen development model; two-sided paired mean sign-flip with symmetric-null assumption and Holm3.',
      'new_theta_truth_claim':False,'monetary_cost_claim':False,'patient_source':'Previously explored TRACERx, current development/confirmation disjoint.'}
    (R/'ASSESSMENT.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2))
    print(json.dumps(decision,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
