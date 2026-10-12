"""Score frozen acquisition actions against later tissue lineage support."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
O=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=O/'results/metastasis';out.mkdir(exist_ok=True)
    M=O/'data/metastasis_audit'
    s=pd.read_csv(M/'native_cluster_future_marker_support.tsv',sep='\t',dtype={'native_cluster':str,'future_cluster':str})
    s=s[s.multimarker_future_lineage_support & ~s.native_is_trunk]
    primary=pd.read_csv(O/'data/native_audit/cluster_region_support.tsv.gz',sep='\t',dtype={'cluster':str})
    primary=primary[primary.region_class.eq('SURGICAL_PRIMARY')&primary.detected&primary.eligible_multimarker_reference]
    allowed=set(zip(primary.tumour_id,primary.cluster))
    s=s[[ (t,c) in allowed for t,c in zip(s.tumour_id,s.native_cluster) ]]
    targets=s.groupby(['patient_id','tumour_id']).native_cluster.apply(set).to_dict()
    roster=pd.read_csv(M/'metastasis_regions.tsv',sep='\t')
    future=set(roster.loc[roster.sampleTypeDetail.eq('metachronousMet'),'patient_id'])
    rawpath=O/'results/acquisition/confirmation_raw.tsv.gz'
    raw=pd.read_csv(rawpath,sep='\t',keep_default_na=False)
    protocol=json.loads((O/'configs/PROTOCOL.json').read_text())
    freeze={'status':'FROZEN_SECONDARY_SCORING_RULE_BEFORE_METASTATIC_OUTCOMES',
      'acquisition_actions_sha256':sha(O/'results/acquisition/confirmation_actions_frozen.tsv'),
      'reference_sha256':sha(M/'native_cluster_future_marker_support.tsv'),
      'rule':'Primary-only nontrunk author cluster has >=3 shared own variants supported in one actual metachronous metastatic region; LN/synchronous not counted as later progression.',
      'population':'All acquisition confirmation patients with actual metachronous sampled tissue; no exclusion by seedingClones=True or performance. Reference-free patients retained with undefined ratio.',
      'analysis':'Prespecified secondary biological endpoint, patient average over frozen five initial-region rotations; no confirm action changed.'}
    (out/'SCORING_FREEZE.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2))
    r=raw[raw.patient_id.isin(future)].copy()
    def score(x):
        t=targets.get((x.patient_id,x.tumour_id),set());seen=set(str(x.recovered_branch_ids).split(';'))-{''}
        return pd.Series({'future_branch_reference_n':len(t),'future_branches_recovered':len(t&seen),
          'future_branch_recall':len(t&seen)/len(t) if t else np.nan,
          'future_reference_status':'MULTIMARKER_LINEAGE_SUPPORT' if t else 'NO_EVALUABLE_NONTRUNK_REFERENCE'})
    r=pd.concat([r,r.apply(score,axis=1)],axis=1)
    r.to_csv(out/'confirmation_raw.tsv.gz',sep='\t',index=False)
    keys=['population','patient_id','tumour_id','budget','overhead','method']
    q=r.groupby(keys,sort=True)[['future_branch_reference_n','future_branches_recovered','future_branch_recall']].mean().reset_index()
    q.to_csv(out/'confirmation_patients.tsv',sep='\t',index=False)
    rows=[]
    for (pop,b,o,method),v in q.groupby(['population','budget','overhead','method']):
        z=v.future_branch_recall.dropna().to_numpy();rng=np.random.default_rng([20261009,91,b,o,len(rows)])
        if len(z):
            bs=z[rng.integers(0,len(z),(2000,len(z)))].mean(1);mean=z.mean();lo,hi=np.quantile(bs,[.025,.975])
        else:mean=lo=hi=np.nan
        rows.append(dict(population=pop,budget=b,overhead=o,method=method,n_future_patients=len(v),n_evaluable_patients=len(z),mean=mean,lo=lo,hi=hi))
    pd.DataFrame(rows).to_csv(out/'SUMMARY.tsv',sep='\t',index=False)
    a=protocol['acquisition'];primary=pd.DataFrame(rows)
    primary=primary[primary.budget.eq(a['primary_budget'])&primary.overhead.eq(a['primary_overhead'])]
    result={'scope':'Real later tissue marker-supported lineage reference, not true theta or causal metastasis mechanism.',
      'all_future_source_patients':len(future),'all_future_source_confirmation_patients':int(pd.read_csv(O/'configs/PATIENT_SPLIT.tsv',sep='\t').query('split=="confirmation"').patient_id.isin(future).sum()),
      'primary_cost_cell':primary.to_dict('records'),'diagnostic_superiority_claim':False,
      'sample_size_interpretation':'Small biological bridge subset; no upgrade to a large external validation.'}
    (out/'ASSESSMENT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
