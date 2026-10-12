import pandas as pd,numpy as np
from g3_design import G

def main():
 d=pd.read_csv(G/'results/G3_replicate_results.tsv.gz',sep='\t',low_memory=False);rows=[]
 for cell,f in d.groupby('cell_id'):
  reference='SHARED_K1' if cell<6 else 'ORDINARY_ONLY' if cell<8 else 'OMITTED_U' if cell<12 else 'MISSPECIFIED_SHARED'
  if reference not in set(f.method_id):continue
  a=f[f.method_id==reference].set_index('replicate_id')
  for m in sorted(set(f.method_id)-{reference}):
   b=f[f.method_id==m].set_index('replicate_id');ids=a.index.intersection(b.index)
   for metric in ['correct_call','wrong_call','covers_truth','population_prediction_KL']:
    aa=a.loc[ids,metric];bb=b.loc[ids,metric];valid=aa.notna()&bb.notna();diff=bb[valid].astype(float)-aa[valid].astype(float);n=len(diff);se=diff.std(ddof=1)/np.sqrt(n) if n>1 else np.nan;mean=diff.mean()
    rows.append(dict(cell_id=cell,reference=reference,method=m,metric=metric,n_paired=n,paired_difference=mean,paired_mcse=se,normal95_low=mean-1.96*se,normal95_high=mean+1.96*se,meaning='method minus reference; coverage comparison complete paired records only'))
 for cell in [6,7]:
  f=d[d.cell_id==cell]
  for share in ['0.1','0.25']:
   reference='RANDOM_X_'+share;m='CALIBRATION_XS_'+share;a=f[f.method_id==reference].set_index('replicate_id');b=f[f.method_id==m].set_index('replicate_id');ids=a.index.intersection(b.index)
   for metric in ['correct_call','wrong_call','covers_truth','population_prediction_KL']:
    aa=a.loc[ids,metric];bb=b.loc[ids,metric];valid=aa.notna()&bb.notna();diff=bb[valid].astype(float)-aa[valid].astype(float);n=len(diff);se=diff.std(ddof=1)/np.sqrt(n) if n>1 else np.nan;mean=diff.mean()
    rows.append(dict(cell_id=cell,reference=reference,method=m,metric=metric,n_paired=n,paired_difference=mean,paired_mcse=se,normal95_low=mean-1.96*se,normal95_high=mean+1.96*se,meaning='XS minus random X at equal proxy budget; records counts differ by registered cost'))
 pd.DataFrame(rows).to_csv(G/'results/G3_paired_comparisons.tsv',sep='\t',index=False)
if __name__=='__main__':main()
