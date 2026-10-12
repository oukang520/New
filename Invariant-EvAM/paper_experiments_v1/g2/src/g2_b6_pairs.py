from pathlib import Path
import numpy as np,pandas as pd
G=Path(__file__).resolve().parents[1]
def main():
 d=pd.concat([pd.read_csv(G/f'results/B6_matched_cell{i}.tsv',sep='\t') for i in range(4)]);d['correct_call']=d.called&~d.wrong;out=[]
 for cell,g in d.groupby('cell_id'):
  a=g[g.method_id=='M_MATCHED'].set_index('replicate_id');b=g[g.method_id=='B6_OFFICIAL'].set_index('replicate_id');assert a.index.equals(b.index)
  for metric in ['covers_truth','correct_call','called','wrong']:
   delta=a[metric].fillna(False).astype(float)-b[metric].fillna(False).astype(float);mean=float(delta.mean());se=float(delta.std(ddof=1)/np.sqrt(len(delta)));out.append({'cell_id':cell,'truth':float(g.truth.iloc[0]),'metric':metric,'n_pairs':len(delta),'M_minus_B6':mean,'paired_mcse':se,'paired_normal95_low':mean-1.96*se,'paired_normal95_high':mean+1.96*se,'endpoint':'MATCHED_CLINICAL_OBSERVATION'})
 pd.DataFrame(out).to_csv(G/'results/G2_B6_paired_comparisons.tsv',sep='\t',index=False)
if __name__=='__main__':main()
