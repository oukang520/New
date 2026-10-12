from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import binomtest
P=Path(__file__).resolve().parents[2];O=P/'closure_r8/reports'
g1=pd.read_csv(P/'g1/revision_r8/results/evaluation_geometry.tsv',sep='\t');pairs=g1.pivot(index=['d','topology','regime','replicate'],columns='design',values='target_information');pairs['ratio']=pairs.COMPLEMENTARY/pairs.DUPLICATE.replace(0,np.nan);pairs.to_csv(P/'g1/revision_r8/results/paired_information.tsv',sep='\t');g1.groupby(['d','topology','design']).agg(n=('replicate','size'),identified=('target_identified','mean'),stable=('rank_tolerance_stable','mean'),information=('target_information','median')).to_csv(O/'G1_BY_COMPLEXITY.tsv',sep='\t')
g2=pd.read_csv(P/'g2/revision_r8/results/evaluation_raw.tsv',sep='\t');ref=pd.read_csv(P/'g2/revision_r8/results/evaluation_refined_profile_v2.tsv',sep='\t');g2=pd.concat([g2,ref],ignore_index=True);g2.to_csv(P/'g2/revision_r8/results/evaluation_all_methods.tsv',sep='\t',index=False);rows=[]
for key,z in g2.groupby(['design','n','method']):
 row=dict(zip(['design','n','method'],key));row['replicates']=len(z)
 for metric in ['covered','finite','correct','wrong','valid']:
  k=int(z[metric].sum());ci=binomtest(k,len(z)).proportion_ci();row[metric]=k/len(z);row[metric+'_lo']=ci.low;row[metric+'_hi']=ci.high
 row['median_width']=z.width.median();nz=z[z.truth!=0];zz=z[z.truth==0];row['correct_nonzero']=nz.correct.mean();row['false_positive_zero']=zz.wrong.mean();rows.append(row)
s=pd.DataFrame(rows);s.to_csv(O/'G2_SUMMARY.tsv',sep='\t',index=False)
g5=pd.read_csv(P/'g5/revision_r8/results/evaluation_raw.tsv',sep='\t');freeze=json.loads((P/'g5/revision_r8/configs/COMPARATOR_FREEZE.json').read_text());fixed=freeze['best_fixed_common_action'];raw=g5[~g5.policy.isin(['TARGET_C_OPT','NO_ACTION'])];random=raw.groupby(['case','label','replicate','truth'])[['covered','finite','correct','wrong','valid','restricted_width']].mean().reset_index();random['policy']='RANDOM_FEASIBLE_EXPECTATION';fixeddf=g5[g5.policy==fixed].copy();fixeddf['policy']='DEVELOPMENT_BEST_FIXED';g5=pd.concat([g5,random,fixeddf],ignore_index=True);g5.to_csv(P/'g5/revision_r8/results/evaluation_all_policies.tsv',sep='\t',index=False)
g5.groupby(['label','policy']).agg(n=('replicate','size'),coverage=('covered','mean'),finite=('finite','mean'),correct=('correct','mean'),wrong=('wrong','mean'),width=('restricted_width','mean'),valid=('valid','mean')).to_csv(O/'G5_SUMMARY.tsv',sep='\t');summary=g5.groupby('policy').agg(n=('replicate','size'),coverage=('covered','mean'),finite=('finite','mean'),correct=('correct','mean'),wrong=('wrong','mean'),width=('restricted_width','mean'),valid=('valid','mean'));summary.to_csv(O/'G5_OVERALL.tsv',sep='\t');r=np.random.default_rng(817335);rows=[]
for comparison in ['NO_ACTION','MORE_SAME','REFERENCE_MARGIN','NEW_SOURCE','RANDOM_FEASIBLE_EXPECTATION','DEVELOPMENT_BEST_FIXED']:
 a=g5[g5.policy=='TARGET_C_OPT'].set_index(['case','replicate']);b=g5[g5.policy==comparison].set_index(['case','replicate']);w=(b.restricted_width-a.restricted_width).unstack('case').values;boots=w[r.integers(0,len(w),(10000,len(w)))].mean(axis=(1,2));rows.append(dict(comparator=comparison,mean_width_reduction=float(w.mean()),ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),unit='replicate block across four prespecified scenarios; exploratory percentile interval'))
pd.DataFrame(rows).to_csv(O/'G5_PAIRED_WIDTH.tsv',sep='\t',index=False)
print('G2\n',s[['design','n','method','covered','finite','correct','wrong','valid']].to_string(index=False));print('G5\n',summary.to_string());print(pd.DataFrame(rows).to_string(index=False))
