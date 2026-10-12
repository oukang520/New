from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import binomtest
P=Path(__file__).resolve().parents[2];CFG=json.loads((P/'closure_r5/configs/PROTOCOL.json').read_text())
def wilson(k,n):
 if n==0:return (np.nan,np.nan)
 p=k/n;z=1.95996398454;den=1+z*z/n;center=(p+z*z/(2*n))/den;delta=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return center-delta,center+delta

def g5():
 G=P/'g5/revision_r5';d=pd.read_csv(G/'results/policy_raw.tsv',sep='\t');d['coverage_end']=d.covered&d.valid;d['failed']=~d.valid;rows=[]
 for key,q in d.groupby(['scenario','reference_source_cost','policy']):
  z=dict(zip(['scenario','reference_source_cost','policy'],key));z.update(n=len(q),correct=q.correct.mean(),wrong=q.wrong.mean(),called=q.called.mean(),coverage=q.coverage_end.mean(),conditional_coverage=q.loc[q.valid,'covered'].mean(),finite=q.finite.mean(),failure=q.failed.mean(),cost=q.cost.mean())
  for metric in ['correct','wrong','coverage_end']:
   lo,hi=wilson(q[metric].sum(),len(q));z[metric+'_lo']=lo;z[metric+'_hi']=hi
  rows.append(z)
 pd.DataFrame(rows).to_csv(G/'results/policy_summary.tsv',sep='\t',index=False)
 pairs=[]
 for scope,q in [('CORE',d[d.scenario.lt(3)]),('OUTSIDE_FAMILY',d[d.scenario.eq(3)])]:
  for metric in ['correct','wrong','coverage_end']:
   w=q.pivot(index=['scenario','reference_source_cost','replicate'],columns='policy',values=metric)
   for ref in [m for m in w if m!='ROBUST_TARGET']:
    v=w.ROBUST_TARGET.astype(float)-w[ref].astype(float);rng=np.random.default_rng([CFG['seed'],901]);boots=[]
    groups=[z.to_numpy() for _,z in v.groupby(level=[0,1])]
    for i in range(4000):boots.append(np.mean([rng.choice(z,len(z)).mean() for z in groups]))
    lo,hi=np.quantile(boots,[.025,.975]);a=w.ROBUST_TARGET.astype(bool);b=w[ref].astype(bool);ab=int((a&~b).sum());ba=int((~a&b).sum());p=binomtest(ab,ab+ba,.5).pvalue if ab+ba else 1.
    pairs.append(dict(scope=scope,metric=metric,reference=ref,n=len(v),gain=v.mean(),lo=lo,hi=hi,p_raw=p,discordant_a=ab,discordant_b=ba))
 pd.DataFrame(pairs).to_csv(G/'results/policy_paired_summary.tsv',sep='\t',index=False)
 # Explicit figure families: each cost has 4 scenarios x 5 non-reference methods.
 tests=[]
 for cost in [4,8]:
  start=len(tests)
  for s in range(4):
   w=d[d.scenario.eq(s)&d.reference_source_cost.eq(cost)].pivot(index='replicate',columns='policy',values='correct')
   for m in ['COUNT_ONLY','RANDOM','MAX_SOURCE','C_OPT','D_OPT']:
    a=w[m].astype(bool);b=w.ROBUST_TARGET.astype(bool);ab=int((a&~b).sum());ba=int((~a&b).sum());p=binomtest(ab,ab+ba,.5).pvalue if ab+ba else 1.;tests.append(dict(reference_source_cost=cost,scenario=s,policy=m,reference='ROBUST_TARGET',n=len(w),p_raw=p))
  inds=sorted(range(start,len(tests)),key=lambda i:tests[i]['p_raw']);prev=0
  for rank,i in enumerate(inds):
   prev=max(prev,min(1,tests[i]['p_raw']*(len(inds)-rank)));tests[i]['p_holm']=prev;tests[i]['mark']='ns' if prev>=.05 else '***' if prev<.001 else '**' if prev<.01 else '*'
 pd.DataFrame(tests).to_csv(G/'results/figure_paired_tests.tsv',sep='\t',index=False)
 act=d.groupby(['scenario','reference_source_cost','policy','action']).size().rename('n').reset_index();act.to_csv(G/'results/action_frequencies.tsv',sep='\t',index=False)
 choices=d.pivot(index=['scenario','reference_source_cost','replicate'],columns='policy',values='action');stats=dict(confirmation_datasets=len(choices),policy_rows=len(d),same_copt=float(choices.ROBUST_TARGET.eq(choices.C_OPT).mean()),same_dopt=float(choices.ROBUST_TARGET.eq(choices.D_OPT).mean()),chosen_actions=choices.ROBUST_TARGET.value_counts().to_dict());(G/'results/DECISION_AUDIT.json').write_text(json.dumps(stats,indent=2))

def g4():
 G=P/'g4/revision_r5';d=pd.read_csv(G/'results/nested_raw.tsv',sep='\t');ledger=pd.read_csv(G/'results/nested_ledger.tsv',sep='\t');pairs=[]
 for comp,a,b in [('B','MORE_CTMC','BRIDGE_CTMC'),('C','BASE_CTMC','AUDIT_CELL_16')]:
  for metric in ['log_score','brier','ece']:
   w=d.pivot(index=['cancer','outer','inner','design'],columns='method',values=metric);ok=d.pivot(index=['cancer','outer','inner','design'],columns='method',values='fit_success');gain=(w[a]-w[b]).where(ok[a]&ok[b]);v=gain.rename('gain').reset_index();v['comparison']=comp;v['metric']=metric;pairs.append(v)
 pairs=pd.concat(pairs);pairs.to_csv(G/'results/nested_paired_raw.tsv',sep='\t',index=False);outer=[];summary=[]
 for key,q in pairs.groupby(['cancer','comparison','metric']):
  block=q.groupby(['outer','design']).gain.mean().unstack().reindex(columns=CFG['g4']['designs']);valid=block.notna().all(axis=1);means=block[valid].mean(axis=1);lo,hi=np.quantile(means,[.025,.975]);z=dict(zip(['cancer','comparison','metric'],key));summary.append(dict(**z,gain=means.mean(),lo=lo,hi=hi,n_outer=len(means),excluded_outer=int((~valid).sum()),outer_planned=50,inner_planned=8,completed_comparisons=int(q.gain.notna().sum())))
  outer += [dict(**z,outer=int(i),gain=float(v)) for i,v in means.items()]
 pd.DataFrame(outer).to_csv(G/'results/nested_outer_means.tsv',sep='\t',index=False);pd.DataFrame(summary).to_csv(G/'results/nested_summary.tsv',sep='\t',index=False)
 ledger.groupby(['cancer','design','status']).size().rename('n').reset_index().to_csv(G/'results/nested_capacity_summary.tsv',sep='\t',index=False)
 positive=pairs.assign(positive=pairs.gain.gt(0)).groupby(['cancer','design','comparison','metric']).positive.sum().rename('n_positive_completed').reset_index();positive['planned']=400;positive['positive_delivery_per_planned_attempt']=positive.n_positive_completed/400;positive.to_csv(G/'results/nested_planned_delivery.tsv',sep='\t',index=False)
if __name__=='__main__':
 import sys
 for group in sys.argv[1:] :globals()[group]()
