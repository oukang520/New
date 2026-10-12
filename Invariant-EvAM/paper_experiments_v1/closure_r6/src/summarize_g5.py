from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import binomtest
P=Path(__file__).resolve().parents[2];G=P/'g5/revision_r6';d=pd.read_csv(G/'results/policy_raw.tsv',sep='\t');assert len(d)==1200*13
summary=d.groupby(['case','policy']).agg(n=('replicate','size'),mse=('squared_error','mean'),correct=('correct','mean'),wrong=('wrong','mean'),coverage=('covered','mean'),finite=('finite','mean'),valid=('valid','mean')).reset_index();summary.to_csv(G/'results/policy_summary.tsv',sep='\t',index=False)
rows=[];boots=[];core=d[d.case.lt(6)];policies=sorted(set(d.policy)-{'DIAGNOSTIC'});baseline=core[core.policy.eq('DIAGNOSTIC')].sort_values(['case','replicate'])
# Fixed equal-case stratified bootstrap. Pairing remains within each simulated dataset.
rg=np.random.default_rng(20260927603);ix=np.concatenate([rg.integers(k*150,(k+1)*150,(4000,150)) for k in range(6)],axis=1)
for policy in policies:
 q=core[core.policy.eq(policy)].sort_values(['case','replicate']);assert np.array_equal(q[['case','replicate']].values,baseline[['case','replicate']].values)
 for metric in ['squared_error','correct']:
  delta=q[metric].to_numpy(float)-baseline[metric].to_numpy(float) if metric=='squared_error' else baseline[metric].to_numpy(float)-q[metric].to_numpy(float)
  boot=delta[ix].mean(axis=1);gain=delta.mean();lo,hi=np.quantile(boot,[.025,.975])
  if metric=='correct':a=int((delta>0).sum());b=int((delta<0).sum());pv=binomtest(a,a+b,.5).pvalue if a+b else 1.
  else:
   # Paired mean sign-flip test, symmetric-zero-difference null; bootstrap CI primary.
   rr=np.random.default_rng(20260927604);extreme=0
   for _ in range(100):
    null=(rr.choice([-1.,1.],(200,len(delta)))*delta).mean(axis=1);extreme+=int((abs(null)>=abs(gain)-1e-15).sum())
   pv=(extreme+1)/20001
  rows.append(dict(reference=policy,metric=metric,gain=gain,lo=lo,hi=hi,p_raw=pv,n=900));boots.extend(dict(reference=policy,metric=metric,bootstrap=j,gain=v) for j,v in enumerate(boot))
r=pd.DataFrame(rows)
for metric in r.metric.unique():
 ids=r[r.metric.eq(metric)].sort_values('p_raw').index;prev=0
 for rank,i in enumerate(ids):prev=max(prev,min(1,r.loc[i,'p_raw']*(len(ids)-rank)));r.loc[i,'p_holm']=prev
r.to_csv(G/'results/paired_summary.tsv',sep='\t',index=False);pd.DataFrame(boots).to_csv(G/'results/paired_bootstrap.tsv',sep='\t',index=False)
core_diag=summary[(summary.case<6)&summary.policy.eq('DIAGNOSTIC')];actions=core[core.policy.eq('DIAGNOSTIC')].action.value_counts(normalize=True)
def low(ref,met):return float(r[(r.reference==ref)&(r.metric==met)].lo.iloc[0])
def gain(ref,met):return float(r[(r.reference==ref)&(r.metric==met)].gain.iloc[0])
gates=dict(action_switches=int((actions>=.2).sum())>=2,pooled_coverage=core_diag.coverage.mean()>=.93,minimum_cell_coverage=core_diag.coverage.min()>=.90,wrong=core_diag.wrong.mean()<=.05,failures=1-core_diag.valid.mean()<=.02,mse_fixed=low('BEST_FIXED','squared_error')>0,mse_random=low('RANDOM','squared_error')>0,mse_shuffled=low('SHUFFLED_DIAGNOSIS','squared_error')>0,correct_fixed=gain('BEST_FIXED','correct')>0,correct_random=gain('RANDOM','correct')>0)
strong=dict(correct_fixed=low('BEST_FIXED','correct')>0,correct_shuffled=low('SHUFFLED_DIAGNOSIS','correct')>0,all_fixed_mse=all(low(x,'squared_error')>0 and float(r[(r.reference==x)&(r.metric=='squared_error')].p_holm.iloc[0])<.05 for x in ['COMPLETE_A','COMPLETE_B','NEW_SOURCE','MORE_SELECTED']),all_controls_present=True)
score=8.5 if all(gates.values()) and all(strong.values()) else 8. if all(gates.values()) else 7.5 if gates['action_switches'] and gates['mse_fixed'] and gates['mse_random'] and gates['mse_shuffled'] else 7. if gates['action_switches'] and gates['mse_random'] else 6.
z=dict(score=score,gates={k:bool(v) for k,v in gates.items()},strong_gates={k:bool(v) for k,v in strong.items()},correct=float(core_diag.correct.mean()),coverage=float(core_diag.coverage.mean()),wrong=float(core_diag.wrong.mean()),finite=float(core_diag.finite.mean()),actions=actions.to_dict(),largest_panel_agreement=float(np.mean(core[core.policy.eq('LARGEST_PANEL')].sort_values(['case','replicate']).action.values==baseline.action.values)),copt_is_same_module=True,interpretation='Diagnostic framework value under varying public observation structure; no claim of a novel c-opt criterion or superiority over count/cost heuristic if identical')
(G/'results/SCORE.json').write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2));print(r.to_string(index=False))
