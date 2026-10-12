"""Audit all raw tasks and independently expose derivation/status inconsistencies.
Raw JSON immutable. No selected-replicate suppression and no cutoff tuning.
"""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from scipy.optimize import brentq
R=Path(__file__).resolve().parents[3];G=R/'paper_experiments_v1/g2';sys.path.insert(0,str(G/'src'))
from g2_engine import snapshot,operators,Likelihood,wald,sign

def main():
 pub=json.loads((G/'configs/public_contract.json').read_text());notes=[];aliases=[];tasks=[]
 for path in sorted((G/'runs/core').glob('*.json')):
  a=json.loads(path.read_text());by={}
  for row in a['results']:by.setdefault(row['replicate_id'],[]).append(row)
  for art in a['details']:
   if 'exception' in art:continue
   o=art['observed'];cell=o['cell'];rep=o['replicate'];design=['strong','weak','nonidentifying'][cell//4];rows=by[rep];joint=art['joint'];ops=operators(pub['operators'][design]);lik=Likelihood([np.array(x) for x in o['counts']],ops);joint['x']=np.array(joint['x']);tasks.append({'cell_id':cell,'replicate_id':rep,'seed':json.dumps(o['seed']),'n_results':len(rows),'n_observed':sum(sum(x) for x in o['counts'])})
   for r in rows:
    if r['method_id'] not in ['B5','M_SECONDARY_WALD']:continue
    v={'primary':[0,0,0,1,0],'interaction_difference':[0,0,0,1,-1],'interaction_mean':[0,0,0,.5,.5]}[r['target_id']];w=wald(lik,joint,np.r_[v,np.zeros(lik.dim-5)])
    if abs(r['estimate']-w['estimate'])>1e-10 or r['computational_status']!=('COMPLETE' if joint['success'] else 'FAILED'):
     notes.append({'cell_id':cell,'replicate_id':rep,'method_id':r['method_id'],'target_id':r['target_id'],'issue':'STALE_DERIVED_WALD_ROW_AFTER_COMMON_REFIT','old_estimate':r['estimate'],'recomputed_estimate':w['estimate'],'final_joint_success':joint['success']})
   if design=='nonidentifying':
    base=joint['x'][:5].copy();p0=snapshot(base)[0];q0=p0[[2,3,6,7]].sum();witness=[]
    for target in [-1.,1.]:
     t=base.copy();t[3]=target
     def fun(x):
      t[1]=x;return snapshot(t)[0][[2,3,6,7]].sum()-q0
     t[1]=brentq(fun,-15,6,xtol=1e-12);q=snapshot(t)[0];err=max(abs(aa@q-aa@p0).max() for aa,ff in ops);witness.append({'target':target,'error':float(err),'theta':t.tolist()})
    for r in rows:
     if r['target_id']!='primary':continue
     excludes=any(not(r['ci_lower']<=w['target']<=r['ci_upper']) for w in witness)
     aliases.append({'cell_id':cell,'replicate_id':rep,'method_id':r['method_id'],'max_observable_error':max(w['error'] for w in witness),'witnesses':json.dumps(witness),'numerical_success':r['computational_status']=='COMPLETE','sign_call':r['sign_call'],'excludes_at_least_one_equivalent_sign_witness':excludes,'structural_false_certainty':r['called'] and max(w['error'] for w in witness)<1e-9})
 pd.DataFrame(notes,columns=['cell_id','replicate_id','method_id','target_id','issue','old_estimate','recomputed_estimate','final_joint_success']).to_csv(G/'results/G2_derivation_audit.tsv',sep='\t',index=False)
 t=pd.DataFrame(tasks);t.to_csv(G/'results/G2_task_inventory.tsv',sep='\t',index=False);assert not t.duplicated(['cell_id','replicate_id']).any();assert t.n_results.eq(14).all();assert t.n_observed.eq(1000).all()
 pd.DataFrame(aliases).to_csv(G/'results/G2_core_alias_audit.tsv',sep='\t',index=False);print('tasks',len(t),'derivation issues',len(notes),'alias rows',len(aliases))
if __name__=='__main__':main()
