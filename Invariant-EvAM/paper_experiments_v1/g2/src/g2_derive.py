"""Versioned corrections to derived fields; raw simulation JSON never modified."""
import numpy as np
from g2_engine import Likelihood,operators,wald,sign
TARGETS={'primary':[0,0,0,1,0],'interaction_difference':[0,0,0,1,-1],'interaction_mean':[0,0,0,.5,.5]}
def normalize_batch(payload,public):
 rows=payload['results'];changes=[];by={}
 for r in rows:by.setdefault(r['replicate_id'],[]).append(r)
 for art in payload['details']:
  if 'exception' in art:continue
  obs=art['observed'];rs=by[obs['replicate']];joint=art['joint'];joint['x']=np.array(joint['x']);design=['strong','weak','nonidentifying'][obs['cell']//4];lik=None
  for r in rs:
   old=r.copy()
   if r['target_id']!='primary':r['structural_status']='NOT_EVALUATED_FOR_SECONDARY';r['null_leakage']=float('nan')
   if design=='nonidentifying':r['operator_knowledge']='K0_KNOWN_MEASUREMENT_NO_SELECTION'
   if r['method_id'] in ['B5','M_SECONDARY_WALD']:
    a=np.r_[TARGETS[r['target_id']],np.zeros(len(joint['x'])-5)];est=float(a@joint['x'])
    if abs(r['estimate']-est)>1e-10 or (r['computational_status']=='COMPLETE')!=joint['success']:
     if lik is None:lik=Likelihood([np.array(c) for c in obs['counts']],operators(public['operators'][design]))
     w=wald(lik,joint,a);r.update(estimate=w['estimate'],ci_lower=w['lo'],ci_upper=w['hi'],fitted_null_leakage=w['null_leakage'],computational_status='COMPLETE' if joint['success'] else 'FAILED')
   if np.isnan(r['ci_lower']) or np.isnan(r['ci_upper']):r['computational_status']='FAILED'
   ok=r['computational_status']=='COMPLETE';r['coverage_defined']=ok;r['covers_truth']=bool(r['ci_lower']<=r['truth']<=r['ci_upper']) if ok else None;r['sign_call']=sign(r['ci_lower'],r['ci_upper'],'COMPLETE' if ok else 'FAILED');r['called']=r['sign_call'] in ['POSITIVE','NEGATIVE'];r['correct_call']=r['called'] and ((r['sign_call']=='POSITIVE' and r['truth']>0) or (r['sign_call']=='NEGATIVE' and r['truth']<0));r['wrong_sign']=r['called'] and not r['correct_call'];r['boundary_hit']=bool(np.isinf(r['ci_lower']) or np.isinf(r['ci_upper']))
   fields=['estimate','ci_lower','ci_upper','computational_status','coverage_defined','covers_truth','sign_call','called','correct_call','wrong_sign','boundary_hit','structural_status','null_leakage','operator_knowledge'];changed=[]
   for k in fields:
    a,b=old.get(k),r.get(k)
    if isinstance(a,float) and isinstance(b,float) and np.isnan(a) and np.isnan(b):continue
    if a!=b:changed.append(k)
   if changed:changes.append({'cell_id':r['cell_id'],'replicate_id':r['replicate_id'],'method_id':r['method_id'],'target_id':r['target_id'],'changed_fields':changed,'before':{k:old.get(k) for k in changed},'after':{k:r.get(k) for k in changed},'reason':'Re-derive secondary/common-fit and failure fields from stored final fit; no re-estimation or threshold change'})
 primary={(r['cell_id'],r['replicate_id'],r['method_id']):r for r in rows if r['target_id']=='primary'}
 for risk in payload['risk']:
  r=primary[(risk['cell_id'],risk['replicate_id'],risk['method_id'])];ok=r['computational_status']=='COMPLETE'
  if risk['success']!=ok:changes.append({'cell_id':risk['cell_id'],'replicate_id':risk['replicate_id'],'method_id':risk['method_id'],'target_id':'risk_curve','changed_fields':['success'],'before':risk['success'],'after':ok,'reason':'Propagate all method-specific fit failures to risk curve'})
  risk['success']=ok
  if not ok:risk['called']=False;risk['wrong']=False
 return rows,payload['risk'],changes
