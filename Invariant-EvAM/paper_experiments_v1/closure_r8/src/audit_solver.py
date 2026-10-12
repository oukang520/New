from pathlib import Path
import json
import refine_profile_v2 as ref
G=ref.G/'fresh_confirmation';D=G/'solver_audit';D.mkdir(exist_ok=True);ref.D=D;original=ref.fit
paths=[p for p in (G/'refined').glob('*.json') if not json.loads(p.read_text())['valid']];summary=[]
for path in paths:
 trace=D/(path.stem+'_trace.json')
 if trace.exists():
  row=json.loads(trace.read_text());summary.append({k:v for k,v in row.items() if k!='calls'});continue
 calls=[]
 def wrapped(*args,**kwargs):
  z=original(*args,**kwargs);calls.append(dict(fixed=kwargs.get('fixed'),nll=z['nll'],success=z['success'],boundary=z['boundary'],messages=[a['message'] for a in z['attempts']]));return z
 ref.fit=wrapped;ref.one(G/'runs'/path.name);unc=[c for c in calls if c['fixed'] is None];con=[c for c in calls if c['fixed'] is not None];row=dict(file=path.name,unconstrained_fits=len(unc),last_unconstrained_success=unc[-1]['success'],last_unconstrained_boundary=unc[-1]['boundary'],last_unconstrained_nll=unc[-1]['nll'],best_constrained_nll=min(c['nll'] for c in con),failed_fit_calls=sum(not c['success'] for c in calls),calls=calls);(D/(path.stem+'_trace.json')).write_text(json.dumps(row));summary.append({k:v for k,v in row.items() if k!='calls'})
(G/'solver_audit_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
