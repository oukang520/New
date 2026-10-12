"""Uniform prespecified G2 bound diagnostic, never replaces frozen primary ±3 results."""
import json,hashlib,datetime
from pathlib import Path
import g5_natural as n

def main():
 f=n.G/'configs/BOUNDARY_DIAGNOSTIC_FREEZE.json'
 if not f.exists():
  assert not list((n.G/'runs/boundary_diagnostic').glob('*.json'))
  source=Path(__file__);f.write_text(json.dumps(dict(bounds=[4.5,6.0],specs=n.P['natural_bootstrap_specs'],reason='Test whether constrained natural incompatibility / finite target signs depend on G2 prespecified search bounds; diagnosis after primary results, no replacement or tuning',time=datetime.datetime.now().isoformat(),files={str(source.relative_to(n.R)):hashlib.sha256(source.read_bytes()).hexdigest()}),indent=2))
 rr=n.rows();counts=n.counts_for(rr,n.PANELS)
 for bound in [4.5,6.0]:
  n.P['theta_bound']=bound
  for reg in n.P['natural_specs']:
   if reg['id'] not in n.P['natural_bootstrap_specs']:continue
   for topology in ['SPARSE']+(['FULL'] if reg['id'] in n.P['competition_specs'] else []):
    path=n.G/f"runs/boundary_diagnostic/{reg['id']}_{topology}_{bound}.json"
    if path.exists():continue
    r=n.do_fit(reg,counts,topology,profiles=True);n.put(path,dict(bound=bound,spec_id=reg['id'],topology=topology,result=r));print(bound,reg['id'],topology,r['success'],r['deviance'],flush=True)
if __name__=='__main__':main()
