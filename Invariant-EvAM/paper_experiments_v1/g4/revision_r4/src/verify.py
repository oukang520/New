from pathlib import Path
import json,hashlib
import numpy as np
G=Path(__file__).resolve().parents[1];P=G.parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
checks={};fail=[]
old=json.loads((P/'reviewer_audit/redesign_r2/PRIOR_SCIENTIFIC_HASHES.json').read_text())
for p,h in old.items():
    if sha(P/p)!=h:fail.append('prior hash '+p)
checks['prior_scientific_hashes']=len(old)
raw=json.loads((P/'g4/configs/DATA_FREEZE.json').read_text())['source_files']
for p,v in raw.items():
    if sha(p)!=v['sha256']:fail.append('raw source hash '+p)
checks['raw_source_hashes']=len(raw)
dev=json.loads((G/'configs/DEVELOPMENT_PROTOCOL.json').read_text())
for cancer,h in dev['source_hashes'].items():assert sha(P/f'g4/data_private/reference_records/{cancer}.json')==h
checks['derived_reference_record_hashes']=len(dev['source_hashes'])
f=json.loads((G/'configs/EVALUATION_FREEZE.json').read_text());assert sha(G/'src/bridge.py')==f['code_sha256']
assert sha(G/'results/development_raw.tsv')==f['development_raw_sha256']
checks['frozen_code_and_development_results']='PASS'
for stage,n in [('evaluation',300),('bootstrap',1200)]:
    paths=sorted((G/'runs').glob(stage+'_*.json'));assert len(paths)==n,(stage,len(paths));done=0;capacity=0;fit_failures=0;scorechecks=0
    for p in paths:
        r=json.loads(p.read_text())
        if r['status']=='CAPACITY_FAILURE':capacity+=1;continue
        done+=1
        for arm,d in r['public_inputs'].items():
            partial=sum(map(sum,d['counts']));full=sum(map(sum,d['bridge']));cal=sum(map(sum,d['calibration']));assert cal==80
            assert 3*partial+4*full+4*cal==(560 if arm=='BASE' else 638)
            assert partial+full==({'BASE':80,'MORE':106,'BRIDGE':80}[arm])
        for method,z in r['methods'].items():
            if not z.get('fit',{}).get('success',True):fit_failures+=1
            pr=np.array(z['patient']['p']);y=np.array(z['patient']['y']);assert np.isfinite(pr).all() and ((pr>0)&(pr<1)).all();assert len(pr)==z['scores']['n']
            assert np.isclose(np.mean((pr-y)**2),z['scores']['brier'],atol=1e-12)
            assert np.isclose(-np.mean(y*np.log(pr)+(1-y)*np.log1p(-pr)),z['scores']['log_score'],atol=1e-12)
            assert len(set(z['patient']['index']))==len(pr);scorechecks+=1
    checks[stage]=dict(planned=n,completed=done,capacity_failures=capacity,fit_failures=fit_failures,recomputed_proper_scores=scorechecks)
assert not fail,fail
checks['implementation_checks']=json.loads((G/'reports/IMPLEMENTATION_CHECKS.json').read_text())
checks['figures']={ext:len(list((G.parent/'final_figures').glob('G4_R4_*.'+ext))) for ext in ['pdf','svg']};assert checks['figures']=={'pdf':6,'svg':6}
checks['status']='PASS_WITH_REPORTED_CAPACITY_FAILURES';checks['external_patient_validation']=False
(G/'reports/VERIFICATION.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
