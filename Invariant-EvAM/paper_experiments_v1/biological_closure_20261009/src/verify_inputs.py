"""Verify identity, input provenance and scorer/visible-data separation."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
O=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    protocol=json.loads((O/'configs/PROTOCOL.json').read_text())
    freeze=json.loads((O/'configs/PROTOCOL_FREEZE.json').read_text())
    assert sha(O/'configs/PROTOCOL.json')==freeze['protocol_sha256']
    assert sha(O/'configs/PATIENT_SPLIT.tsv')==freeze['split_sha256']
    split=pd.read_csv(O/'configs/PATIENT_SPLIT.tsv',sep='\t')
    assert not split.patient_id.duplicated().any()
    assert set(split.split)=={'development','confirmation'}
    for row in split.itertuples():
        u=int(hashlib.sha256(('Invariant-EvAM-bio-v1|'+row.patient_id).encode()).hexdigest()[:16],16)/2**64
        assert row.split==('development' if u<.6 else 'confirmation')
    for name,h in protocol['source_hashes'].items():
        assert sha(Path(protocol['native_source'])/name)==h,name
    A=O/'data/native_audit'
    patients=pd.read_csv(A/'patient_primary_tumour.tsv',sep='\t')
    assert not patients.patient_id.duplicated().any()
    assert set(patients.patient_id)<=set(split.patient_id)
    regions=pd.read_csv(A/'regions.tsv',sep='\t')
    assert not regions.RegionID.duplicated().any()
    markers=pd.read_csv(A/'branch_owned_markers.tsv.gz',sep='\t',dtype={'cluster':str})
    assert not markers[['tumour_id','mutation_id']].duplicated().any()
    assert not markers[['tumour_id','variant_key']].duplicated().any()
    branches=pd.read_csv(A/'branch_reference.tsv',sep='\t',dtype={'cluster':str})
    assert not branches[['tumour_id','cluster']].duplicated().any()
    assert branches.loc[branches.eligible_multimarker_reference,'reference_markers'].ge(3).all()
    assert not branches.loc[branches.eligible_multimarker_reference,'is_trunk'].any()
    observed=pd.read_csv(A/'region_variants.tsv.gz',sep='\t',nrows=50)
    forbidden=['cluster','DriverMut','PhyloCCF_SC','CCF_phylo_SC','Recurrence_time_use','relapse','is_trunk','eligible_multimarker_reference']
    assert not (set(forbidden)&set(observed.columns)), 'Scorer-only field in policy-facing observation table'
    C=O/'data/ctdna/processed'
    plasma=pd.read_csv(C/'Table_17_patient_mutation_data.tsv.gz',sep='\t',usecols=['PublicationID','tracerx_id','chromosome','position','reference','alternate','dao','ddp','days_post_surgery'])
    assert not plasma[['PublicationID','tracerx_id','chromosome','position','reference','alternate']].duplicated().any()
    assert not plasma[['dao','ddp','days_post_surgery']].isna().any().any()
    assert (plasma.dao>=0).all() and (plasma.ddp>=plasma.dao).all()
    # The plasma publication includes nine additional patients excluded from
    # the 421 tissue release. Preserve them and extend the SAME hash rule.
    cp=pd.DataFrame({'patient_id':sorted(set(plasma.PublicationID))})
    cp['split']=[('development' if int(hashlib.sha256(('Invariant-EvAM-bio-v1|'+x).encode()).hexdigest()[:16],16)/2**64<.6 else 'confirmation') for x in cp.patient_id]
    overlap=cp.merge(split,on='patient_id',suffixes=('_plasma','_native'))
    assert overlap.split_plasma.eq(overlap.split_native).all()
    cp.to_csv(O/'configs/CTDNA_PATIENT_SPLIT.tsv',sep='\t',index=False)
    extension={'reason':'Nine ctDNA patients are outside the tissue421 eligibility release; same hash split applied without outcomes or model results.',
        'native_overlap':len(overlap),'additional_plasma_patients':len(cp)-len(overlap),
        'ctdna_split_sha256':sha(O/'configs/CTDNA_PATIENT_SPLIT.tsv'),'protocol_unchanged':True}
    (O/'configs/CTDNA_SPLIT_EXTENSION.json').write_text(json.dumps(extension,indent=2))
    files={str(p.relative_to(O)):sha(p) for p in A.glob('*') if p.is_file()}
    files.update({str(p.relative_to(O)):sha(p) for p in C.glob('*.tsv.gz')})
    (O/'configs/DERIVED_INPUT_FREEZE.json').write_text(json.dumps(files,indent=2))
    result={'protocol_and_split_hashes':'PASS','native_input_hashes':'PASS','patients_unique':'PASS',
        'cluster_marker_identity':'PASS','observer_has_no_scorer_fields':'PASS',
        'plasma_mutation_counts':'PASS','derived_files_frozen':len(files),
        'patient_ids':len(split),'plasma_patients':plasma.PublicationID.nunique(),'additional_plasma_patients':len(cp)-len(overlap),
        'no_outcome_fitting_performed_by_this_check':True}
    (O/'audit/INPUT_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__': main()
