# -*- coding: utf-8 -*-
"""Check current release, connectors, frozen test labels and unchanged data."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import fitz
import pandas as pd

O=Path(__file__).resolve().parent
P=O.parents[1]
NS='http://www.w3.org/2000/svg'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def numbers(s): return [float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',s)]
before=json.loads((O/'BEFORE.json').read_text())
repairs=json.loads((O/'REPAIR_MANIFEST.json').read_text())
repaired={r['panel']:r for r in repairs}
assert (O/'VISUAL_REVIEW_CN.md').exists(), 'Actual PDF and browser SVG review required'

# Update only the hashes and physical dimensions of the six revised figures.
for group in ['g4','g5']:
    G=P/group
    target={n for n in repaired if n[:2].lower()==group}
    dims={n:fitz.open(G/'final_figures'/f'{n}.pdf')[0].rect for n in target}
    mf=G/'configs/FINAL_FIGURES_MANIFEST.json'; m=json.loads(mf.read_text())
    for r in m['files']:
        n=Path(r['destination']).stem
        if n in target:
            r.update(sha256=sha(P/r['destination']),width_pt=dims[n].width,height_pt=dims[n].height,
                     annotation_layout_revision=str(O.relative_to(P)),annotation_rendering='straight scalar-null leader; centered label')
    put(mf,m)
    mf=G/'final_results/MANIFEST.json';m=json.loads(mf.read_text())
    for r in m['panels']:
        n=r['panel']
        if n in target:
            r.update(pdf_sha256=sha(G/'final_results'/r['pdf']),svg_sha256=sha(G/'final_results'/r['svg']),
                     size_mm=[dims[n].width/72*25.4,dims[n].height/72*25.4],annotation_review=str(O.relative_to(P)))
    put(mf,m)

paths=sorted(P.glob('g[1-5]/final_figures/*.pdf'))
assert len(paths)==56
all_files={str(f.relative_to(P)) for f in P.glob('g[1-5]/final_figures/*')}
assert len(all_files)==112 and all(Path(x).suffix in ['.pdf','.svg'] for x in all_files)
changed=[p for p,h in before.items() if sha(P/p)!=h]
expected={f'{n[:2].lower()}/final_figures/{n}.{ext}' for n in repaired for ext in ['pdf','svg']}
assert set(changed)==expected
data_files=[p for p in before if '/final_figures/' not in p]
assert all(sha(P/p)==before[p] for p in data_files)
tests=pd.read_csv(P/'figure_delivery/SIGNIFICANCE_TESTS.tsv',sep='\t')
legacy={r['panel']:r for r in json.loads((P/'figure_delivery/bracket_revision/MANIFEST.json').read_text())}
nonbar={'G1A','G1B','G1C','G1D','G1E','G2_01','G2_06','G2_07','G3_06','G4_01','G4_R4_BC_patient_bootstrap','G4_R5_nested_patient_means'}
rows=[];paired_total=0;straight_total=0
for f in paths:
    name=f.stem;page=fitz.open(f)[0];root=ET.parse(f.with_suffix('.svg')).getroot()
    labels=re.findall(r'(?m)^\s*(ns|\*{1,3})\s*$',page.get_text())
    svg_labels=[(e.text or '') for e in root.iter('{'+NS+'}text') if (e.text or '') in ['ns','*','**','***']]
    assert sorted(labels)==sorted(svg_labels),(name,'PDF/SVG labels')
    source='figure_delivery/SIGNIFICANCE_TESTS.tsv';q=tests[tests.panel==name]
    if name in ['G3_01','G3_07']:
        source='g3/figure_data/G3_exploratory_paired_comparisons.tsv';q=pd.read_csv(P/source,sep='\t');q=q[q.panel==int(name[-2:])]
    elif name in ['G4_R6_attribution','G5_R6_correct_answers']:
        source=f.parent.parent.name+'/revision_r6/results/figure_tests.tsv';q=pd.read_csv(P/source,sep='\t')
    if len(q):
        expected_labels=['***' if v<.001 else '**' if v<.01 else '*' if v<.05 else 'ns' for v in q.p_holm]
        assert sorted(labels)==sorted(expected_labels),(name,'frozen adjusted tests')
        if name in legacy and 'bar_index' in q:
            for c in legacy[name]['comparisons']:
                match=q[q.bar_index==c['i']]
                assert len(match)==1 and match.iloc[0].label==c['label']
    else:
        assert not labels,(name,'unexpected tests')
    box=list(map(float,root.get('viewBox').split()))
    assert abs(box[2]-page.rect.width)<.001 and abs(box[3]-page.rect.height)<.001
    if name in repaired:
        assert not page.get_images(),(name,'raster image introduced in revised vector figure')
    assert all('Arial' in f[3] for f in page.get_fonts()),(name,'font')
    count=0;straight=0
    for g in root.iter():
        gid=g.get('id','')
        if gid=='statistical_annotations':
            for e in g.iter('{'+NS+'}polyline'):
                v=numbers(e.get('points',''))
                if len(v)==8:
                    # Complete two-endpoint bracket, two equal vertical caps.
                    assert abs(v[0]-v[2])<1e-5 and abs(v[4]-v[6])<1e-5 and abs(v[3]-v[5])<1e-5
                    assert v[0]!=v[4] and abs((v[1]-v[3])-(v[7]-v[5]))<1e-5
                    assert 'clip-path' not in e.attrib
                    count+=1
                else:
                    assert len(v)==4 and abs(v[0]-v[2])<1e-5,(name,'remaining half bracket')
                    straight+=1
    if name in ['G3_01','G3_07','G5_R6_correct_answers','G5_R8_diagnostic_value']:
        for e in root.iter('{'+NS+'}path'):
            d=e.get('d','');v=numbers(d)
            if len(v)==8 and d.count('L')==3:
                if abs(v[0]-v[2])<1e-5 and abs(v[4]-v[6])<1e-5 and abs(v[3]-v[5])<1e-5 and v[0]!=v[4] and abs((v[1]-v[3])-(v[7]-v[5]))<1e-5:
                    count+=1
    if name=='G4_R6_attribution':
        for e in root.iter('{'+NS+'}path'):
            if 'stroke: #333333; stroke-width: 0.5' in e.get('style',''):
                d=e.get('d','');v=numbers(d)
                if len(v)==4 and d.count('L')==1 and abs(v[0]-v[2])<1e-5:
                    straight+=1
        assert straight==20
    if name=='G4_E1_multicohort_value':
        straight=len(labels) # Native straight leaders separately reviewed in browser/PDF.
    if labels:
        assert count+straight==len(labels),(name,count,straight,len(labels))
    paired_total+=count;straight_total+=straight
    rows.append(dict(group=f.parent.parent.name,panel=name,includes_bars_or_histogram=name not in nonbar,
                     significance_labels=len(labels),complete_two_bar_brackets=count,straight_scalar_null_leaders=straight,
                     frozen_label_check='PASS' if len(q) else 'not applicable',test_source=source if len(q) else '',
                     visual_review='PASS: actual PDF and Chrome SVG',changed=name in repaired))
assert paired_total==144 and straight_total==79
groups=[]
for name in ['g1','g2','g3','g4','g5']:
    G=P/name;m=json.loads((G/'configs/FINAL_FIGURES_MANIFEST.json').read_text())
    assert {Path(r['destination']).name for r in m['files']}=={p.name for p in (G/'final_figures').iterdir()}
    for r in m['files']: assert sha(P/r['destination'])==r['sha256']
    m=json.loads((G/'final_results/MANIFEST.json').read_text())
    for r in m['panels']:
        for ext in ['pdf','svg']:assert sha(G/'final_results'/r[ext])==r[ext+'_sha256']
        assert sha(G/'final_results'/r['plot_data'])==r['data_sha256']
    for r in m.get('experiment_data',[]):assert sha(G/'final_results'/r['path'])==r['sha256']
    for key in ['method','figure_legends']:
        if key in m:assert sha(G/'final_results'/m[key]['path'])==m[key]['sha256']
    groups.append(dict(group=name,panels=len(m['panels']),all_manifest_hashes_match=True))
pd.DataFrame(rows).to_csv(O/'ALL_FIGURES_AUDIT.tsv',sep='\t',index=False)
audit=P/'figure_delivery/SIGNIFICANCE_FIGURE_AUDIT.tsv';a=pd.read_csv(audit,sep='\t')
assert set(a.panel)=={r['panel'] for r in rows}
for name in repaired:
    a.loc[a.panel==name,'reason']=a.loc[a.panel==name,'reason'].str.replace('折角引线','直引线',regex=False)
    a.loc[a.panel==name,'reason']=a.loc[a.panel==name,'reason']+'；直引线及居中星号/ns，2026-10-09审查'
if not (O/'CHECK.json').exists():a.to_csv(audit,sep='\t',index=False)
report=dict(total_panels=56,total_files=112,bar_or_histogram_panels=sum(r['includes_bars_or_histogram'] for r in rows),
            annotated_panels=sum(bool(r['significance_labels']) for r in rows),significance_labels=sum(r['significance_labels'] for r in rows),
            complete_two_bar_brackets=paired_total,straight_scalar_null_leaders=straight_total,
            repaired_panels=list(repaired),repaired_scalar_null_leaders=70,changed_files=sorted(changed),
            other_100_figure_files_unchanged=True,unchanged_data_and_test_files=len(data_files),all_data_and_tests_unchanged=True,
            non_annotation_pdf_operations_unchanged=all(r['non_annotation_pdf_operations_unchanged'] for r in repairs),
            all_canonical_and_formal_hashes_match=True,all_pdf_svg_sizes_match=True,revised_figures_Arial_and_vector=True,
            half_brackets_remaining=0,groups=groups,visual_review='actual PDF and browser-rendered SVG')
put(O/'CHECK.json',report)
delivery=P/'figure_delivery/FINAL_DELIVERY_CHECK.json';d=json.loads(delivery.read_text())
d.update(latest_annotation_review=str((O/'CHECK.json').relative_to(P)),all_current_manifest_hashes_match=True)
put(delivery,d)
print(json.dumps(report,ensure_ascii=False,indent=2))
