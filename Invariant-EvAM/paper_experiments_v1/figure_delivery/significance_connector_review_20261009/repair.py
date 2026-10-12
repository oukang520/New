# -*- coding: utf-8 -*-
"""Surgical vector-only correction; no scientific data or tests are recomputed."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import fitz
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, FloatObject, NameObject

O = Path(__file__).resolve().parent
P = O.parents[1]
NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
COUNTS = {
    'G4_R4_B_gain_LUAD': 12,
    'G4_R4_B_gain_COAD': 12,
    'G4_R4_C_proper_score_controls': 10,
    'G4_finite_calibration_ablation': 6,
    'G5_R3_natural_transfer': 10,
    'G4_R6_attribution': 20,
}

def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()

def nums(s):
    return [float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', s)]

def mark(args):
    if not args or not isinstance(args[0], list):
        return None
    value = ''.join(x for x in args[0] if isinstance(x, str))
    return value if value in ['ns', '*', '**', '***'] else None

baseline = json.loads((O/'BEFORE.json').read_text())
assert not (O/'REPAIR_MANIFEST.json').exists(), 'Already repaired; run verify.py instead'
ledger = []
for name, count in COUNTS.items():
    svg = next(P.glob('g*/final_figures/'+name+'.svg'))
    pdf = svg.with_suffix('.pdf')
    for f in [svg, pdf]:
        assert sha(f) == baseline[str(f.relative_to(P))]
    root = ET.parse(svg).getroot()
    native = name == 'G4_R6_attribution'
    edits = []
    if native:
        for e in root.iter('{'+NS+'}path'):
            v = nums(e.get('d', ''))
            if len(v) == 6 and e.get('d', '').count('L') == 2 and 'stroke: #333333; stroke-width: 0.5' in e.get('style', ''):
                x, low, x2, top, stub, top2 = v
                if abs(x-x2)<1e-5 and abs(top-top2)<1e-5 and abs(stub-x-3.931126)<1e-4:
                    e.set('d', f'M {x} {low} L {x} {top}')
                    edits.append((x, top))
        labels = [e for e in root.iter('{'+NS+'}text') if (e.text or '') in ['ns', '*', '**', '***']]
        assert len(edits) == len(labels) == count
        for label, (x, top) in zip(labels, edits):
            label.attrib.pop('transform', None)
            label.set('x', str(x)); label.set('y', str(top-3))
            label.set('text-anchor', 'middle')
    else:
        ann = next(e for e in root.iter() if e.get('id') == 'statistical_annotations')
        ann.set('data-connector', 'straight leader to one bar; scalar null zero')
        lines = list(ann.iter('{'+NS+'}polyline'))
        labels = list(ann.iter('{'+NS+'}text'))
        assert len(lines) == len(labels) == count
        for line, label in zip(lines, labels):
            v = nums(line.get('points'))
            assert len(v) == 6 and abs(v[2]-v[0]-2)<1e-5 and abs(v[1]-v[3])<1e-5 and abs(v[2]-v[4])<1e-5
            line.set('points', f'{v[2]},{v[3]} {v[4]},{v[5]}')
            label.set('x', str(v[2]))

    reader = PdfReader(pdf)
    page = reader.pages[0]
    stream = ContentStream(page.get_contents(), reader)
    operations = stream.operations
    leaders = []
    drop = set()
    allowed = set()
    for i, (args, op) in enumerate(operations):
        if op != b'm' or [j[1] for j in operations[i:i+4]] != [b'm', b'l', b'l', b'S']:
            continue
        v = [list(map(float, j[0])) for j in operations[i:i+3]]
        if native:
            matched = abs(v[0][0]-v[1][0])<1e-4 and abs(v[1][1]-v[2][1])<1e-4 and abs(v[2][0]-v[1][0]-3.931126)<1e-4
            if matched:
                leaders.append(v[1]); drop.add(i+2)
        else:
            matched = abs(v[1][0]-v[0][0]-2)<1e-4 and abs(v[0][1]-v[1][1])<1e-4 and abs(v[1][0]-v[2][0])<1e-4 and abs(v[1][1]-v[2][1]-5)<1e-4
            if matched:
                operations[i] = (operations[i+1][0], b'm')
                drop.add(i+1); allowed.add(i)
                leaders.append(v[1])
    # PDF includes clipped historical content segments from prior layout merging.
    # Correct every copy of the exact annotation geometry, without touching bars.
    assert len(leaders) == (count if native else count*(3 if 'B_gain' in name else 2))
    doc = fitz.open(pdf)
    font_xref = next(f[0] for f in doc[0].get_fonts() if 'Arial' in f[3])
    font = fitz.Font(fontbuffer=doc.extract_font(font_xref)[3])
    text_edits = 0
    for i, (args, op) in enumerate(operations):
        label = mark(args) if op == b'TJ' else None
        if label is None:
            continue
        ci = next(j for j in range(i-1, max(-1, i-20), -1) if operations[j][1] == b'cm')
        cm = list(map(float, operations[ci][0]))
        if native:
            assert abs(cm[0])<1e-5 and abs(cm[1]-1)<1e-5
            x, top = leaders[text_edits]
            cm = [1, 0, 0, 1, x-font.text_length(label, fontsize=5.6)/2, top+3]
        else:
            assert abs(cm[0]-1)<1e-5 and abs(cm[1])<1e-5
            cm[4] += 2
        operations[ci] = ([FloatObject(v) for v in cm], b'cm')
        allowed.add(ci); text_edits += 1
    assert text_edits == len(leaders)
    # All non-annotation PDF drawing and text operations must remain identical.
    old_operations = ContentStream(page.get_contents(), reader).operations
    for i, old in enumerate(old_operations):
        if i not in allowed and i not in drop:
            assert old == operations[i], (name, i)
    stream.operations = [op for i, op in enumerate(operations) if i not in drop]
    page[NameObject('/Contents')] = stream
    writer = PdfWriter(); writer.add_page(page)
    writer.add_metadata({'/Title': name, '/Producer': 'Invariant-EvAM vector annotation repair; 600 dpi export convention'})
    tmp = pdf.with_suffix('.tmp.pdf')
    with tmp.open('wb') as f:
        writer.write(f)
    assert sorted(re.findall(r'(?m)^\s*(ns|\*{1,3})\s*$', fitz.open(tmp)[0].get_text())) == sorted(re.findall(r'(?m)^\s*(ns|\*{1,3})\s*$', doc[0].get_text()))
    tmp.replace(pdf)
    svg.write_bytes(ET.tostring(root, encoding='utf-8', xml_declaration=True))
    ledger.append(dict(panel=name,scalar_null_labels=count,pdf_path_copies_corrected=len(leaders),
                       non_annotation_pdf_operations_unchanged=True,rendering='straight leader; centered horizontal label',
                       before={e:baseline[str(svg.with_suffix('.'+e).relative_to(P))] for e in ['pdf','svg']},
                       after={e:sha(svg.with_suffix('.'+e)) for e in ['pdf','svg']}))
(O/'REPAIR_MANIFEST.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2))
print('Corrected',len(ledger),'panels and',sum(x['scalar_null_labels'] for x in ledger),'visible scalar-null leaders')
