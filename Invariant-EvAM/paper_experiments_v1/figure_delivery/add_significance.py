"""Frozen exploratory comparisons; preserve data/geometry; vector annotation overlay."""
from pathlib import Path
import json, hashlib, itertools, re, io, shutil
import numpy as np
import pandas as pd
from scipy.stats import binomtest
P=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent
ROWS=[]; PANELS={}; SOURCES={}
def read(s):
 p=P/s; SOURCES[s]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p,sep='\t')
def panel(name, labels, key): PANELS[name]={'labels':labels,'key':key}
def continuous(v):
 v=np.asarray(v,float);assert len(v) and np.isfinite(v).all();n=len(v);obs=abs(v.mean());tol=1e-12*max(1,obs)
 if n<=16:
  z=np.array(list(itertools.product([-1,1],repeat=n)));p=float((np.abs(z@v/n)>=obs-tol).mean());test='exact paired mean sign-flip'
 else:
  rng=np.random.default_rng(20260924);k=0;B=49999
  for start in range(0,B,256):
   z=rng.choice([-1.,1.],size=(min(256,B-start),n));k+=int((np.abs(z@v/n)>=obs-tol).sum())
  p=(k+1)/(B+1);test='paired mean sign-flip MC49999 seed20260924'
 return p,test,n,float(v.mean())
def add(name,bar,contrast,unit,v=None,p=None,n=None,test=None,effect=None,**extra):
 if v is not None:p,test,n,effect=continuous(v)
 ROWS.append(dict(panel=name,bar_index=bar,contrast=contrast,unit=unit,n=n,effect=effect,p_raw=p,test=test,**extra));return len(ROWS)-1
def paired(name,bar,q,a,b,metric,index,unit):
 w=q.pivot(index=index,columns='method',values=metric)[[a,b]];assert w.notna().all().all();av=w[a].astype(bool);bv=w[b].astype(bool);ab=int((av&~bv).sum());ba=int((~av&bv).sum());p=binomtest(ab,ab+ba,.5).pvalue if ab+ba else 1.
 return add(name,bar,a+' vs '+b+'; '+metric,unit,p=p,n=len(w),test='exact paired McNemar',effect=float(av.mean()-bv.mean()),a_only=ab,b_only=ba)
# G1: parameter-draw units, NOT simulation rows.
d=read('g1/revision_r2/results/G1_R2_finite_raw.tsv')
for regime in ['rare','ordinary','frequent']:
 name='G1_R2_recovery_'+regime;labels=[]
 for budget in [400,4000]:
  w=d[d.regime.eq(regime)&d.n_standard.eq(budget)].groupby(['draw','design']).abs_error.mean().unstack()
  for m in ['duplicated','heterogeneous','missing_joint']:
   labels.append('ref' if m=='heterogeneous' else add(name,len(labels),f'{m} vs heterogeneous; budget={budget}; MAE','parameter draw (50 datasets averaged)',v=w[m]-w.heterogeneous))
 panel(name,labels,'vs heterogeneous; 4 parameter draws; Holm')
# G2 coverage vs nominal.
d=read('g2/results/G2_core_summary.tsv');name='G2_02';labels=[]
for m in ['B1','B2','B3','B4','B5','M']:
 for cell in range(12):
  r=d[d.method_id.eq(m)&d.cell_id.eq(cell)].iloc[0];n=int(r.n_interval_completed);k=int(r.n_covered)
  labels.append(add(name,len(labels),f'{m}; cell={cell}; coverage vs 0.95','completed simulation interval',p=binomtest(k,n,.95).pvalue,n=n,test='two-sided exact binomial',effect=k/n-.95,excluded=int(r.n_tasks-n)))
panel(name,labels,'coverage vs 95%; Holm (72 tests)')
# Official matched comparisons.
d=pd.concat([read(f'g2/results/B6_matched_cell{i}.tsv') for i in range(4)]).rename(columns={'method_id':'method'});d['correct']=d.called&~d.wrong;name='G2_04';labels=[]
for m in ['B6_OFFICIAL','M_MATCHED']:
 for metric in ['covers_truth','correct']:
  for cell in range(4):
   labels.append('ref' if m=='B6_OFFICIAL' else paired(name,len(labels),d[d.cell_id.eq(cell)],m,'B6_OFFICIAL',metric,'replicate_id','paired simulation dataset'))
panel(name,labels,'vs official within effect/endpoint; Holm')
# Exact discordant counts from frozen binary paired sufficient moments.
d=read('g2/results/G2_paired_comparisons.tsv');s=read('g2/results/G2_core_summary.tsv');name='G2_05';labels=[]
for m in ['B4','B5','M']:
 for design in ['strong','weak','nonidentifying']:
  if m=='M':labels.append('ref');continue
  cells=s[s.design.eq(design)].cell_id.unique();q=d[d.cell_id.isin(cells)&d.comparison.eq('M_vs_'+m)&d.metric.eq('correct_call')];assert len(q)==4
  ab=ba=n=0
  for r in q.itertuples():
   sd=r.n_pairs*r.paired_difference;ss=(r.n_pairs-1)*r.n_pairs*r.paired_mcse**2+r.n_pairs*r.paired_difference**2
   b,c=(ss+sd)/2,(ss-sd)/2;assert abs(b-round(b))<1e-5 and abs(c-round(c))<1e-5
   ab+=round(c);ba+=round(b);n+=r.n_pairs
  labels.append(add(name,len(labels),f'{m} vs M; {design}; correct_call','paired dataset; equal four-effect mixture',p=binomtest(ab,ab+ba,.5).pvalue if ab+ba else 1.,n=n,test='exact paired McNemar; discordances reconstructed from frozen binary moments',effect=(ab-ba)/n,a_only=ab,b_only=ba))
panel(name,labels,'vs M within design; Holm')
d=read('g2/revision_r2/results/G2_R2_raw.tsv')
for design in ['strong','weak']:
 name='G2_R2_sign_'+design;labels=[]
 for effect in sorted(d.effect.unique()):
  q=d[d.design.eq(design)&d.effect.eq(effect)]
  for m in ['OLD_BONFERRONI_BANDS','EXACT_PER_ENV_BONFERRONI','EXACT_STRATIFIED','LOG_OR_WALD']:
   labels.append('ref' if m=='EXACT_STRATIFIED' else paired(name,len(labels),q,m,'EXACT_STRATIFIED','correct','replicate','paired simulation dataset'))
 panel(name,labels,'vs stratified exact within effect; Holm')
# G3 reference anchor.
d=read('g3/revision_r3/results/anchor_evaluation_raw.tsv');d['correct']=d.called&~d.wrong;name='G3_R3_reference_anchor';labels=[]
for cell in [3,10]:
 for m in ['NO_ANCHOR','BALANCED_X','REFERENCE_X']:
  labels.append('ref' if m=='REFERENCE_X' else paired(name,len(labels),d[d.cell.eq(cell)],m,'REFERENCE_X','correct','replicate','paired simulation dataset'))
panel(name,labels,'vs reference X within scenario; Holm')
# G4 matched perturbations; conditioned on patient pool.
d=read('g4/revision_r4/results/evaluation_raw.tsv');designs=['REDUNDANT','HETEROGENEOUS','CROSSVIEW_FAILURE']
for cancer in ['LUAD','COAD']:
 name='G4_R4_B_gain_'+cancer;labels=[]
 for design in designs:
  w=d[d.cancer.eq(cancer)&d.design.eq(design)].pivot(index='replicate',columns='method',values='log_score')
  for m in ['CTMC','POOLED','HIERARCHICAL','TWO_STEP']:
   labels.append(add(name,len(labels),f'{design}; MORE_{m} minus BRIDGE_{m}','paired perturbation seed; fixed patient pool',v=w['MORE_'+m]-w['BRIDGE_'+m]))
 panel(name,labels,'paired gain vs 0; conditional seed test; Holm')
 name='G4_R4_C_calibration_'+cancer;labels=[]
 for design in designs:
  w=d[d.cancer.eq(cancer)&d.design.eq(design)].pivot(index='replicate',columns='method',values='ece')
  for m in ['BASE_CTMC','AUDIT_CELL_16','BRIDGE_CTMC','AUDIT_MARGINAL_1']:
   labels.append('ref' if m=='BASE_CTMC' else add(name,len(labels),f'{design}; {m} vs BASE_CTMC; ECE','paired perturbation seed; fixed patient pool',v=w[m]-w.BASE_CTMC))
 panel(name,labels,'vs base CTMC; conditional seed test; Holm')
d=read('g4/revision_r4/figure_data/G4_R4_C_proper_score_controls.tsv');name='G4_R4_C_proper_score_controls';labels=[]
for cancer in ['LUAD','COAD']:
 for m in ['MORE_CTMC','AUDIT_CELL_16','BRIDGE_CTMC','AUDIT_DIRECT_1','AUDIT_MARGINAL_1']:
  labels.append(add(name,len(labels),f'{cancer}; BASE_CTMC minus {m}','paired complete three-design seed block; fixed patient pool',v=d[d.cancer.eq(cancer)&d.method.eq(m)].gain_millinats))
panel(name,labels,'paired gain vs 0; complete seed blocks; Holm')
d=read('g4/figure_data/G4_finite_calibration_ablation.tsv');name='G4_finite_calibration_ablation';labels=[]
for design in designs:
 for cancer in ['LUAD','COAD']:
  labels.append(add(name,len(labels),f'{cancer}; {design}; no calibration minus calibration','paired perturbation seed; fixed patient pool',v=d[d.cancer.eq(cancer)&d.design.eq(design)].gain))
panel(name,labels,'paired gain vs 0; conditional seed test; Holm')
d=read('g4/results/G4_model_anchored_auxiliary.tsv');name='G4_07';labels=[]
for cancer in ['LUAD','COAD']:
 for bound in [3,6]:
  w=d[d.cancer.eq(cancer)&d.bound.eq(bound)].groupby(['source_draw','method']).covers_truth.mean().unstack()
  labels.extend([add(name,len(labels),f'{cancer}; bound={bound}; M vs K0 noncoverage','source fit (10 nested simulations averaged)',v=w.K0-w.M),'ref'])
panel(name,labels,'vs K0 within source bound; 10 source fits; Holm')
# G5 anchored outcomes and conditional natural transfer.
d=read('g5/revision_r3/results/anchor_interval_evaluation_raw.tsv');d['method']='ANCHOR_'+d.method;d=pd.concat([d,read('g5/revision_r3/results/anchor_baseline_raw.tsv')])
for metric in ['called','covered']:
 name='G5_R3_anchor_'+metric;labels=[]
 for scenario in [0,1,2]:
  for m in ['COUNT_EXPANSION_CLASSIC','ANCHOR_CLASSIC','ANCHOR_CALIBRATED']:
   labels.append('ref' if m=='COUNT_EXPANSION_CLASSIC' else paired(name,len(labels),d[d.scenario.eq(scenario)],m,'COUNT_EXPANSION_CLASSIC',metric,'replicate','paired simulation draw'))
 panel(name,labels,'vs more selected patients within scenario; Holm')
d=read('g5/revision_r3/results/natural_patient_scores.tsv');name='G5_R3_natural_transfer';labels=[]
for heldout in ['MSK-IMPACT410','MSK-IMPACT505']:
 w=d[d.heldout.eq(heldout)].pivot(index='patient_key',columns='method',values='log_score')
 for m in ['LEGACY_IDENTITY_SHARED','LEGACY_BURDEN_SELECTION_SHARED','LEGACY_PRESPECIFIED_GENE_SELECTION','TARGET_ONLY_DIRICHLET','POOLED_TRAINING']:
  labels.append(add(name,len(labels),f'{heldout}; {m} minus TRANSFER_SHRINKAGE','paired patient; conditional on fitted models',v=w[m]-w.TRANSFER_SHRINKAGE))
panel(name,labels,'paired gain vs 0; conditional patient test; Holm')
# Holm, one frozen family per updated figure.
for name in PANELS:
 ids=[i for i,r in enumerate(ROWS) if r['panel']==name];order=sorted(ids,key=lambda i:ROWS[i]['p_raw']);prev=0
 for rank,i in enumerate(order):
  prev=max(prev,min(1,ROWS[i]['p_raw']*(len(order)-rank)));ROWS[i]['p_holm']=prev;ROWS[i]['family_size']=len(order);ROWS[i]['label']='ns' if prev>=.05 else '***' if prev<.001 else '**' if prev<.01 else '*'
pd.DataFrame(ROWS).to_csv(OUT/'SIGNIFICANCE_TESTS.tsv',sep='\t',index=False)
(OUT/'SIGNIFICANCE_SOURCES.json').write_text(json.dumps(SOURCES,indent=2))
(OUT/'SIGNIFICANCE_PANEL_PLAN.json').write_text(json.dumps(PANELS,indent=2))
print('Computed',len(ROWS),'tests for',len(PANELS),'figures',flush=True)
# Native vector overlays: only a compact annotation row above main plot, no mark shifts.
import xml.etree.ElementTree as ET
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pypdf import PdfReader,PdfWriter,PageObject
from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject
plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','font.family':'Arial'})
NS='http://www.w3.org/2000/svg';ET.register_namespace('',NS)
backup=Path('/tmp/invariant_significance_originals');backup.mkdir(exist_ok=True)
SPECS=[]
for name,plan in PANELS.items():
 group=name[:2].lower();svg=P/group/'final_figures'/f'{name}.svg';pdf=svg.with_suffix('.pdf')
 for f in [svg,pdf]:
  b=backup/f.name
  if not b.exists():shutil.copy2(f,b)
 root=ET.fromstring(svg.read_text());
 for old in list(root):
  if old.get('id')=='statistical_annotations':root.remove(old)
 W,H=map(float,root.attrib['viewBox'].split()[2:]);ax=next(e for e in root.iter() if e.get('id')=='axes_1');bars=[]
 for e in ax.iter('{'+NS+'}path'):
  style=e.get('style','');path=e.get('d','')
  if 'clip-path' in e.attrib and 'fill: #' in style and path.count('L')==3:
   xy=np.array([float(z) for z in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',path)]).reshape(-1,2)
   if len(np.unique(xy[:,0]))==2 and len(np.unique(xy[:,1]))<=2:bars.append((float(xy[:,0].mean()),e))
 assert len(bars)==len(plan['labels']),(name,len(bars),len(plan['labels']))
 clipid=re.search(r'#([^)]*)',bars[0][1].get('clip-path')).group(1);clip=next(e for e in root.iter() if e.get('id')==clipid);rect=next(iter(clip));top=float(rect.get('y'));y=top-5
 # Reserve existing whitespace above axes; dense G2 row uses rotated small labels.
 rotate=90 if len(bars)>40 else 0;size=5.5 if len(bars)>40 else 6.2
 fig=plt.figure(figsize=(W/72,H/72));fig.patch.set_alpha(0)
 g=ET.SubElement(root,'{'+NS+'}g',{'id':'statistical_annotations','aria-label':plan['key']})
 texts=[]
 for (x,_),label in zip(bars,plan['labels']):
  label=ROWS[label]['label'] if isinstance(label,int) else label
  props={'x':str(x),'y':str(y),'text-anchor':'middle','style':f'font: {size}px Arial; fill: #222222'}
  if rotate:props['transform']=f'rotate(-90 {x} {y})';props['text-anchor']='start'
  ET.SubElement(g,'{'+NS+'}text',props).text=label
  fig.text(x/W,1-y/H,label,ha='left' if rotate else 'center',va='baseline',fontsize=size,rotation=rotate,color='#222222')
  texts.append(dict(x=x,y=y,label=label))
 # Last margin line stays outside existing footnotes. Legend text is self-contained.
 key=plan['key']+' | ns >= .05; * < .05; ** < .01; *** < .001'
 keysize=4.7 if W<410 else 5.0
 ET.SubElement(g,'{'+NS+'}text',{'x':str(W*.15),'y':str(H-3),'style':f'font: {keysize}px Arial; fill: #333333'}).text=key
 fig.text(.15,3/H,key,fontsize=keysize,va='baseline',color='#333333')
 buf=io.BytesIO();fig.savefig(buf,format='pdf',dpi=600,transparent=True);plt.close(fig);buf.seek(0)
 original=PdfReader(str(pdf));source=original.pages[0]
 if '/EvAMAnnotationBase' in source:
  base=source['/EvAMAnnotationBase']
 else:
  original=PdfReader(str(backup/pdf.name));source=original.pages[0]
  base_stream=DecodedStreamObject();base_stream.set_data(source.get_contents().get_data())
  base=DictionaryObject({NameObject('/Contents'):base_stream,NameObject('/Resources'):source['/Resources']})
 # Fresh page and content object: never mutate the embedded base via an indirect reference.
 page=PageObject.create_blank_page(width=W,height=H)
 stream=DecodedStreamObject();stream.set_data(base['/Contents'].get_data())
 page[NameObject('/Contents')]=stream;page[NameObject('/Resources')]=base['/Resources']
 page.merge_page(PdfReader(buf).pages[0]);page[NameObject('/EvAMAnnotationBase')]=base
 writer=PdfWriter();writer.add_page(page)
 with pdf.open('wb') as f:writer.write(f)
 svg.write_bytes(ET.tostring(root,encoding='utf-8',xml_declaration=True))
 SPECS.append(dict(panel=name,key=key,annotations=texts,width_pt=W,height_pt=H,rotation=rotate))
(OUT/'SIGNIFICANCE_OVERLAY_SPEC.json').write_text(json.dumps(SPECS,indent=2))
print('Updated vector PDF/SVG pairs:',len(SPECS))
# Keep user-required explicit comparison connectors when regenerating legacy overlays.
import runpy
runpy.run_path(str(OUT/'bracket_revision/update.py'),run_name='__main__')
