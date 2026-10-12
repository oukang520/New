# -*- coding: utf-8 -*-
"""Replace orphan significance labels with mapped brackets/leaders, frozen tests unchanged."""
from pathlib import Path
import json,re,io,copy,hashlib
import xml.etree.ElementTree as ET
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pypdf import PdfReader,PdfWriter,PageObject,Transformation
from pypdf.generic import NameObject,DecodedStreamObject,DictionaryObject,RectangleObject
P=Path(__file__).resolve().parents[2];O=P/'figure_delivery';D=Path(__file__).resolve().parent
NS='http://www.w3.org/2000/svg';XL='http://www.w3.org/1999/xlink';ET.register_namespace('',NS);ET.register_namespace('xlink',XL)
plans=json.loads((O/'SIGNIFICANCE_PANEL_PLAN.json').read_text());tests=pd.read_csv(O/'SIGNIFICANCE_TESTS.tsv',sep='\t');plt.rcParams.update({'font.family':'Arial','pdf.fonttype':42,'svg.fonttype':'none'})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ledger=[]
def refindex(name,i):
 if name.startswith('G1_'):return (i//3)*3+1
 if name=='G2_04':return i-8
 if name=='G2_05':return 6+i%3
 if name.startswith('G2_R2'):return (i//4)*4+2
 if name=='G3_R3_reference_anchor':return (i//3)*3+2
 if 'C_calibration' in name:return (i//4)*4
 if name=='G4_07':return i+1
 if name.startswith('G5_R3_anchor'):return (i//3)*3
 return None
for name,plan in plans.items():
 svg=P/name[:2].lower()/'final_figures'/f'{name}.svg';pdf=svg.with_suffix('.pdf');root=ET.fromstring(svg.read_text());before={e:sha(f) for e,f in [('svg',svg),('pdf',pdf)]}
 # Reruns restore embedded original SVG group, never stack bands.
 previous=root.find('.//{'+NS+'}g[@id="bracket_base"]')
 if previous is not None:
  W,H=map(float,root.get('data-original-size').split());new=ET.Element(root.tag,{'width':f'{W}pt','height':f'{H}pt','viewBox':f'0 0 {W} {H}','version':'1.1'})
  for el in list(previous):new.append(copy.deepcopy(el))
  root=new
 else:W,H=map(float,root.attrib['viewBox'].split()[2:])
 for el in list(root):
  if el.get('id')=='statistical_annotations':root.remove(el)
 ax=next(e for e in root.iter() if e.get('id')=='axes_1');bars=[]
 for e in ax.iter('{'+NS+'}path'):
  st=e.get('style','');path=e.get('d','')
  if 'clip-path' in e.attrib and 'fill: #' in st and path.count('L')==3:
   xy=np.array([float(z) for z in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',path)]).reshape(-1,2)
   if len(np.unique(xy[:,0]))==2 and len(np.unique(xy[:,1]))<=2:bars.append(dict(x=float(xy[:,0].mean()),y=float(xy[:,1].min()),e=e))
 assert len(bars)==len(plan['labels']),(name,len(bars))
 clipid=re.search(r'#([^)]*)',bars[0]['e'].get('clip-path')).group(1);rect=next(iter(next(e for e in root.iter() if e.get('id')==clipid)));top=float(rect.get('y'));cut=top-10
 pairs=[]
 for i,l in enumerate(plan['labels']):
  if isinstance(l,int):pairs.append(dict(i=i,j=refindex(name,i),label=str(tests.iloc[l]['label']),test_row=l))
 paired=any(r['j'] is not None for r in pairs);levels=[]
 # Greedy interval packing retains correct endpoints and avoids crossing label tiers.
 for r in sorted(pairs,key=lambda r:abs(bars[r['i']]['x']-bars[r['j']]['x']) if r['j'] is not None else 0):
  if r['j'] is None:r['level']=0;continue
  l,h=sorted([bars[r['i']]['x'],bars[r['j']]['x']]);lev=0
  while lev<len(levels) and any(not(h+4<a or l-4>b) for a,b in levels[lev]):lev+=1
  if lev==len(levels):levels.append([])
  levels[lev].append((l,h));r['level']=lev
 band=12+len(levels)*10 if paired else 0;HH=H+band
 # Split at axes top to reserve a bracket band without rescaling data/axes/text.
 out=ET.Element(root.tag,{'width':f'{W}pt','height':f'{HH}pt','viewBox':f'0 0 {W} {HH}','version':'1.1','data-original-size':f'{W} {H}'})
 defs=ET.SubElement(out,'{'+NS+'}defs');base=ET.SubElement(defs,'{'+NS+'}g',{'id':'bracket_base'})
 for el in list(root):base.append(el)
 for id,y,height in [('bracket_upper',0,cut),('bracket_lower',cut,H-cut)]:
  cp=ET.SubElement(defs,'{'+NS+'}clipPath',{'id':id});ET.SubElement(cp,'{'+NS+'}rect',{'x':'0','y':str(y),'width':str(W),'height':str(height)})
 ET.SubElement(out,'{'+NS+'}rect',{'width':str(W),'height':str(HH),'fill':'white'})
 ET.SubElement(out,'{'+NS+'}use',{'{'+XL+'}href':'#bracket_base','clip-path':'url(#bracket_upper)'})
 g=ET.SubElement(out,'{'+NS+'}g',{'transform':f'translate(0 {band})'});ET.SubElement(g,'{'+NS+'}use',{'{'+XL+'}href':'#bracket_base','clip-path':'url(#bracket_lower)'})
 ann=ET.SubElement(out,'{'+NS+'}g',{'id':'statistical_annotations','aria-label':plan['key']});fig=plt.figure(figsize=(W/72,HH/72));fig.patch.set_alpha(0)
 def line(xs,ys):
  ET.SubElement(ann,'{'+NS+'}polyline',{'points':' '.join(f'{x},{y}' for x,y in zip(xs,ys)),'style':'fill:none;stroke:#222222;stroke-width:0.65;stroke-linejoin:miter'})
  fig.add_artist(Line2D(np.array(xs)/W,1-np.array(ys)/HH,transform=fig.transFigure,c='#222222',lw=.65))
 def text(x,y,s,size=6.2,rot=0):
  attrs={'x':str(x),'y':str(y),'text-anchor':'middle','style':f'font: {size}px Arial;fill:#222222'}
  if rot:attrs.update(transform=f'rotate(-90 {x} {y})',**{'text-anchor':'start'})
  ET.SubElement(ann,'{'+NS+'}text',attrs).text=s;fig.text(x/W,1-y/HH,s,ha='left' if rot else 'center',va='baseline',fontsize=size,rotation=rot)
 for r in pairs:
  x=bars[r['i']]['x']
  if r['j'] is not None:
   x2=bars[r['j']]['x'];y=top+band-9-r['level']*10;line([x,x,x2,x2],[y+3,y,y,y+3]);text((x+x2)/2,y-1.5,r['label'])
  else:
   # A scalar-null test has one bar endpoint; no half-bracket elbow.
   near=[bars[r['i']]['y']]
   for el in ax.iter():
    if el.tag.endswith('path') and 'clip-path' in el.attrib:
     vals=re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',el.get('d',''))
     if len(vals)%2==0:
      pts=np.array([float(v) for v in vals]).reshape(-1,2)
      if len(pts) and np.max(np.abs(pts[:,0]-x))<2:near.extend(pts[:,1][pts[:,1]>=top].tolist())
    if el.tag.endswith('use') or el.tag.endswith('text'):
     try:
      transform=el.get('transform','');tr=re.search(r'translate\(([-.0-9]+)[ ,]+([-.0-9]+)\)',transform)
      xx,yy=(float(tr[1]),float(tr[2])) if tr else (float(el.get('x')),float(el.get('y')))
      if abs(xx-x)<3.5 and yy>=top:
       extent=max(6,len(el.text or '')*3.7) if 'rotate(-90)' in transform else 6
       near.append(yy-extent if el.tag.endswith('text') else yy)
     except (TypeError,ValueError):pass
   peak=min(near);y=peak-3;line([x,x],[y-5,y]);text(x,y-8,r['label'],5.2 if len(bars)>40 else 6.2)
  r.update(x=x,reference_x=bars[r['j']]['x'] if r['j'] is not None else None)
 key=plan['key']+' | ns >= .05; * < .05; ** < .01; *** < .001';text(W*.55,HH-3,key,4.6)
 buf=io.BytesIO();fig.savefig(buf,format='pdf',dpi=600,transparent=True);plt.close(fig);buf.seek(0)
 src=PdfReader(str(pdf)).pages[0];basepdf=src['/EvAMAnnotationBase'];page0=PageObject.create_blank_page(width=W,height=H);stream=DecodedStreamObject();stream.set_data(basepdf['/Contents'].get_data());page0[NameObject('/Contents')]=stream;page0[NameObject('/Resources')]=basepdf['/Resources']
 page=PageObject.create_blank_page(width=W,height=HH)
 upper=copy.copy(page0);upper.cropbox=RectangleObject((0,H-cut,W,H));page.merge_transformed_page(upper,Transformation().translate(0,band))
 lower=copy.copy(page0);lower.cropbox=RectangleObject((0,0,W,H-cut));page.merge_page(lower)
 page.merge_page(PdfReader(buf).pages[0]);page[NameObject('/EvAMAnnotationBase')]=basepdf;writer=PdfWriter();writer.add_page(page)
 with pdf.open('wb') as f:writer.write(f)
 svg.write_bytes(ET.tostring(out,encoding='utf-8',xml_declaration=True))
 ledger.append(dict(panel=name,before=before,after=dict(svg=sha(svg),pdf=sha(pdf)),width_pt=W,height_pt=HH,added_band_pt=band,comparisons=pairs,kind='two-bar brackets' if paired else 'one-bar null-test straight leaders',statistical_values_unchanged=True))
(D/'MANIFEST.json').write_text(json.dumps(ledger,indent=2));print('Updated',len(ledger),'panels')
