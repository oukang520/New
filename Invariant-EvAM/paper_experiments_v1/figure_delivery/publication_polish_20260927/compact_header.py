from pathlib import Path
import json,copy,hashlib
import xml.etree.ElementTree as ET
import fitz,numpy as np
from pypdf import PdfReader,PdfWriter,PageObject,Transformation
from pypdf.generic import RectangleObject
P=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;NS='http://www.w3.org/2000/svg';XL='http://www.w3.org/1999/xlink';ET.register_namespace('',NS);ET.register_namespace('xlink',XL)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
manifest=json.loads((O/'MANIFEST.json').read_text());by={z['panel']:z for z in manifest}
for pdf in sorted(P.glob('g*/final_figures/*.pdf')):
 svg=pdf.with_suffix('.svg');root=ET.fromstring(svg.read_text())
 if root.get('data-header-compacted'):continue
 doc=fitz.open(pdf);pg=doc[0];W,H=pg.rect.width,pg.rect.height;ys=[]
 for d in pg.get_drawings():
  if d.get('color') and max(d['color'])<.8:
   for it in d['items']:
    if it[0]=='l':
     a,b=it[1:3]
     if abs(a.x-b.x)<.05 and abs(a.y-b.y)>35:ys.append(min(a.y,b.y))
 limit=min(ys) if ys else 0
 pix=pg.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False,colorspace=fitz.csRGB);im=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,3);occupied=(im.min(axis=2)<248).any(axis=1);empty=~occupied[:max(0,int(limit*2))];runs=[];start=None
 for i,b in enumerate(list(empty)+[False]):
  if b and start is None:start=i
  elif not b and start is not None:
   if start>8 and (i-start)/2>=12:runs.append((start/2+3,i/2-3))
   start=None
 if not runs:doc.close();continue
 D=sum(b-a for a,b in runs);cx,cy,vw,vh=map(float,root.get('viewBox').split());intervals=[];a=0;shift=0
 for lo,hi in runs:intervals.append((a,lo,shift));a=hi;shift+=hi-lo
 intervals.append((a,H,shift));src=PdfReader(pdf).pages[0];newpage=PageObject.create_blank_page(width=W,height=H-D)
 for a,b,shift in intervals:
  q=copy.copy(src);q.cropbox=RectangleObject([0,H-b,W,H-a]);newpage.merge_transformed_page(q,Transformation().translate(0,-D+shift))
 out=ET.Element(root.tag,dict(root.attrib));out.set('height',f'{H-D:.6f}pt');out.set('viewBox',f'{cx:.6f} {cy:.6f} {vw:.6f} {H-D:.6f}');out.set('data-header-compacted','20260927');defs=ET.SubElement(out,'{'+NS+'}defs');base=ET.SubElement(defs,'{'+NS+'}g',{'id':'publication_compact_source'})
 for el in list(root):base.append(el)
 for i,(a,b,shift) in enumerate(intervals):
  id=f'publication_slice_{i}';cp=ET.SubElement(defs,'{'+NS+'}clipPath',{'id':id});ET.SubElement(cp,'{'+NS+'}rect',dict(x=str(cx),y=str(cy+a-shift),width=str(vw),height=str(b-a)));g=ET.SubElement(out,'{'+NS+'}g',{'clip-path':f'url(#{id})'});ET.SubElement(g,'{'+NS+'}use',{'{'+XL+'}href':'#publication_compact_source','transform':f'translate(0 {-shift})'})
 writer=PdfWriter();writer.add_page(newpage);writer.add_metadata({'/Title':pdf.stem,'/Subject':'Arial vector artwork; outer and header whitespace reduced'});tmp=pdf.with_suffix('.tmp');writer.write(tmp);doc.close();tmp.replace(pdf);svg.write_bytes(ET.tostring(out,encoding='utf-8',xml_declaration=True))
 z=by[pdf.stem];z['header_removed_bands_pt']=runs;z['new_size_pt']=[W,H-D];z['after']={'pdf':sha(pdf),'svg':sha(svg)};z['area_reduction']=1-W*(H-D)/np.prod(z['old_size_pt']);print(pdf.stem,'header reduction',D,flush=True)
 (O/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
