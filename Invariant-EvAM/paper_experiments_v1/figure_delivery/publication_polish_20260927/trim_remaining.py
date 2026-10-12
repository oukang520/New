from pathlib import Path
import xml.etree.ElementTree as E
import json,hashlib,io
import fitz,numpy as np
from pypdf import PdfReader,PdfWriter,Transformation
from pypdf.generic import RectangleObject
P=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;NS='http://www.w3.org/2000/svg';E.register_namespace('',NS);E.register_namespace('xlink','http://www.w3.org/1999/xlink');sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
m=json.loads((O/'MANIFEST.json').read_text());by={z['panel']:z for z in m}
for name in ['G2_01','G2_07','G2_R2_sign_strong','G2_R2_sign_weak']:
 svg=P/'g2/final_figures'/f'{name}.svg';pdf=svg.with_suffix('.pdf');root=E.fromstring(svg.read_text());par={c:p for p in root.iter() for c in p};todo=[]
 for e in root.iter('{'+NS+'}text'):
  t=''.join(e.itertext()).strip()
  if t=='Answerability and conditional error' or t.startswith(('All 800 paired estimates','Stratified exact: null wrong calls')):todo.append((e,t))
 if not todo:continue
 doc=fitz.open(pdf);pg=doc[0];H=pg.rect.height;W=pg.rect.width
 for e,t in todo:
  rr=pg.search_for(t);assert rr,(name,t)
  for rect in rr:pg.add_redact_annot(rect,fill=None,cross_out=False)
  par[e].remove(e);by[name]['removed_or_shortened'].append(dict(original=t,replacement='',pdf_matches=len(rr)))
 pg.apply_redactions(images=0,graphics=0,text=0);pix=pg.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False,colorspace=fitz.csRGB);a=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,3);y,x=np.where(a.min(axis=2)<248);x0=max(0,x.min()/2-6);y0=max(0,y.min()/2-6);x1=min(W,(x.max()+1)/2+6);y1=min(H,(y.max()+1)/2+6);w=x1-x0;h=y1-y0;cx,cy,_,_=map(float,root.get('viewBox').split());root.set('width',f'{w:.6f}pt');root.set('height',f'{h:.6f}pt');root.set('viewBox',f'{cx+x0:.6f} {cy+y0:.6f} {w:.6f} {h:.6f}');page=PdfReader(io.BytesIO(doc.tobytes(garbage=4,deflate=True))).pages[0];page.add_transformation(Transformation().translate(-x0,-(H-y1)));page.mediabox=RectangleObject([0,0,w,h]);page.cropbox=RectangleObject([0,0,w,h]);wr=PdfWriter();wr.add_page(page);tmp=pdf.with_suffix('.tmp');wr.write(tmp);doc.close();tmp.replace(pdf);svg.write_bytes(E.tostring(root,encoding='utf-8',xml_declaration=True));by[name].update(after=dict(pdf=sha(pdf),svg=sha(svg)),new_size_pt=[float(w),float(h)],area_reduction=float(1-w*h/np.prod(by[name]['old_size_pt'])));print(name,'cleaned')
(O/'MANIFEST.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
