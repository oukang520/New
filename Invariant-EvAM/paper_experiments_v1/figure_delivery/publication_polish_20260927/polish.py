"""Postprocess selected vector figures only; scientific marks and data stay fixed."""
from pathlib import Path
import re,json,hashlib,io,argparse
import xml.etree.ElementTree as ET
import fitz
import numpy as np
from pypdf import PdfReader,PdfWriter,Transformation
from pypdf.generic import RectangleObject
P=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
NS='http://www.w3.org/2000/svg';ET.register_namespace('',NS);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
FONT='/System/Library/Fonts/Supplemental/Arial.ttf';BOLD='/System/Library/Fonts/Supplemental/Arial Bold.ttf'
def is_global(e,parents):
 while e in parents:
  e=parents[e]
  if e.get('id','').startswith('axes_'):return False
 return True

def decision(name,t):
 if name in ['G1A','G1B','G1C','G1D','G1E'] and t in ['a','b','c','d','e']:return ''
 if t.startswith('G1 | Fixed-cost recovery'):return t.split('·')[-1].strip().capitalize()
 if t.startswith('G2 | Exact sign inference'):return t.split('·')[-1].strip().capitalize()
 if t.startswith('G3 | Declared selection-bias bound'):return 'True effect = '+t.split('effect ')[-1]
 if t.startswith('G3 | Two-phase calibration'):return 'Total cost = '+t.split('total cost ')[-1]
 if t.startswith('G4 | Joint information at equal assay cost') or t.startswith('G4B | Equal-cost bridge measurements') or t.startswith('G4C | Hidden-event probability calibration') or t.startswith('G5 | Equal actual acquisition cost'):return t.split('·')[-1].strip()
 if t.startswith('Independent simulation |'):return t.split('|',1)[1].strip()
 if re.match(r'^G[1-5][A-Z–]*\s*\|',t) or t.startswith(('Direct-record |','Model-anchored |','Direct records |')):return ''
 if t=='Diagnostic acquisition relative to each comparator':return ''
 if len(t)>58 or re.match(r'^(?:\d[\d,]*(?:\s|/)|Four independent|Four-alias coverage|Open points:|All outcomes|Whiskers:|Crossbars:|Black diamonds|Maximum absolute|Main:|Caps:|Magnitude intervals|No calibration|Guarantee applies|The same selected|Additional-information|Positive =|Total =)',t):return ''
 return None

def operate(svg):
 name=svg.stem;pdf=svg.with_suffix('.pdf');root=ET.fromstring(svg.read_text())
 if root.get('data-publication-polish')=='20260927':return None
 par={c:p for p in root.iter() for c in p};changes=[]
 for e in root.iter('{'+NS+'}text'):
  if not is_global(e,par):continue
  t=''.join(e.itertext()).strip();new=decision(name,t)
  if new is not None and new!=t:changes.append((e,t,new))
 before={ext:sha(f) for ext,f in [('svg',svg),('pdf',pdf)]};doc=fitz.open(pdf);page=doc[0];W,H=page.rect.width,page.rect.height;replacements=[];removed=[];unmatched=[]
 spans=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l.get('spans',[])]
 for e,t,new in changes:
  matches=[s for s in spans if s['text'].strip()==t]
  rects=[fitz.Rect(s['bbox']) for s in matches]
  if not rects and len(t)>4:rects=page.search_for(t)
  if not rects:unmatched.append(t)
  else:
   for r in rects:
    # Text only: preserve underlying lines, markers, fills and comparison connectors.
    page.add_redact_annot(r,fill=None,cross_out=False)
   if new:
    if matches:s=min(matches,key=lambda s:s['origin'][1]);point=s['origin'];bold='Bold' in s['font']
    else:point=(rects[0].x0,rects[0].y1-2);bold=True
    replacements.append((point,new,bold))
  removed.append(dict(original=t,replacement=new,pdf_matches=len(rects)))
  if new:
   for child in list(e):e.remove(child)
   e.text=new;st=e.get('style','');st=re.sub(r'font-size:\s*[^;]+','font-size: 8px',st);st=re.sub(r'(font:\s*(?:[^;]*?))\b[\d.]+px',r'\g<1>8px',st);e.set('style',st)
  else:par[e].remove(e)
 assert not unmatched,(name,'unmatched PDF text',unmatched)
 page.apply_redactions(images=0,graphics=0,text=0)
 for point,t,bold in replacements:page.insert_text(point,t,fontsize=8,fontname='PublicationArialBold' if bold else 'PublicationArial',fontfile=BOLD if bold else FONT,color=(0,0,0))
 # Audit Arial on all retained SVG text, without altering sizes/weights or mathematical symbols.
 for e in root.iter('{'+NS+'}text'):
  st=e.get('style','');st=re.sub(r'font-family:\s*[^;]+',"font-family: 'Arial'",st)
  if 'font:' in st:
   st=re.sub(r"(font:\s*[^;]*?\b[\d.]+px)\s+[^;]+",r"\1 'Arial'",st)
  elif 'font-family:' not in st:st+=";font-family: 'Arial'"
  e.set('style',st)
 # Crop only outer whitespace; no stretching, axes rescaling, or data-path edits.
 pix=page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False,colorspace=fitz.csRGB);im=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,3);mask=im.min(axis=2)<248;y,x=np.where(mask);pad=6
 crop=[max(0,float(x.min()/2-pad)),max(0,float(y.min()/2-pad)),min(W,float((x.max()+1)/2+pad)),min(H,float((y.max()+1)/2+pad))];cx,cy,x1,y1=crop;ww=x1-cx;hh=y1-cy
 root.set('width',f'{ww:.6f}pt');root.set('height',f'{hh:.6f}pt');root.set('viewBox',f'{cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}');root.set('data-publication-polish','20260927');root.set('data-export-dpi','600')
 reader=PdfReader(io.BytesIO(doc.tobytes(garbage=4,deflate=True)));pp=reader.pages[0]
 if '/EvAMAnnotationBase' in pp:del pp['/EvAMAnnotationBase']
 pp.add_transformation(Transformation().translate(-cx,-(H-y1)));pp.mediabox=RectangleObject([0,0,ww,hh]);pp.cropbox=RectangleObject([0,0,ww,hh]);writer=PdfWriter();writer.add_page(pp);writer.add_metadata({'/Title':name,'/Subject':'Arial vector artwork; current selected result; 600 dpi export convention'})
 temp=pdf.with_suffix('.tmp');writer.write(temp);temp.replace(pdf);svg.write_bytes(ET.tostring(root,encoding='utf-8',xml_declaration=True));doc.close()
 return dict(panel=name,group=name[:2].lower(),before=before,after={'svg':sha(svg),'pdf':sha(pdf)},removed_or_shortened=removed,old_size_pt=[W,H],crop_pt=crop,new_size_pt=[ww,hh],area_reduction=1-ww*hh/(W*H),font='Arial',dpi=600,scientific_marks_unchanged=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--panels',nargs='*');args=a.parse_args();mf=O/'MANIFEST.json';old=json.loads(mf.read_text()) if mf.exists() else [];new=[]
 for svg in sorted(P.glob('g*/final_figures/*.svg')):
  if args.panels and svg.stem not in args.panels:continue
  r=operate(svg)
  if r:new.append(r);print(r['panel'],len(r['removed_or_shortened']),f"area -{r['area_reduction']:.1%}",flush=True)
  # Commit per panel so a later failure cannot lose the audit of already changed files.
  ids={r['panel'] for r in new};mf.write_text(json.dumps([r for r in old if r['panel'] not in ids]+new,ensure_ascii=False,indent=2))
