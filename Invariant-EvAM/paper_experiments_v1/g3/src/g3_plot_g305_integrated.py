# coding: utf-8
"""Reference-detail revision of G3 panels 02–05 only. Frozen fits are read-only."""
from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch,Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap,Normalize
from g3_design import G,R
from g3_summarize import wilson
sys.path.insert(0,str(R/'src'))
from invariant_evam.figure_style import apply_publication_style
apply_publication_style({'svg.fonttype':'none','pdf.fonttype':42,'font.family':'Arial','font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'axes.linewidth':.55,'xtick.major.width':.55,'ytick.major.width':.55,'xtick.major.size':2,'ytick.major.size':2})
C=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7'];OUT=G/'FINAL_FIGURES_01_07';DATA=G/'figure_data';PRE=Path('/tmp/g3_fidelity');PRE.mkdir(exist_ok=True)
A=G/'archive/before_g305_integrated_insets_20260924';A.mkdir(parents=True,exist_ok=True)
for rel in ['FINAL_FIGURES_01_07','figure_data','reports','configs/FIGURE_MANIFEST.json','DELIVERY_MANIFEST.json']:
 src=G/rel;dst=A/rel
 if not dst.exists():
  dst.parent.mkdir(parents=True,exist_ok=True)
  if src.is_dir():shutil.copytree(src,dst)
  else:shutil.copy2(src,dst)
S=pd.read_csv(G/'results/G3_summary.tsv',sep='\t');Z=pd.read_csv(G/'results/G3_sensitivity_regions.tsv',sep='\t');M=json.loads((G/'configs/FIGURE_MANIFEST.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
KEEP={str(p):sha(p) for i in [1,2,3,4,6,7] for p in [OUT/f'G3_{i:02d}.pdf',OUT/f'G3_{i:02d}.svg']}
def figure(w,h):return plt.figure(figsize=(w/25.4,h/25.4))
def save(fig,i,data,ref,endpoint,adapt):
 name=f'G3_{i:02d}';data.to_csv(DATA/f'{name}.tsv',sep='\t',index=False)
 for ext in ['pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=600)
 fig.savefig(PRE/f'{name}.png',dpi=220)
 # Actual text bounding boxes are checked, not automatically marked accepted.
 fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.bbox.width,fig.bbox.height;outside=[]
 from matplotlib.text import Text
 for t in fig.findobj(Text):
  if not t.get_visible() or not t.get_text():continue
  b=t.get_window_extent(renderer)
  if b.x0<-1 or b.y0<-1 or b.x1>w+1 or b.y1>h+1:outside.append(t.get_text())
 (PRE/f'{name}_text_bounds.json').write_text(json.dumps(outside,ensure_ascii=False))
 rec=next(x for x in M if x['panel_id']==name);rec.update(primary_reference_id=ref,endpoint=endpoint,adaptations=adapt,revision='reference_details_20260924',size_mm=(fig.get_size_inches()*25.4).tolist(),source_data=str(DATA/f'{name}.tsv'),data_sha256=sha(DATA/f'{name}.tsv'),review_status='awaiting_visual_review',dpi=600,palette=C);plt.close(fig)
# 05: actual three-point sensitivity trajectories; real endpoint insets replace inapplicable time AUCs.
curve_specs=[(3,[0,.2,.5],C[1],'Bio + obs\nε=0,0.2,0.5'),(10,[0],C[0],'Common U\nε=0'),(10,[.2,.5],C[2],'Common U\nε=0.2,0.5')]
fig=figure(116,122);ax=fig.add_axes([.18,.15,.77,.73]);ax.axvspan(.5,1.025,facecolor='#eeeeee',zorder=0);ax.axvline(.5,c='#333333',lw=.9,ls='--',zorder=1);insets=[ax.inset_axes([.075,.745,.365,.205]),ax.inset_axes([.57,.745,.405,.205])];tab=[];inrows=[]
for cell,eps_list,c,label in curve_specs:
 points=[]
 for eta in [0,.5,1]:
  counts=[]
  for eps in eps_list:
   f=Z[(Z.cell==cell)&(Z.epsilon_log_rate==eps)&(Z.eta_selection_coefficient==eta)];n=len(f);assert n==20;wrong=int(f.sign.eq('POSITIVE').sum());called=int(f.sign.isin(['POSITIVE','NEGATIVE']).sum());failed=int((~f.status.eq('COMPLETE')).sum());unknown=n-called-failed;lo,hi=wilson(wrong,n);counts.append((wrong,called,unknown,failed));tab.append(dict(cell=cell,epsilon=eps,eta=eta,n=n,wrong=wrong,called=called,unknown=unknown,failed=failed,coverage=f.covers_truth.mean(),wrong_rate=wrong/n,wrong_low=lo,wrong_high=hi,inside_allowed_set=bool(f.true_mechanism_in_allowed_set.iloc[0])))
  assert len(set(t[0] for t in counts))==1,'Cannot merge distinct wrong-rate trajectories'
  points.append([eta,wrong/n,lo,hi])
  if eta==.5:
   for eps,count in zip(eps_list,counts):
    for k,event in enumerate([count[0],count[2]]):inrows.append(dict(curve=label,cell=cell,epsilon=eps,eta=eta,endpoint=['wrong','unknown'][k],n=n,count=event,rate=event/n,low=wilson(event,n)[0],high=wilson(event,n)[1],color=c))
 p=np.array(points);ax.plot(p[:,0],p[:,1]*100,lw=2.25,c=c,zorder=2);ax.errorbar(p[:,0],p[:,1]*100,yerr=np.maximum(0,[100*(p[:,1]-p[:,2]),100*(p[:,3]-p[:,1])]),fmt='o',ms=5.3,mec='#222222',mew=.9,mfc=c,ecolor='#222222',elinewidth=.8,capsize=2,zorder=3)
for k,a in enumerate(insets):
 a.patch.set_alpha(0)
 rows=[t for t in inrows if t['endpoint']==['wrong','unknown'][k]]
 for j,r in enumerate(rows):
  val=r['rate']*100;a.barh(j,val,height=.55,fc=r['color'],ec='#333333',lw=.5);a.errorbar(val,j,xerr=np.maximum(0,[[val-r['low']*100],[r['high']*100-val]]),fmt='none',ecolor='#222222',elinewidth=.5,capsize=.9)
 a.set(xlim=(0,106),ylim=(5.6,-.6),xticks=[0,50,100]);a.set_yticks(range(6),[('B' if t['cell']==3 else 'U')+f":{t['epsilon']:g}" for t in rows] if k==0 else []);a.tick_params(labelsize=5.1,length=1.5,pad=1);a.spines[['top','right']].set_visible(False);a.set_title(['Wrong at η=0.5','Unknown at η=0.5'][k],fontsize=6.3,pad=3)
ax.set(xlim=(-.025,1.025),ylim=(-3,155),xticks=[0,.5,1],yticks=[0,20,40,60,80,100],xlabel='Allowed selection coefficient η',ylabel='Wrong sign calls (%)');ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=7)
fig.legend(handles=[Patch(fc=c,ec='#333333',lw=.6,label=label) for _,_,c,label in curve_specs],loc='upper center',bbox_to_anchor=(.565,.985),ncol=3,frameon=False,fontsize=6.2,handlelength=1.4,handletextpad=.35,columnspacing=.9)
pd.DataFrame(inrows).to_csv(DATA/'G3_05_insets.tsv',sep='\t',index=False)
save(fig,5,pd.DataFrame(tab),'R04','Sensitivity paths and actual midpoint wrong/unknown rates','Two actual inset axes placed INSIDE one near-square master coordinate frame; no detached top row. Compact horizontal boxed-color legend, solid thick lines, black-edge circles and capped errors. Insets use real η=0.5 wrong/unknown rates, not invented time AUCs; all 20 repeats/curve. Dashed η=0.5 guide identifies the actual evaluation slice used in both insets; gray area denotes η≥0.5, not a time phase or identified-validity region. Headroom above 100% is reserved for insets, not data. Main trajectories merge only identical wrong counts; six separate configurations retained in each inset, without pooling N.')
assert all(sha(Path(k))==v for k,v in KEEP.items()),'Accepted figures changed'
(G/'configs/FIGURE_MANIFEST.json').write_text(json.dumps(M,ensure_ascii=False,indent=2))
(G/'runs/FIDELITY_PRESERVATION.json').write_text(json.dumps({'unchanged_accepted_files':KEEP,'changed_panels':[5],'review_status':'awaiting_visual_review'},indent=2))
print('Rebuilt only G3_05 with nested insets; all other figures byte-identical')
