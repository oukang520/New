# -*- coding: utf-8 -*-
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import AutoMinorLocator,FixedLocator,FuncFormatter
from scipy.stats import gaussian_kde
HERE=Path(__file__).resolve().parent;P=HERE.parents[1];R=P.parent
sys.path.insert(0,str(R/'src'))
from invariant_evam.figure_style import apply_publication_style
apply_publication_style({'font.family':'Arial','font.size':7.5,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.65,'svg.fonttype':'none','pdf.fonttype':42})
C=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7']
CAT=json.loads((R/'configs/figure_reference_catalog.json').read_text())['records'];MAN=[]
DATA=HERE/'data';DATA.mkdir(exist_ok=True);PRE=Path('/tmp/evam_reference_20260925');PRE.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(s):
 p=P/s;return pd.read_csv(p,sep='\t'),dict(path=s,sha256=sha(p))
def save(f,name,ref,d,source,components,adapt):
 target=P/name[:2].lower()/'final_figures';dp=DATA/(name+'.tsv');d.to_csv(dp,sep='\t',index=False)
 old={ext:sha(target/(name+'.'+ext)) for ext in ['pdf','svg']}
 ledger=HERE/'REPLACEMENT_BEFORE_HASHES.json'
 if not ledger.exists():ledger.write_text('{}')
 history=json.loads(ledger.read_text());history.setdefault(name,old);ledger.write_text(json.dumps(history,indent=2))
 for ext in ['pdf','svg']:f.savefig(target/(name+'.'+ext),dpi=600)
 # No persistent preview renders outside final_figures.
 rr=next(x for x in CAT if x['id']==ref)
 MAN.append(dict(panel_id=name,primary_reference_id=ref,reference_file=rr['source_file_absolute'],reference_sha256=sha(Path(rr['source_file_absolute'])),source=source,plot_data=str(dp.relative_to(P)),plot_data_sha256=sha(dp),components=components,adaptations=adapt,palette=C,size_mm=(f.get_size_inches()*25.4).tolist(),dpi=600,review_status='awaiting_actual_visual_review',user_approved=False,outputs=[str((target/(name+'.'+e)).relative_to(P)) for e in ['pdf','svg']]))
 plt.close(f)
# R08: same four conditions in each of three diagnostic blocks.
d,src=read('g3/figure_data/G3_06.tsv');order=['time','measurement'];labs=['Gamma time','Measurement'];d=d[d.mechanism.isin(order)].copy()
f=plt.figure(figsize=(136/25.4,140/25.4));dens=[]
for block,(metric,label,c) in enumerate([('display_KL_x1000','Predictive KL (×0.001)',C[2]),('deviance','Training saturated deviance',C[5]),('target_effective_information','Effective target information',C[0])]):
 a=f.add_axes([.24,.725-block*.275,.71,.225]);values=d[metric].to_numpy();band=1.06*values.std(ddof=1)*len(values)**(-.2);hi=values.max()+3*band;grid=np.linspace(0,hi,700)
 density=[]
 for m in order:
  v=d[d.mechanism.eq(m)][metric].to_numpy();assert len(v)==200 and np.isfinite(v).all()
  # Boundary-reflected fixed Gaussian kernel; common bandwidth/height scale within each block.
  den=(np.exp(-.5*((grid[:,None]-v)/band)**2)+np.exp(-.5*((grid[:,None]+v)/band)**2)).mean(axis=1)/(band*np.sqrt(2*np.pi));density.append(den)
 peak=max(z.max() for z in density)
 for j,(m,l,z) in enumerate(zip(order,labs,density)):
  base=1-j;alpha=[.50,1][j];a.plot([0,hi],[base,base],c=c,alpha=alpha,lw=.7,zorder=j*3)
  a.fill_between(grid,base,base+z/peak*1.5,color=c,alpha=alpha,zorder=j*3+1);a.plot(grid,base+z/peak*1.5,c='white',lw=.6,zorder=j*3+2)
  a.text(-.035,base+.04,l,transform=a.get_yaxis_transform(),ha='right',va='bottom',fontsize=7,weight='bold',color=c,alpha=max(.65,alpha))
  dens.extend(dict(block=block,metric=metric,mechanism=m,x=float(x),density=float(y),bandwidth=float(band),n=200) for x,y in zip(grid,z))
 a.set(xlim=(0,hi),ylim=(-.08,2.6),xlabel=label);a.set_yticks([]);a.spines[['left','top','right']].set_visible(False);a.spines['bottom'].set_visible(False);a.tick_params(axis='x',length=2,pad=2);a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5));a.xaxis.labelpad=2
f.text(.24,.047,'200 datasets per row; retained time / measurement endpoints; no within-group filtering.',fontsize=6.2)
f.text(.24,.025,'Common bandwidth within each block; reflected KDE; density is not uncertainty.',fontsize=6.2)
pd.DataFrame(dens).to_csv(DATA/'G3_06_density.tsv',sep='\t',index=False)
save(f,'G3_06','R08',d,src,['three vertically stacked ridgeline blocks','colored direct row labels','thin full-width baselines','white ridge outlines','light-to-dark fill by row','one labeled x axis per block, no density ticks'],['Two retained mechanisms rather than six reference timepoints; withdrawn mechanism distributions omitted from final presentation.','Three existing diagnostics from the same 400 retained frozen records; KL endpoint fully retained.','Palette-base opacity gradients; each block has one user-approved base color.','Fixed bandwidth and shared height normalization within block; reflected boundary, no artificial peak shifting.'])

(HERE/'FIGURE_MANIFEST.json').write_text(json.dumps(MAN,ensure_ascii=False,indent=2))
