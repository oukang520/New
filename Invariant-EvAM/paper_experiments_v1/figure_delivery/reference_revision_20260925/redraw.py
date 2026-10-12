# -*- coding: utf-8 -*-
raise SystemExit('Historical renderer retired. Use selection_cleanup_20260926/redraw_g3.py for the selected G3_06; do not restore withdrawn figures.')
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
 f.savefig(PRE/(name+'.png'),dpi=180)
 rr=next(x for x in CAT if x['id']==ref)
 MAN.append(dict(panel_id=name,primary_reference_id=ref,reference_file=rr['source_file_absolute'],reference_sha256=sha(Path(rr['source_file_absolute'])),source=source,plot_data=str(dp.relative_to(P)),plot_data_sha256=sha(dp),components=components,adaptations=adapt,palette=C,size_mm=(f.get_size_inches()*25.4).tolist(),dpi=600,review_status='awaiting_actual_visual_review',user_approved=False,outputs=[str((target/(name+'.'+e)).relative_to(P)) for e in ['pdf','svg']]))
 plt.close(f)
# R02: retain all four natural model families and the original fixed display subset.
d,src=read('g5/figure_data/G5_02_compatibility_information.tsv');assert (d[['training_deviance','target_information']]>0).all().all()
order=['IDENTITY_SHARED','BURDEN_SELECTION_SHARED','PRESPECIFIED_GENE_SELECTION','BOUNDED_BIOLOGY'];labels=['Identity','Burden','Gene selection','Bounded biology'];colors=[C[0],C[1],C[5],C[2]]
x=np.log10(d.training_deviance.to_numpy());y=np.log10(d.target_information.to_numpy());slope,intercept=np.polyfit(x,y,1);d['log10_residual']=y-(intercept+slope*x)
f=plt.figure(figsize=(142/25.4,152/25.4));a=f.add_axes([.16,.225,.66,.617]);top=f.add_axes([.16,.86,.66,.115]);right=f.add_axes([.84,.225,.115,.617])
# Exact log-coordinate limits keep the marginals aligned with the square plotting field.
xlim=(75,290);ylim=(1.25,55);xx=np.linspace(np.log10(xlim[0]),np.log10(xlim[1]),350);yy=np.linspace(np.log10(ylim[0]),np.log10(ylim[1]),350)
for m,l,c in zip(order,labels,colors):
 q=d[d.spec_id.eq(m)];shown=q[q.displayed_point.eq(True)]
 a.scatter(shown.training_deviance,shown.target_information,s=12,fc=c,ec='#333333',lw=.25,alpha=.6,label=f'{l} (N=200)',zorder=3)
 # Summary retained from original; not confidence bars or significance tests.
 a.errorbar(q.training_deviance.mean(),q.target_information.mean(),xerr=q.training_deviance.std(),yerr=q.target_information.std(),fmt='D',ms=3.4,mfc=c,mec='#222222',mew=.5,ecolor=c,elinewidth=.65,capsize=1.6,zorder=4)
 dx=gaussian_kde(np.log10(q.training_deviance))(xx);dy=gaussian_kde(np.log10(q.target_information))(yy)
 top.fill_between(10**xx,0,dx,fc=c,alpha=.24);top.plot(10**xx,dx,c=c,lw=.65)
 right.fill_betweenx(10**yy,0,dy,fc=c,alpha=.24);right.plot(dy,10**yy,c=c,lw=.65)
a.set(xscale='log',yscale='log',xlim=xlim,ylim=ylim,xlabel='Training saturated deviance',ylabel='Effective information for TP53 → KRAS')
a.plot(10**xx,10**(intercept+slope*xx),c=C[4],lw=.9,ls=(0,(3,2)),zorder=2)
a.xaxis.set_major_locator(FixedLocator([100,200]));a.yaxis.set_major_locator(FixedLocator([2,5,10,20,50]))
fmt=FuncFormatter(lambda v,p:f'{v:g}');a.xaxis.set_major_formatter(fmt);a.yaxis.set_major_formatter(fmt);a.tick_params(which='minor',labelbottom=False,labelleft=False)
a.grid(which='major',ls=(0,(2,2)),lw=.55,c='#bdbdbd');a.tick_params(which='both',direction='in',top=True,right=True)
a.legend(loc='upper left',fontsize=5.9,frameon=True,fancybox=False,edgecolor='#b0b0b0',handlelength=1,handletextpad=.4,borderpad=.45,labelspacing=.32)
for ax in [top,right]:
 ax.spines[['left','top','right']].set_visible(False);ax.tick_params(which='both',length=0,labelbottom=False,labelleft=False);ax.set_xticks([]);ax.set_yticks([])
top.set(xscale='log',xlim=xlim,ylim=(0,None));right.set(yscale='log',ylim=ylim,xlim=(0,None));right.spines['left'].set_visible(True);right.spines['bottom'].set_visible(False)
ins=a.inset_axes([.49,.06,.48,.265]);ins.set_facecolor('white');res=[d[d.spec_id.eq(m)].log10_residual.to_numpy() for m in order]
bp=ins.boxplot(res,vert=False,positions=[3,2,1,0],widths=.5,patch_artist=True,showfliers=True,flierprops=dict(marker='.',ms=1.3,mec='#555555',alpha=.5),medianprops=dict(c='#222222',lw=.7),whiskerprops=dict(c='#555555',lw=.55),capprops=dict(c='#555555',lw=.55))
for box,c in zip(bp['boxes'],colors):box.set(fc=c,alpha=.6,ec='#333333',lw=.55)
ins.axvline(0,lw=.5,ls=':',c='#666666');ins.set_yticks([]);ins.set_ylim(-.6,3.7);ins.set_title('Residuals (log10)',fontsize=6.6,pad=3);ins.tick_params(labelsize=5.7,direction='in',length=2);ins.set_xticks([-.4,0,.4]);ins.set_xlim(-.9,.6)
for sp in ins.spines.values():sp.set_visible(False)
ins.patch.set_visible(False)
ins.add_patch(FancyBboxPatch((0,0),1,1,boxstyle='round,pad=0.018,rounding_size=0.06',transform=ins.transAxes,fc='white',ec='#888888',lw=.6,clip_on=False,zorder=-1))
f.text(.16,.12,'20 / 200 bootstrap points per family; diamonds: mean ± SD from all 200.',fontsize=6.1)
f.text(.16,.091,'Marginals and residual boxes use all 800 records; dashed line: descriptive pooled fit.',fontsize=6.1)
f.text(.16,.062,'All four families fail the original compatibility test (p = 0.005).',fontsize=6.1)
f.text(.16,.033,'Positive information is local; bootstrap replicates are not independent patients.',fontsize=6.1)
save(f,'G5_02_compatibility_information','R02',d,src,['square log-log scatter','aligned top and right filled marginal KDE','four-family N legend','dashed descriptive fit','lower-right horizontal residual boxes','original mean/SD crossbars'],['Four model families rather than three; palette follows user six colors.','Only original fixed 20/200 points shown per family; densities/boxes all records.','OLS in log coordinates is descriptive; no bootstrap-as-patient residual significance.','Zero-information line cannot appear on log axis; positivity and local limitation stated.'])
# R08: same four conditions in each of three diagnostic blocks.
d,src=read('g3/figure_data/G3_06.tsv');order=['time','higher_order','mixture','measurement'];labs=['Gamma time','Higher order','Mixture','Measurement']
f=plt.figure(figsize=(136/25.4,168/25.4));dens=[]
for block,(metric,label,c) in enumerate([('display_KL_x1000','Predictive KL (×0.001)',C[2]),('deviance','Training saturated deviance',C[5]),('target_effective_information','Effective target information',C[0])]):
 a=f.add_axes([.24,.725-block*.275,.71,.225]);values=d[metric].to_numpy();band=1.06*values.std(ddof=1)*len(values)**(-.2);hi=values.max()+3*band;grid=np.linspace(0,hi,700)
 density=[]
 for m in order:
  v=d[d.mechanism.eq(m)][metric].to_numpy();assert len(v)==200 and np.isfinite(v).all()
  # Boundary-reflected fixed Gaussian kernel; common bandwidth/height scale within each block.
  den=(np.exp(-.5*((grid[:,None]-v)/band)**2)+np.exp(-.5*((grid[:,None]+v)/band)**2)).mean(axis=1)/(band*np.sqrt(2*np.pi));density.append(den)
 peak=max(z.max() for z in density)
 for j,(m,l,z) in enumerate(zip(order,labs,density)):
  base=3-j;alpha=[.40,.58,.78,1][j];a.plot([0,hi],[base,base],c=c,alpha=alpha,lw=.7,zorder=j*3)
  a.fill_between(grid,base,base+z/peak*1.5,color=c,alpha=alpha,zorder=j*3+1);a.plot(grid,base+z/peak*1.5,c='white',lw=.6,zorder=j*3+2)
  a.text(-.035,base+.04,l,transform=a.get_yaxis_transform(),ha='right',va='bottom',fontsize=7,weight='bold',color=c,alpha=max(.65,alpha))
  dens.extend(dict(block=block,metric=metric,mechanism=m,x=float(x),density=float(y),bandwidth=float(band),n=200) for x,y in zip(grid,z))
 a.set(xlim=(0,hi),ylim=(-.08,4.6),xlabel=label);a.set_yticks([]);a.spines[['left','top','right']].set_visible(False);a.spines['bottom'].set_visible(False);a.tick_params(axis='x',length=2,pad=2);a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5));a.xaxis.labelpad=2
f.text(.24,.047,'200 datasets per row; all four mechanisms retained in every block.',fontsize=6.2)
f.text(.24,.025,'Common bandwidth within each block; reflected KDE; density is not uncertainty.',fontsize=6.2)
pd.DataFrame(dens).to_csv(DATA/'G3_06_density.tsv',sep='\t',index=False)
save(f,'G3_06','R08',d,src,['three vertically stacked ridgeline blocks','colored direct row labels','thin full-width baselines','white ridge outlines','light-to-dark fill by row','one labeled x axis per block, no density ticks'],['Four actual mechanism groups rather than six timepoints.','Three existing diagnostics from the same 800 frozen records; KL endpoint fully retained.','Palette-base opacity gradients; each block has one user-approved base color.','Fixed bandwidth and shared height normalization within block; reflected boundary, no artificial peak shifting.'])
# R12: all twelve independent simulation conditions, including exact overlapping zeros.
d,src=read('g2/figure_revision_v3/data/G2_v3_03_paired_gain_scatter.tsv');f=plt.figure(figsize=(138/25.4,152/25.4));a=f.add_axes([.16,.205,.66,.59]);top=f.add_axes([.16,.81,.66,.16]);cax=f.add_axes([.845,.205,.026,.59]);xlim=(-10,105)
edges=np.linspace(-10,105,13);top.hist(d.x,bins=edges,color=C[5],alpha=.6,ec='white',lw=.3);tw=top.twinx();grid=np.linspace(*xlim,500);den=gaussian_kde(d.x)(grid);tw.plot(grid,den,c='#333333',lw=.85);tw.set_ylabel('Density',fontsize=6.5,labelpad=3);tw.tick_params(labelsize=5.6,direction='in',length=2);tw.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3));top.set_ylabel('Conditions',fontsize=6.5,labelpad=4);top.set(xlim=xlim);top.tick_params(labelbottom=False,direction='in',top=True,labelsize=6,length=2);top.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3,integer=True));top.text(.02,.87,'a',transform=top.transAxes,weight='bold')
methods=['strong','weak','nonidentifying'];colors=[C[0],C[3],C[5]];markers=['o','s','D']
for m,c,mk in zip(methods,colors,markers):
 q=d[d.design.eq(m)]
 a.errorbar(q.x,q.y,xerr=[q.x-q.x_low,q.x_high-q.x],yerr=[q.y-q.y_low,q.y_high-q.y],fmt='none',ecolor='#888888',elinewidth=.65,capsize=1.5,zorder=2)
 a.scatter(q.x,q.y,s=18,marker=mk,facecolors='white',edgecolors=c,linewidths=.8,zorder=3)
 for row in q.itertuples():
  if row.x!=0 or row.y!=0:a.annotate(f'{row.truth:+g}',(row.x,row.y),xytext=(4,4),textcoords='offset points',fontsize=5.8,color='#555555')
zero=d[d.x.eq(0)&d.y.eq(0)];a.annotate(f'{len(zero)} conditions at (0, 0)',(0,0),xytext=(27,3),textcoords='data',fontsize=6.2,arrowprops=dict(arrowstyle='-',lw=.55,color='#555555'))
a.axhline(0,ls=':',c='#777777',lw=.65);a.axvline(0,ls=':',c='#bbbbbb',lw=.55);a.plot(xlim,xlim,c='#cccccc',ls=(0,(3,3)),lw=.55,zorder=0)
a.set(xlim=xlim,ylim=(-7,103),xlabel='Correct-call gain over B5 (pp)',ylabel='Correct-call gain over B4 (pp)');a.tick_params(direction='in',top=True,right=True,which='both');a.xaxis.set_minor_locator(AutoMinorLocator(4));a.yaxis.set_minor_locator(AutoMinorLocator(4));a.text(.02,.94,'b',transform=a.transAxes,weight='bold')
cb=matplotlib.colorbar.ColorbarBase(cax,cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(4)-.5,3),boundaries=np.arange(4)-.5,ticks=range(3));cb.ax.set_yticklabels(['Complementary','Repeated family','Separate panels'],rotation=90,va='center',fontsize=6.5);cb.ax.tick_params(length=0,pad=4);cb.outline.set_linewidth(.5)
f.text(.16,.13,'12 conditions; 500 paired datasets per condition; labels: true interaction.',fontsize=6.1)
f.text(.16,.102,'Crossbars: paired 95% Monte Carlo intervals. All six zero gains retained.',fontsize=6.1)
f.text(.16,.074,'Top: condition counts and descriptive KDE; no additional replicate pooling.',fontsize=6.1)
f.text(.16,.046,'Linear axes retain zero / negative gains; faint diagonal denotes equal gains.',fontsize=6.1)
save(f,'G2_03','R12',d,src,['top histogram and black density with two labeled y axes','shared x limits','bottom scatter with both-axis uncertainty','closed four-sided frame with minor ticks','right discrete category color strip','a/b letters and horizontal reference'],['Linear signed axes needed for negative/zero gains, not reference astrophysical scale.','12 conditions (6 exactly at origin); counts include every condition, no jitter or invented points.','Three actual design categories and user palette replace spectral classes.','Both uncertainties are original paired Monte Carlo intervals, not new model error bars.'])
(HERE/'FIGURE_MANIFEST.json').write_text(json.dumps(MAN,ensure_ascii=False,indent=2));print('Replaced exactly three final PDF/SVG pairs')
