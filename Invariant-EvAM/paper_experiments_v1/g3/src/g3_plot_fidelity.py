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
A=G/'archive/before_detail_revision_20260924';A.mkdir(parents=True,exist_ok=True)
for rel in ['FINAL_FIGURES_01_07','figure_data','reports','configs/FIGURE_MANIFEST.json','DELIVERY_MANIFEST.json']:
 src=G/rel;dst=A/rel
 if not dst.exists():
  dst.parent.mkdir(parents=True,exist_ok=True)
  if src.is_dir():shutil.copytree(src,dst)
  else:shutil.copy2(src,dst)
S=pd.read_csv(G/'results/G3_summary.tsv',sep='\t');Z=pd.read_csv(G/'results/G3_sensitivity_regions.tsv',sep='\t');M=json.loads((G/'configs/FIGURE_MANIFEST.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
KEEP={str(p):sha(p) for i in [1,6,7] for p in [OUT/f'G3_{i:02d}.pdf',OUT/f'G3_{i:02d}.svg']}
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
# 02: retain correct-call endpoint while incorporating its complete outcome composition.
methods=['SHARED_K1','BOUNDED_BIOLOGY','FLEXIBLE_SELECTION','ORACLE_STRATIFIED'];short=['Shared','Bounded','Flexible','Known + strata'];conds=['Bio= / Obs=','Bio= / Obs≠','Bio≠ / Obs=','Bio≠ / Obs≠','Local alignment','Outside nuisance']
fig=figure(105,159);ax=fig.add_axes([.52,.14,.40,.72]);cx=fig.add_axes([.52,.915,.40,.018]);cmap=LinearSegmentedColormap.from_list('rate',['white',C[1]]);norm=Normalize(0,100);tab=[];ys=[];labels=[]
for cell in range(6):
 for j,method in enumerate(methods):
  r=S[(S.cell_id==cell)&(S.method_id==method)].iloc[0];n=int(r.n_tasks);fail=int(r.numerical_failures);unknown=n-int(r.called)-fail;counts=[int(r.correct_calls),int(r.wrong_calls),unknown,fail];assert sum(counts)==n
  y=cell*4+j+(.9 if cell>=4 else 0);ys.append(y);labels.append(f'{conds[cell]} · {short[j]}')
  for k,v in enumerate(counts):
   rate=100*v/n;ax.add_patch(Rectangle((k-.5,y-.5),1,1,fc=cmap(norm(rate)),ec='#555555',lw=.32));ax.text(k,y,f'{rate:g}',ha='center',va='center',fontsize=5.8,color='#222222')
  tab.append({**r.to_dict(),'display_row':y,'correct_percent':100*counts[0]/n,'wrong_percent':100*counts[1]/n,'unknown_percent':100*counts[2]/n,'failed_percent':100*counts[3]/n})
ax.set(xlim=(-.52,3.52),ylim=(24.5,-.6));ax.set_yticks(ys,labels);ax.set_xticks(range(4),['Correct','Wrong','Unknown','Failure'],rotation=48,ha='right');ax.tick_params(length=0,pad=2)
for sp in ax.spines.values():sp.set_visible(False)
for lo,hi,label in [(-.5,15.5,'D3'),(16.4,24.4,'D4')]:
 ax.plot([-4.55,-4.55],[lo,hi],c='#555555',lw=.5,clip_on=False);ax.text(-4.83,(lo+hi)/2,label,rotation=90,ha='center',va='center',fontsize=7)
cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cx,orientation='horizontal',ticks=[0,50,100]);cb.solids.set_rasterized(False);cx.set_title('Replicates (%)',fontsize=7,pad=5)
save(fig,2,pd.DataFrame(tab),'R20','Correct-call yield with full decision composition','Narrow dense rectangular matrix; 24 real condition-method rows, four outcome columns, two D3/D4 blocks; thin grid, top narrow horizontal key, small rotated labels. Correct-call endpoint retained as first column; full composition adds context. Numeric rates replace unsupported significance marks; 0–100% linear scale, user palette.')
# 03: thin side composition, compact rows, prominent reference-to-strategy connectors.
f=pd.read_csv(A/'figure_data/G3_03.tsv',sep='\t');f['interval_direction']=np.where(f.normal95_low>0,'Above zero',np.where(f.normal95_high<0,'Below zero','Includes zero'));fig=figure(125,113);side=fig.add_axes([.39,.25,.043,.62]);ax=fig.add_axes([.46,.25,.50,.62]);yl=[];yt=[]
for r in f.itertuples():
 y=r.display_row;delta=100*r.paired_difference;lo=100*r.normal95_low;hi=100*r.normal95_high;c={'Above zero':C[2],'Below zero':C[3],'Includes zero':C[1]}[r.interval_direction];yt.append(y)
 label={'RANDOM_X_0.1':'Random X · 10%','CALIBRATION_XS_0.1':'X + S · 10%','RANDOM_X_0.25':'Random X · 25%','CALIBRATION_XS_0.25':'X + S · 25%','KNOWN_OPERATOR_ORACLE':'Known operator'}[r.method];yl.append(r.model_label+' / '+label)
 ax.axhline(y,c='#ededed',lw=.35,zorder=0);ax.plot([lo,hi],[y,y],c='#dddddd',lw=3.3,solid_capstyle='round',zorder=1);ax.plot([0,delta],[y,y],c='#555555',lw=.75,zorder=2)
 ax.scatter(0,y,s=15,fc='white',ec='#666666',lw=.65,zorder=3);ax.scatter(delta,y,s=15,fc=c,ec=c,lw=.3,zorder=4)
 left=0
 for val,col in zip([r.ordinary_fraction,r.random_X_fraction,r.random_XS_fraction],[C[1],C[0],C[2]]):side.barh(y,val,left=left,height=.62,fc=col,ec='white',lw=.25);left+=val
for a in [side,ax]:a.set_ylim(10.8,-.85)
side.set(xlim=(0,1),xticks=[],yticks=yt,yticklabels=yl);side.tick_params(length=0,pad=2,labelsize=6.2)
for sp in side.spines.values():sp.set_visible(False)
ax.set(yticks=[],xlim=(-11,7),xticks=[-10,-5,0,5],xlabel='Correct-call change (pp)');ax.spines[['top','left','right']].set_visible(False);ax.set_title('Ordinary ○ — ● strategy',fontsize=7,pad=8);side.set_title('Budget',fontsize=6,pad=8)
fig.legend(handles=[Patch(fc=c,label=l) for c,l in zip([C[1],C[0],C[2]],['Ordinary','Random X','X + S'])],loc='lower left',bbox_to_anchor=(.02,.065),ncol=1,frameon=True,edgecolor='#cccccc',fancybox=False,fontsize=6,handlelength=1,borderpad=.4,labelspacing=.3)
fig.legend(handles=[Line2D([0],[0],marker='o',ls='none',ms=3,c=c,label=l) for c,l in [(C[2],'95% interval above 0'),(C[1],'95% interval includes 0'),(C[3],'95% interval below 0')]]+[Line2D([0],[0],c='#dddddd',lw=3,label='Paired 95% interval')],loc='lower right',bbox_to_anchor=(.975,.035),frameon=True,edgecolor='#cccccc',fancybox=False,fontsize=5.8,borderpad=.4,labelspacing=.3)
save(fig,3,f,'R25','Paired strategy effects with budget composition','Match thin left composition strips, compact row labels, pale thick uncertainty strokes behind thin gray connectors, small open/filled round points, two boxed lower legends. Filled colors classify paired 95% interval relative to zero, as the reference classifies expected/observed contrast; this is descriptive, not multiplicity-adjusted significance. Ten frozen rows cannot duplicate original ~50 rows; preserve order and effect sizes. Same reference zero and paired intervals, no artificial separation.')
# 04: match original three thin aligned columns and right-hand stacked legends.
d=pd.read_csv(A/'figure_data/G3_04.tsv',sep='\t');fig=figure(172,120);axes=[fig.add_axes(v) for v in [[.335,.20,.17,.68],[.535,.20,.17,.68],[.735,.20,.12,.68]]];labels=[];ys=[]
for k,r in enumerate(d.itertuples()):
 y=k;ys.append(y);base={'OMITTED_U':'Omitted','MORE_ENVS_SAME_BUDGET':'More env.','FLEXIBLE_U':'Flexible','CALIBRATED_FLEXIBLE_U':'Calibrated'}[r.method_id];cond={8:'U=0',9:'U=0.5',10:'U=1',11:'Exact alias'}[r.cell_id];labels.append(cond+' / '+base+(' †' if r.failed_count else ''))
 for ax,counts,colors in [(axes[0],[r.correct_calls,r.wrong_calls,r.unknown_count,r.failed_count],[C[2],C[4],C[1],C[5]]),(axes[1],[r.finite_count,r.unbounded_count,r.failed_count],[C[0],C[3],C[5]])]:
  left=0
  for v,c in zip(counts,colors):ax.barh(y,v/r.n_tasks,left=left,height=.96,fc=c,ec='white',lw=.3);left+=v/r.n_tasks
 if r.called:
  q=r.wrong_given_called*100;axes[2].errorbar(q,y,xerr=[[q-r.risk_low*100],[r.risk_high*100-q]],fmt='o',ms=2.1,c='#222222',elinewidth=.4,capsize=.7)
 else:axes[2].text(50,y,'NA',ha='center',va='center',fontsize=5.5,c='#777777')
for i,ax in enumerate(axes):
 ax.set_ylim(15.55,-.55);ax.set_yticks(ys,labels if i==0 else []);ax.tick_params(axis='y',length=0,pad=2,labelsize=6)
 if i<2:ax.set(xlim=(-.02,1.02),xticks=[0,.25,.5,.75,1],xlabel='Proportion');ax.set_xticklabels(['0.00','0.25','0.50','0.75','1.00'],rotation=90)
 else:ax.set(xlim=(-3,103),xticks=[0,25,50,75,100],xlabel='Risk (%)');ax.set_xticklabels(['0','25','50','75','100'],rotation=90);ax.grid(axis='both',color='#e8e8e8',lw=.35);ax.set_axisbelow(True)
 for sp in ax.spines.values():sp.set_linewidth(.5);sp.set_color('#555555')
 ax.set_title(['Decision','Interval','Wrong / called'][i],fontsize=6.7,pad=7)
for y in [3.5,7.5,11.5]:
 for ax in axes:ax.axhline(y,c='white',lw=.75,zorder=5)
legend1=fig.legend(handles=[Patch(fc=c,label=l) for c,l in zip([C[2],C[4],C[1],C[5]],['Correct','Wrong','Unknown','Failure'])],title='Decision',loc='upper left',bbox_to_anchor=(.866,.80),frameon=False,fontsize=6,title_fontsize=6.5,handlelength=.85,labelspacing=.55,borderaxespad=0)
legend2=fig.legend(handles=[Patch(fc=c,label=l) for c,l in zip([C[0],C[3],C[5]],['Finite','Unbounded','Failure'])],title='Interval',loc='upper left',bbox_to_anchor=(.866,.51),frameon=False,fontsize=6,title_fontsize=6.5,handlelength=.85,labelspacing=.55,borderaxespad=0)
fig.text(.335,.045,'† One numerical failure; NA: no sign calls.',fontsize=6)
save(fig,4,d,'R28','Full decision and interval composition with conditional risk','Rebuild original compact continuous row grid, two equally narrow composition columns and narrower point column, vertical quarter-proportion ticks, right-side two legends, tiny black points and light grid. Data-required additions: Wilson error bars, NA no-call markers and explicit failure footnote. All 16 rows preserved.')
# 05: actual three-point sensitivity trajectories; real endpoint insets replace inapplicable time AUCs.
curve_specs=[(3,[0,.2,.5],C[1],'Bio + obs\nε=0,0.2,0.5'),(10,[0],C[0],'Common U\nε=0'),(10,[.2,.5],C[2],'Common U\nε=0.2,0.5')]
fig=figure(122,124);ax=fig.add_axes([.17,.16,.77,.56]);insets=[fig.add_axes([.25,.77,.26,.13]),fig.add_axes([.66,.77,.26,.13])];tab=[];inrows=[]
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
 rows=[t for t in inrows if t['endpoint']==['wrong','unknown'][k]]
 for j,r in enumerate(rows):
  val=r['rate']*100;a.barh(j,val,height=.55,fc=r['color'],ec='#333333',lw=.5);a.errorbar(val,j,xerr=np.maximum(0,[[val-r['low']*100],[r['high']*100-val]]),fmt='none',ecolor='#222222',elinewidth=.5,capsize=.9)
 a.set(xlim=(0,106),ylim=(5.6,-.6),xticks=[0,50,100]);a.set_yticks(range(6),[('B' if t['cell']==3 else 'U')+f":{t['epsilon']:g}" for t in rows] if k==0 else []);a.tick_params(labelsize=5.1,length=1.5,pad=1);a.spines[['top','right']].set_visible(False);a.set_title(['Wrong at η=0.5','Unknown at η=0.5'][k],fontsize=6.5,pad=4)
ax.set(xlim=(-.025,1.025),ylim=(-3,105),xticks=[0,.5,1],yticks=[0,20,40,60,80,100],xlabel='Allowed selection coefficient η',ylabel='Wrong sign calls (%)');ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=7)
fig.legend(handles=[Patch(fc=c,ec='#333333',lw=.6,label=label) for _,_,c,label in curve_specs],loc='upper center',bbox_to_anchor=(.56,.99),ncol=3,frameon=False,fontsize=6.2,handlelength=1.4,handletextpad=.35,columnspacing=.9)
pd.DataFrame(inrows).to_csv(DATA/'G3_05_insets.tsv',sep='\t',index=False)
save(fig,5,pd.DataFrame(tab),'R04','Sensitivity paths and actual midpoint wrong/unknown rates','Rebuild horizontal colored-box legend, solid thick lines, black-edged circular markers, dark capped errors and two top horizontal summaries. Insets use real η=0.5 wrong/unknown rates, not invented time AUCs; all 20 repeats/curve. No fictional phase shading. Main trajectories merge only identical wrong counts; six separate configurations retained in each inset, without pooling N.')
assert all(sha(Path(k))==v for k,v in KEEP.items()),'Accepted figures changed'
(G/'configs/FIGURE_MANIFEST.json').write_text(json.dumps(M,ensure_ascii=False,indent=2))
(G/'runs/FIDELITY_PRESERVATION.json').write_text(json.dumps({'unchanged_accepted_files':KEEP,'changed_panels':[2,3,4,5],'review_status':'awaiting_visual_review'},indent=2))
print('Rebuilt panels 02–05; 01/06/07 bytes preserved')
