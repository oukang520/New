from pathlib import Path
import sys,json,hashlib,shutil
R=Path(__file__).resolve().parents[3];G=R/'paper_experiments_v1/g2';V=G/'figure_revision_v2'
sys.path.insert(0,str(R/'src'))
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm,Normalize
from invariant_evam.figure_style import apply_publication_style
apply_publication_style({'font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none','pdf.fonttype':42})
P=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7']
S=pd.read_csv(G/'results/G2_core_summary.tsv',sep='\t');S=S[S.method_id.isin(['B1','B2','B3','B4','B5','M'])]
D=['strong','weak','nonidentifying'];DN=['Complementary','Repeated family','Separate panels'];M=['B1','B2','B3','B4','B5','M'];T=[-.8,0,.15,.8]
(V/'previews').mkdir(exist_ok=True);(V/'data').mkdir(exist_ok=True);(V/'figures').mkdir(exist_ok=True);manifest=[]
def save(fig,name,data,ref,endpoint,adapt):
 for ext in ['pdf','svg']:fig.savefig(V/'figures'/f'{name}.{ext}',dpi=600)
 fig.savefig(V/'previews'/f'{name}.png',dpi=150)
 data.to_csv(V/'data'/f'{name}.tsv',sep='\t',index=False)
 manifest.append(dict(panel_id=name,primary_reference_id=ref,endpoint=endpoint,adaptations=adapt,size_mm=(fig.get_size_inches()*25.4).tolist(),dpi=600,palette=P,review_status='awaiting_visual_review',source_data=str(V/'data'/f'{name}.tsv'),outputs=[str(V/'figures'/f'{name}.{e}') for e in ['pdf','svg']],failure_handling='All conditions retained; failures distinct from unknown; no-call conditional risk undefined',N='500 paired replicates/cell in core; 200/cell in independent B6 endpoint',data_sha256=hashlib.sha256((V/'data'/f'{name}.tsv').read_bytes()).hexdigest()))
 plt.close(fig)
def strip(ax,y0,n,text,x=-1.8,width=.38):
 ax.add_patch(Rectangle((x,y0-.5),width,n,facecolor='#eeeeee',edgecolor='none',clip_on=False))
 ax.text(x+width/2,y0+(n-1)/2,text,rotation=90,ha='center',va='center',fontsize=8,clip_on=False)
def get(d,t,m):return S[(S.design==d)&(S.truth==t)&(S.method_id==m)].iloc[0]
# Reference R24: one data matrix with three row blocks, area and color keys.
fig,ax=plt.subplots(figsize=(150/25.4,155/25.4));fig.subplots_adjust(left=.27,right=.72,bottom=.13,top=.93)
norm=Normalize(0,1);cmap=LinearSegmentedColormap.from_list('risk',['#ffffff',P[4]])
rows=[]
for di,d in enumerate(D):
 for ti,t in enumerate(T):
  y=di*5+ti
  for j,m in enumerate(M):
   r=get(d,t,m);risk=r.wrong_sign_among_called
   if r.called:ax.scatter(j,y,s=270*r.call_rate,c=[risk],cmap=cmap,norm=norm,edgecolors='#555555',linewidths=.5)
   else:ax.plot(j,y,marker='x',color='#999999',ms=3,mew=.65)
   rows.append({**r.to_dict(),'matrix_row':y,'bubble_area_pt2':270*r.call_rate,'color_conditional_error':risk})
  ax.text(-.65,y,f'{t:+.2g}',ha='right',va='center')
 ax.add_patch(Rectangle((-.5,di*5-.5),6,4,fill=False,ec='#888888',lw=.6));strip(ax,di*5,4,DN[di],x=-1.8)
ax.set(xlim=(-.55,5.55),ylim=(13.65,-.75),xticks=range(6),xticklabels=M,yticks=[]);ax.set_xlabel('Inference method');ax.set_title('Answerability and conditional error',loc='left',pad=14)
for sp in ax.spines.values():sp.set_visible(False)
ax.tick_params(length=0)
cax=fig.add_axes([.79,.60,.028,.23]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax);cb.set_ticks([0,.5,1]);cax.set_title('Wrong / called',fontsize=7,pad=9)
handles=[ax.scatter([],[],s=270*v,facecolor='white',edgecolor='#555555',lw=.5,label=f'{v:.0%}') for v in [.1,.5,1]]
fig.legend(handles=handles,title='Called / all',loc='upper left',bbox_to_anchor=(.755,.48),frameon=False,labelspacing=1.4,fontsize=7,title_fontsize=7)
fig.text(.77,.20,'×  No calls\nRisk undefined',fontsize=7,linespacing=1.6)
fig.text(.27,.025,'500 replicates per cell; all 12 conditions retained.',fontsize=7)
save(fig,'G2_v2_01_answerability_matrix',pd.DataFrame(rows),'R24','Area: call rate; color: conditional sign error','Three framed row blocks, grey group strips, independent size and color legends; no-call crosses replace undefined colors.')
# Reference R20: all coverage results, with transparent numeric failure annotation.
fig,ax=plt.subplots(figsize=(145/25.4,157/25.4));fig.subplots_adjust(left=.29,right=.94,bottom=.12,top=.80)
cmap=LinearSegmentedColormap.from_list('coverage',[P[4],'#ffffff',P[0]]);norm=TwoSlopeNorm(vmin=-.25,vcenter=0,vmax=.25)
rows=[]
for di,d in enumerate(D):
 for ti,t in enumerate(T):
  y=di*5+ti
  for j,m in enumerate(M):
   r=get(d,t,m);gap=r.conditional_coverage-.95
   ax.add_patch(Rectangle((j-.5,y-.5),1,1,facecolor=cmap(norm(gap)),edgecolor='#aaaaaa',lw=.45))
   txt=f'{100*r.conditional_coverage:.1f}'
   if r.solver_failures:txt+=f'\nf={int(r.solver_failures)}'
   ax.text(j,y,txt,ha='center',va='center',fontsize=7,color='white' if abs(gap)>.17 else '#222222')
   rows.append({**r.to_dict(),'coverage_gap':gap})
  ax.text(-.65,y,f'{t:+.2g}',ha='right',va='center')
 strip(ax,di*5,4,DN[di],x=-1.7)
ax.set(xlim=(-.5,5.5),ylim=(13.6,-.6),xticks=range(6),xticklabels=M,yticks=[]);ax.tick_params(length=0);ax.set_xlabel('Inference method')
for sp in ax.spines.values():sp.set_visible(False)
cax=fig.add_axes([.35,.875,.53,.023]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal');cb.set_ticks([-.25,0,.25]);cb.set_ticklabels(['−25','0','+25']);cax.set_title('Coverage − nominal 95% (percentage points)',fontsize=8,pad=9)
fig.text(.29,.025,'Cells: coverage (%) among completed intervals.\nf: numerical failures / 500 planned replicates.',fontsize=7,linespacing=1.5)
save(fig,'G2_v2_02_coverage_matrix',pd.DataFrame(rows),'R20','Conditional coverage relative to nominal 95%','Top horizontal diverging color key with symmetric ±25 percentage-point limits; fine cell grid, group gaps, grey group strips. Coverage counts and end-to-end coverage retained in source table.')
# Reference R25: long dumbbells, left composition bars, direct paired differences.
Q=pd.read_csv(G/'results/G2_paired_comparisons.tsv',sep='\t');fig,ax=plt.subplots(figsize=(183/25.4,220/25.4));fig.subplots_adjust(left=.30,right=.72,bottom=.16,top=.91)
rows=[];ys=[];labs=[]
for di,d in enumerate(D):
 for ti,t in enumerate(T):
  r=get(d,t,'M');base=di*9+ti*2
  for k,b in enumerate(['B4','B5']):
   y=base+k;v=get(d,t,b);p=Q[(Q.cell_id==r.cell_id)&(Q.comparison=='M_vs_'+b)&(Q.metric=='correct_call')].iloc[0]
   ax.axhline(y,color='#eeeeee',lw=.5,zorder=0)
   ax.plot([v.correct_call_fraction,r.correct_call_fraction],[y,y],color='#777777',lw=.85)
   ax.scatter(v.correct_call_fraction,y,s=21,facecolor='white',edgecolor='#555555',lw=.8,zorder=3)
   ax.scatter(r.correct_call_fraction,y,s=22,facecolor=P[0],edgecolor='white',lw=.3,zorder=4)
   vals=[r.correct_calls,r.wrong_calls,r.scientific_unknown_count,r.solver_failures];assert sum(vals)==500
   left=-.205
   for value,c in zip(vals,[P[0],P[4],'#dddddd',P[5]]):
    ax.add_patch(Rectangle((left,y-.23),.15*value/500,.46,fc=c,ec='none',clip_on=False));left+=.15*value/500
   ax.text(-.235,y,f'{t:+.2g}  /  {b}',ha='right',va='center',fontsize=7.5,clip_on=False)
   ax.text(1.055,y,f'{100*p.paired_difference:+.1f}  [{100*p.paired_normal95_low:+.1f}, {100*p.paired_normal95_high:+.1f}]',va='center',fontsize=7,clip_on=False)
   rows.append({**r.to_dict(),'comparator':b,'comparator_correct_fraction':v.correct_call_fraction,**p.to_dict()})
 strip(ax,di*9,8,DN[di],x=-.61,width=.07)
ax.set(xlim=(-.015,1.015),ylim=(26,-1.0),yticks=[],xticks=[0,.25,.5,.75,1],xticklabels=['0','25','50','75','100']);ax.set_xlabel('Correct sign calls / all replicates (%)',labelpad=9)
ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
ax.text(-.135,-1.25,'M status',ha='center',fontsize=7,clip_on=False);ax.text(1.055,-1.25,'Δ percentage points [95% MC interval]',fontsize=7,clip_on=False)
fig.legend(handles=[Line2D([],[],marker='o',mfc='white',mec='#555555',ls='none',label='Comparator'),Line2D([],[],marker='o',mfc=P[0],mec='white',ls='none',label='M')],loc='upper center',bbox_to_anchor=(.52,.985),ncol=2,frameon=False)
fig.legend(handles=[Patch(fc=c,label=l) for c,l in zip([P[0],P[4],'#dddddd',P[5]],['Correct','Wrong','Unknown','Failure'])],loc='lower center',bbox_to_anchor=(.51,.062),ncol=4,frameon=False,fontsize=7)
fig.text(.10,.025,'Each row: 500 paired replicates. Fixed design/effect order; no advantage-based selection.\nZero-width MC intervals mean no observed discordance, not proven equality.',fontsize=7,linespacing=1.5)
save(fig,'G2_v2_03_paired_utility',pd.DataFrame(rows),'R25','Correct call fraction M versus B4/B5 with paired MC uncertainty','Long row dumbbells, hollow/filled markers, pale row guides, actual left status composition strips; numerical difference column exposes small effects without moving points. CI is paired normal Monte Carlo, not parameter CI.')
# Separate endpoint: no pooling with core simulation.
B=pd.read_csv(G/'results/B6_matched_summary.tsv',sep='\t');fig,ax=plt.subplots(figsize=(155/25.4,110/25.4));fig.subplots_adjust(left=.32,right=.91,bottom=.23,top=.82);rows=[]
for ei,e in enumerate(['coverage','correct_calls']):
 for ti,t in enumerate(T):
  y=ei*5+ti;a=B[(B.truth==t)&(B.endpoint==e)&(B.method_id=='M_MATCHED')].iloc[0];b=B[(B.truth==t)&(B.endpoint==e)&(B.method_id=='B6_OFFICIAL')].iloc[0]
  ax.axhline(y,color='#eeeeee',lw=.6);ax.plot([a.value,b.value],[y,y],color='#777777',lw=1)
  ax.plot([a.wilson_low,a.wilson_high],[y,y],color='#dddddd',lw=5,solid_capstyle='round',zorder=1)
  ax.scatter(b.value,y,s=27,facecolor='white',edgecolor='#555555',zorder=3);ax.scatter(a.value,y,s=27,color=P[0],edgecolor='white',lw=.4,zorder=4)
  ax.text(-.06,y,f'{t:+.2g}',ha='right',va='center');ax.text(1.045,y,f'{100*(a.value-b.value):+.1f}',va='center',fontsize=7,clip_on=False)
  rows.append(dict(truth=t,endpoint=e,M_value=a.value,B6_value=b.value,M_low=a.wilson_low,M_high=a.wilson_high,B6_low=b.wilson_low,B6_high=b.wilson_high,n=200,difference=a.value-b.value))
 strip(ax,ei*5,4,['Coverage','Correct calls'][ei],x=-.29,width=.07)
ax.set(xlim=(-.01,1.015),ylim=(9,-1),yticks=[],xticks=[0,.25,.5,.75,1],xticklabels=['0','25','50','75','100']);ax.set_xlabel('Replicates (%)');ax.spines[['top','right','left']].set_visible(False)
ax.text(1.04,-.85,'Δ pp',fontsize=7,clip_on=False)
fig.legend(handles=[Line2D([],[],marker='o',mfc='white',mec='#555555',ls='none',label='Official + Wald'),Line2D([],[],marker='o',mfc=P[0],mec='white',ls='none',label='M + profile')],loc='upper center',bbox_to_anchor=(.55,.97),ncol=2,frameon=False,fontsize=8)
fig.text(.13,.035,'Independent matched endpoint; 200 replicates per effect, no solver failures.\nGrey ranges: M Wilson 95% Monte Carlo intervals. Δ: M − official.\nPoint estimates agree (maximum gap 1.33e−9).',fontsize=7,linespacing=1.5)
save(fig,'G2_v2_04_official_matched',pd.DataFrame(rows),'R25','Matched official endpoint: coverage and correct-call fractions','Shared probability scale; two endpoint row blocks; hollow/filled paired dots, pale MC ranges. Omit composition strip because this summary does not define a shared status partition across endpoints.')
# R01 grouped bars: fixed four-effect mixture; no cherry-picked cell selection.
def wilson(k,n):
 z=1.95996398454;p=k/n;mid=(p+z*z/(2*n))/(1+z*z/n);h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return mid-h,mid+h
fig,ax=plt.subplots(figsize=(132/25.4,108/25.4));fig.subplots_adjust(left=.16,right=.96,bottom=.26,top=.81);rows=[]
for j,(m,c) in enumerate(zip(['B4','B5','M'],[P[2],P[1],P[0]])):
 for di,d in enumerate(D):
  f=S[(S.design==d)&(S.method_id==m)];k=int(f.correct_calls.sum());n=int(f.n_tasks.sum());v=k/n;lo,hi=wilson(k,n);x=di+(j-1)*.23
  ax.bar(x,100*v,width=.20,color=c,edgecolor='#333333',lw=.65,label=m if di==0 else None,zorder=2)
  ax.errorbar(x,100*v,yerr=[[100*(v-lo)],[100*(hi-v)]],fmt='none',ecolor='#222222',elinewidth=.8,capsize=2,zorder=3)
  ax.text(x,100*hi+1.6,f'{100*v:.1f}',ha='center',va='bottom',fontsize=7)
  rows.append(dict(design=d,method=m,correct_calls=k,n=n,rate=v,low=lo,high=hi,failures=int(f.solver_failures.sum()),effect_mixture='equal weights: -0.8, 0, 0.15, 0.8'))
ax.set(xticks=range(3),xticklabels=['Complementary','Repeated\nfamily','Separate\npanels'],ylim=(0,65),ylabel='Correct sign calls / all replicates (%)');ax.spines[['top','right']].set_visible(False);ax.legend(ncol=3,loc='lower left',bbox_to_anchor=(0,1.04),frameon=False)
fig.text(.16,.045,'Fixed equal mixture of all four effects; 2,000 replicates per bar.\nWhiskers: Wilson 95% Monte Carlo intervals. Failures remain in denominator.',fontsize=7,linespacing=1.5)
save(fig,'G2_v2_05_grouped_bars',pd.DataFrame(rows),'R01','Correct-call utility across fixed equal mixtures','Narrow grouped bars, dark outlines and capped errors, upper horizontal legend, zero-based axis. Binary replicate dots omitted: 2000 overlapping 0/1 marks obscure rate endpoint. All conditions included; M/B4/B5 shown, all six methods in matrix.')
# R04 line geometry: actual confidence levels, filled dark-edge dots and dark error bars.
F=pd.read_csv(G/'results/G2_risk_answerability.tsv',sep='\t');fig,ax=plt.subplots(figsize=(137/25.4,112/25.4));fig.subplots_adjust(left=.16,right=.96,bottom=.22,top=.77)
for d,m,c,ls in [('strong','M',P[0],'-'),('weak','M',P[1],'--'),('strong','B4',P[2],'-'),('weak','B4',P[3],'--')]:
 f=F[(F.design==d)&(F.method_id==m)].sort_values('confidence');x=f.confidence.to_numpy()*100;y=f.wrong_sign_among_called.to_numpy()*100
 ax.plot(x,y,color=c,lw=1.6,ls=ls,label=f'{m} · '+('complementary' if d=='strong' else 'repeated'))
 ax.errorbar(x,y,yerr=np.maximum(0,np.array([y-100*f.wrong_sign_wilson_low,100*f.wrong_sign_wilson_high-y])),fmt='o',mfc=c,mec='#222222',ms=4.5,mew=.8,ecolor='#222222',elinewidth=.65,capsize=2,zorder=3)
ax.set(xlim=(79,100),ylim=(0,float(F.wrong_sign_wilson_high.max())*110),xticks=[80,90,95,97.5,99],xlabel='Candidate confidence level (%)',ylabel='Wrong signs / called replicates (%)');ax.spines[['top','right']].set_visible(False);ax.legend(ncol=2,loc='lower left',bbox_to_anchor=(-.05,1.06),fontsize=7,frameon=False)
fig.text(.16,.035,'All five frozen confidence levels; Wilson 95% Monte Carlo intervals.\nSeparate panels: zero calls at every level; conditional error undefined.',fontsize=7,linespacing=1.5)
save(fig,'G2_v2_06_risk_lines',F,'R04','Conditional error over all frozen confidence levels','Filled colored circles with dark outline, colored lines, black capped intervals and top horizontal legend. No phase shading or inset AUC: neither has statistical meaning here. Unequal actual confidence spacing retained.')
# R23 raw grouped scatter, known equality line, actual sample means and SD crossbars.
Braw=pd.concat([pd.read_csv(G/f'results/B6_matched_cell{i}.tsv',sep='\t') for i in range(4)])
wide=Braw.pivot(index=['cell_id','replicate_id','truth'],columns='method_id',values='estimate').reset_index()
fig,ax=plt.subplots(figsize=(120/25.4,118/25.4));fig.subplots_adjust(left=.19,right=.95,bottom=.23,top=.88)
for t,c in zip(T,P):
 f=wide[wide.truth==t];x=f.B6_OFFICIAL.to_numpy();y=f.M_MATCHED.to_numpy();ax.scatter(x,y,s=8,color=c,alpha=.55,lw=0,label=f'Truth {t:+.2g}')
 ax.errorbar(x.mean(),y.mean(),xerr=x.std(ddof=1),yerr=y.std(ddof=1),fmt='D',mfc='#222222',mec='#222222',ms=4,ecolor='#222222',elinewidth=.7,capsize=2,zorder=5)
lo=float(wide[['B6_OFFICIAL','M_MATCHED']].min().min())-.1;hi=float(wide[['B6_OFFICIAL','M_MATCHED']].max().max())+.1
ax.plot([lo,hi],[lo,hi],ls=(0,(5,4)),color='#555555',lw=.9,label='Equality');ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Official point estimate',ylabel='M point estimate');ax.set_aspect('equal');ax.tick_params(direction='in',top=True,right=True);ax.legend(loc='upper left',fontsize=6.8,frameon=False)
fig.text(.16,.055,'All 800 paired estimates; no subsampling or jitter.\nBlack diamonds / crossbars: means / ±1 SD.\nMaximum absolute difference = 1.33e−9.',fontsize=7,linespacing=1.5)
save(fig,'G2_v2_07_matched_scatter',wide,'R23','Agreement of official and M point estimates','Raw grouped colored points, black mean diamonds with SD crossbars, full thin frame, inward ticks and equality reference line. Exact overlap is real; no artificial spreading. Agreement is not superiority.')
(V/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('Created',len(manifest),'single figures')
