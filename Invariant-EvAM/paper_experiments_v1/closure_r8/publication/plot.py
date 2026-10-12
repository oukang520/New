from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde,norm
R=Path(__file__).resolve().parents[3];P=R/'paper_experiments_v1';O=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'src'))
from invariant_evam.figure_style import apply_publication_style
apply_publication_style({'font.family':'Arial','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold'})
C=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7'];K='#292929'
OUT=O/'staging';OUT.mkdir(exist_ok=True);D=O/'plot_data';D.mkdir(exist_ok=True)
REV=Path('/tmp/evam_r8_publication_review');REV.mkdir(exist_ok=True)
manifest=[];tests=[]
def tidy(ax,box=False):
 for s in ['top','right']:ax.spines[s].set_visible(box)
 ax.tick_params(direction='out',length=3,pad=2)
 ax.set_axisbelow(True)
def wilson(k,n):
 z=norm.ppf(.975);p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return min(p,max(0,mid-h)),max(p,min(1,mid+h))
def save(fig,g,name,ref,msg,data,adapt):
 fig.canvas.draw();common_bbox=fig.get_tightbbox(fig.canvas.get_renderer()).padded(.035)
 for ext in ['pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=600,facecolor='white',bbox_inches=common_bbox,pad_inches=0)
 fig.savefig(REV/f'{name}.png',dpi=160,bbox_inches='tight',pad_inches=.035)
 template=json.loads((R/'configs/figure_manifest_template.json').read_text());template.update(figure_id=name,panel_id=name,scientific_message=msg,data_source=data,estimand=msg,unit_of_analysis='independent parameter draw or simulated confirmation dataset; see full caption',n_definition='all prescribed cases and failures retained; see plot-data counts',primary_reference_id=ref,reference_selection_reason=msg,code_path=str(Path(__file__)),plot_data_path=str(D/f'{name}.tsv'),output_paths=[str(P/g/'final_figures'/f'{name}.{e}') for e in ['pdf','svg']],adaptations_from_reference=adapt,palette_mapping=dict(enumerate(C)),review_status='awaiting_visual_review',vector_text_preserved=True,final_width_mm=fig.get_figwidth()*25.4,final_height_mm=fig.get_figheight()*25.4)
 manifest.append(template);plt.close(fig)
# G1: same-budget information, full prescribed topology/regime matrix.
f=P/'g1/revision_r8/results/paired_information.tsv';raw=pd.read_csv(f,sep='\t');raw.to_csv(D/'G1B_raw.tsv',sep='\t',index=False)
s=raw.groupby(['d','topology','regime']).agg(x=('DUPLICATE','median'),y=('COMPLEMENTARY','median'),n=('replicate','size')).reset_index();s['displayed_positive']=((s.x>0)&(s.y>0));s.to_csv(D/'G1B.tsv',sep='\t',index=False)
fig=plt.figure(figsize=(5.3,4.5));ax=fig.add_axes([.13,.16,.66,.62]);top=fig.add_axes([.13,.79,.66,.1],sharex=ax);right=fig.add_axes([.80,.16,.1,.62],sharey=ax)
lo=-6;hi=3;grid=np.linspace(lo,hi,400);colors=dict(CHAIN=C[0],DENSE_DAG=C[2],FULL=C[4]);names={'CHAIN':'Chain','DENSE_DAG':'Dense DAG','FULL':'Full bidirectional'}
for kind,col in colors.items():
 z=s[(s.topology==kind)&s.displayed_positive];x=np.log10(z.x);y=np.log10(z.y)
 ax.scatter(x,y,s=29,c=col,edgecolors=K,linewidths=.45,zorder=3,label=f'{names[kind]} ({len(z)} cells)')
 for vals,aa,orient in [(x,top,'x'),(y,right,'y')]:
  density=gaussian_kde(vals,bw_method=.4)(grid)
  if orient=='x':aa.fill_between(grid,0,density,color=col,alpha=.22,lw=0);aa.plot(grid,density,color=col,lw=.75)
  else:aa.fill_betweenx(grid,0,density,color=col,alpha=.22,lw=0);aa.plot(density,grid,color=col,lw=.75)
ax.plot([lo,hi],[lo,hi],ls='--',color=C[5],lw=1);ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Target information: repeated observation',ylabel='Target information: complementary observation');ticks=np.arange(-6,4,3);ax.set_xticks(ticks);ax.set_yticks(ticks);labels=[f'$10^{{{v}}}$' for v in ticks];ax.set_xticklabels(labels);ax.set_yticklabels(labels);ax.grid(True,ls=':',lw=.5,color='#bdbdbd');tidy(ax,True)
ax.legend(loc='upper left',fontsize=7.2,handletextpad=.4,borderpad=.25,labelspacing=.35)
for aa in [top,right]:aa.set_axis_off()
ins=ax.inset_axes([.46,.07,.51,.29]);arr=[]
for kind in colors:
 z=raw[(raw.topology==kind)&(raw.DUPLICATE>0)&(raw.COMPLEMENTARY>0)];arr.append(np.log10(z.COMPLEMENTARY/z.DUPLICATE).values)
bp=ins.boxplot(arr,vert=False,patch_artist=True,widths=.5,showfliers=False,whis=(5,95),medianprops=dict(color=K,lw=1),boxprops=dict(lw=.6),whiskerprops=dict(lw=.6),capprops=dict(lw=.6))
for b,c in zip(bp['boxes'],colors.values()):b.set_facecolor(c);b.set_alpha(.55)
ins.axvline(0,color=K,ls=':',lw=.7);ins.set_yticks([1,2,3],['Chain','DAG','Full'],fontsize=6.3);ins.set_xticks([0,2,4,6]);ins.set_xlim(-.5,6);ins.tick_params(labelsize=6.3,length=2,pad=1);ins.set_title('log10 information gain',fontsize=7,pad=3);tidy(ins,True)
fig.text(.13,.055,'Full network, 3–4 events: 60/60 target-nonidentifiable draws',fontsize=7.3)
fig.text(.13,.012,'3–8 events · 10,000 assays · dots: medians of 10 draws',fontsize=7.2)
save(fig,'g1','G1B','R02','same-budget target information across model complexity',str(f),['30 cell medians replace dense raw scatter; all 360 draws remain in data','zero-information cells explicitly reported outside log axes','gain boxplot replaces reference residual; no borrowed significance'])
# G2: coverage/finite/invalid states, all budgets and fresh confirmation.
a=pd.read_csv(P/'g2/revision_r8/results/evaluation_all_methods.tsv',sep='\t');a=a[a.method.isin(['CLASSIC_PROFILE_REFINED_V2','EXACT_CONDITIONAL_OUTER'])].copy();a['stage']='primary';b=pd.read_csv(P/'g2/revision_r8/fresh_confirmation/results.tsv',sep='\t');b['stage']='fresh';b['valid']=b['valid'].fillna(b.status.eq('COMPLETE'));both=pd.concat([a,b],ignore_index=True);both.to_csv(D/'G2_02_raw.tsv',sep='\t',index=False)
conditions=[('primary',des,n) for des in ['strong','weak','nonidentifying'] for n in [1000,4000,16000]]+[('fresh',des,16000) for des in ['strong','weak']];ys=[0,1,2,3.3,4.3,5.3,6.6,7.6,8.6,10,11];labs=[];rows=[]
fig=plt.figure(figsize=(7.1,4.7));axes=[fig.add_axes(r) for r in [[.24,.19,.235,.69],[.505,.19,.235,.69],[.79,.19,.19,.69]]];statecolors=[C[2],C[1],C[4],C[5]];statekeys=['covered_finite','covered_unbounded','not_covered','invalid']
for row,(stage,des,n) in enumerate(conditions):
 count=200 if stage=='primary' else 100;lab={'strong':'Complementary','weak':'Same selection','nonidentifying':'Separate marginals'}[des];labs.append(f'{lab} · {n//1000}k' if stage=='primary' else f'Fresh: {lab} · 16k')
 for mi,method in enumerate(['CLASSIC_PROFILE_REFINED_V2','EXACT_CONDITIONAL_OUTER']):
  z=both[(both.stage==stage)&(both.design==des)&(both.n.fillna(16000)==n)&(both.method==method)];assert len(z)==count,(stage,des,n,method,len(z));valid=z.valid.astype(bool);covered=z.covered.astype(bool);finite=z.finite.astype(bool);vals=np.array([(valid&covered&finite).sum(),(valid&covered&~finite).sum(),(valid&~covered).sum(),(~valid).sum()]);assert vals.sum()==count
  out=dict(stage=stage,design=des,n=n,method=method,replicates=count,correct=int(z.correct.sum()),**dict(zip(statekeys,vals)));rows.append(out);left=0
  for v,c in zip(vals/count*100,statecolors):axes[mi].barh(ys[row],v,left=left,height=.73,color=c,edgecolor='white',lw=.4);left+=v
  k=int(z.correct.sum());cl,ch=wilson(k,count);mean=k/count*100;axes[2].errorbar(mean,ys[row]+(-.13 if mi==0 else .13),xerr=[[mean-cl*100],[ch*100-mean]],fmt='o' if mi==0 else 's',ms=2.6,mfc=K if mi==0 else 'white',mec=K,mew=.65,color=K,lw=.7,capsize=2.7)
for i,aa in enumerate(axes):
 aa.set_ylim(11.65,-.65);aa.set_xlim(-5 if i==2 else 0,105 if i==2 else 100);aa.set_xticks([0,50,100]);aa.set_xlabel('Share (%)' if i<2 else 'Correct answers (%)',labelpad=4);aa.set_yticks(ys);aa.set_yticklabels(labs if i==0 else [],fontsize=7.5);aa.tick_params(axis='y',length=0);tidy(aa,True)
 for sep in [2.65,5.95,9.35]:aa.axhline(sep,color='#cccccc',lw=.55)
 if i<2:aa.axvline(95,color=K,ls=':',lw=.7)
 aa.set_title(['Profile likelihood','Conditional bounds','95% CI'][i],fontsize=8.5,pad=8)
fig.legend([Patch(fc=c) for c in statecolors],['Covered, finite','Covered, unbounded','Not covered','Numerically invalid'],loc='lower center',bbox_to_anchor=(.51,.052),ncol=4,fontsize=7.1,handlelength=1.2,columnspacing=1.25)
fig.legend([Line2D([],[],marker='o',color=K,lw=0,ms=3.5),Line2D([],[],marker='s',mfc='white',color=K,lw=0,ms=3.5)],['Profile','Conditional'],loc='upper right',bbox_to_anchor=(.994,.982),ncol=2,fontsize=7.1,handletextpad=.25,columnspacing=.65)
fig.text(.24,.012,'n = 200/cell; fresh confirmation n = 100/cell. Dotted line: 95% coverage.',fontsize=7.2)
pd.DataFrame(rows).to_csv(D/'G2_02.tsv',sep='\t',index=False)
save(fig,'g2','G2_02','R28','reliable finite intervals versus abstention and numerical failure',str(P/'g2/revision_r8/results/evaluation_all_methods.tsv'),['paired method compositions and Wilson correct-answer intervals share rows','95% coverage reference replaces uninformative significance labels','primary and new confirmation remain separate; no pooling'])
# G5 action mechanism, plus the equal-size independent check.
g5=pd.read_csv(P/'g5/revision_r8/results/evaluation_all_policies.tsv',sep='\t');chosen=g5[g5.policy=='TARGET_C_OPT'].copy();chosen['stage']='primary';match=pd.read_csv(P/'g5/revision_r8/matched_size/results.tsv',sep='\t');match=match[match.chosen].copy();match['stage']='matched';selected=pd.concat([chosen,match],ignore_index=True);selected.to_csv(D/'G5_R8_diagnostic_actions_raw.tsv',sep='\t',index=False)
actions=['MORE_SAME','JOINT_REVEAL','NEW_ENV','NEW_SOURCE','REFERENCE_MARGIN'];ac=[C[0],C[2],C[4],C[3],C[5]];an=['More samples','Joint reveal','New environment','New source','Reference margin'];conditions=[('primary',i) for i in range(4)]+[('matched',0),('matched',3)];labels=['Sample limited','Missing joint view','Unknown margin','Rare strata','Equal N: low bias','Equal N: high bias'];ys=[0,1,2,3,4.5,5.5];fig=plt.figure(figsize=(7.2,3.6));axes=[fig.add_axes(r) for r in [[.215,.24,.215,.60],[.46,.24,.17,.60],[.68,.24,.105,.60]]];rows=[]
for yi,(stage,case),lab in zip(ys,conditions,labels):
 z=selected[(selected.stage==stage)&(selected.case==case)];n=len(z);left=0;row=dict(stage=stage,case=case,n=n)
 for act,c in zip(actions,ac):
  v=(z.action==act).sum();row[act]=v;axes[0].barh(yi,v/n*100,left=left,height=.7,color=c,edgecolor='white',lw=.45);left+=v/n*100
 correct=int(z.correct.sum());wrong=int(z.wrong.sum());unknown=n-correct-wrong;left=0
 for k,c in [(correct,C[2]),(unknown,C[1]),(wrong,C[4])]:axes[1].barh(yi,k/n*100,left=left,height=.7,color=c,edgecolor='white',lw=.45);left+=k/n*100
 finite=int(z.finite.sum());lo,hi=wilson(finite,n);mean=finite/n*100;axes[2].errorbar(mean,yi,xerr=[[mean-lo*100],[hi*100-mean]],fmt='o',color=K,ms=3.5,lw=.8,capsize=2);row.update(correct=correct,wrong=wrong,unknown=unknown,finite=finite,covered=int(z.covered.sum()));rows.append(row)
for i,aa in enumerate(axes):
 aa.set(ylim=(6.15,-.65),xlim=(-2 if i==2 else 0,104 if i==2 else 100));aa.set_xticks([0,50,100]);aa.set_yticks(ys);aa.set_yticklabels([f'{lab} (n={r["n"]})' for lab,r in zip(labels,rows)] if i==0 else [],fontsize=7.2);aa.tick_params(axis='y',length=0);aa.set_xlabel('Share (%)' if i<2 else 'Finite (%)');aa.set_title(['Selected action','Answer status',''][i],fontsize=8.3,pad=7);aa.axhline(3.75,color='#bbbbbb',lw=.65);tidy(aa,True)
fig.legend([Patch(fc=c) for c in ac],an,loc='upper left',bbox_to_anchor=(.805,.88),fontsize=7,handlelength=.9,handletextpad=.4,labelspacing=.65)
fig.legend([Patch(fc=c) for c in [C[2],C[1],C[4]]],['Correct','Unknown','Wrong'],loc='upper left',bbox_to_anchor=(.805,.46),fontsize=7,handlelength=.9,handletextpad=.4,labelspacing=.65)
fig.text(.215,.095,'Added budget: 6,000 units · independent pilot and confirmation',fontsize=7.4)

pd.DataFrame(rows).to_csv(D/'G5_R8_diagnostic_actions.tsv',sep='\t',index=False)
save(fig,'g5','G5_R8_diagnostic_actions','R28','different information gaps lead to distinct acquisition actions and useful answers',str(P/'g5/revision_r8/results/evaluation_all_policies.tsv'),['six rows including matched-size confirmation instead of the reference 20 categories','finite-interval Wilson intervals replace the side mean points','all unknown and zero-error results retain their full denominator'])
# G5 paired utility: all data in means/CIs/tests; fixed visual subset only.
policies=['MORE_SAME','DEVELOPMENT_BEST_FIXED','TARGET_C_OPT'];pc=[C[0],C[1],C[2]];pnames=['More samples','Best fixed: new source','Target c-opt'];rng=np.random.default_rng(202609289);plot=[];pointrows=[];testrows=[]
for case in range(4):
 z=g5[g5.case==case].pivot(index='replicate',columns='policy',values='restricted_width').sort_index();ids=z.index.to_numpy();shown=np.sort(np.random.default_rng([202609289,case]).choice(ids,12,replace=False));ix=rng.integers(0,len(z),(10000,len(z)))
 for policy in policies:
  vals=z[policy].to_numpy();means=vals[ix].mean(axis=1);plot.append(dict(case=case,policy=policy,n=len(vals),mean=vals.mean(),lo=np.quantile(means,.025),hi=np.quantile(means,.975)))
  for rep,v in zip(ids,vals):pointrows.append(dict(case=case,policy=policy,replicate=rep,width=v,displayed=rep in shown))
 for other in policies[:2]:
  diff=(z[other]-z['TARGET_C_OPT']).to_numpy();effect=diff.mean();flips=np.random.default_rng([202609289,case,policies.index(other)]).choice([-1,1],size=(49999,len(diff)));p=(1+np.sum(abs((flips*diff).mean(axis=1))>=abs(effect)-1e-14))/50000;testrows.append(dict(panel='G5_R8_diagnostic_value',case=case,comparator=other,n=len(diff),effect=effect,p_raw=p,test='two-sided paired mean sign-flip; symmetric null',unit='independent confirmation replicate',family_size=8))
pvals=np.array([r['p_raw'] for r in testrows]);order=np.argsort(pvals);adj=np.maximum.accumulate(pvals[order]*(len(pvals)-np.arange(len(pvals))));q=np.empty(len(pvals));q[order]=np.minimum(adj,1)
for row,v in zip(testrows,q):row['p_holm']=v;row['label']='***' if v<.001 else '**' if v<.01 else '*' if v<.05 else 'ns'
tests.extend(testrows);plot=pd.DataFrame(plot);points=pd.DataFrame(pointrows);plot.to_csv(D/'G5_R8_diagnostic_value.tsv',sep='\t',index=False);points.to_csv(D/'G5_R8_diagnostic_value_points.tsv',sep='\t',index=False)
fig,ax=plt.subplots(figsize=(6.4,3.85));fig.subplots_adjust(left=.115,right=.99,bottom=.19,top=.90);width=.19;offsets=[-.23,0,.23]
for case in range(4):
 for pi,policy in enumerate(policies):
  r=plot[(plot.case==case)&(plot.policy==policy)].iloc[0];x=case+offsets[pi];ax.bar(x,r['mean'],width=width,color=pc[pi],edgecolor=K,lw=.75,zorder=2)
  zz=points[(points.case==case)&(points.policy==policy)&points.displayed];jitter=np.random.default_rng([202609289,case,pi,3]).uniform(-.064,.064,len(zz));ax.scatter(x+jitter,zz.width,s=12,facecolor='white',edgecolor=K,lw=.5,zorder=4,alpha=.85);ax.errorbar(x,r['mean'],yerr=[[r['mean']-r.lo],[r.hi-r['mean']]],fmt='none',ecolor=K,capsize=2,lw=.8,zorder=5)
 # Brackets point precisely to each tested pair, with tiers above all observed points.
 top=max(points[(points.case==case)&points.displayed].width.max(),plot[plot.case==case].hi.max())
 for tier,other in enumerate(policies[:2]):
  test=next(t for t in testrows if t['case']==case and t['comparator']==other);x1=case+offsets[policies.index(other)];x2=case+offsets[2];y=top+.38+tier*.65;ax.plot([x1,x1,x2,x2],[y-.13,y,y,y-.13],color=K,lw=.7,clip_on=False);ax.text((x1+x2)/2,y+.07,test['label'],ha='center',va='bottom',fontsize=7.4)
ax.set_xticks(range(4),['Sample\nlimited','Missing\njoint view','Unknown\nmargin','Rare\nstrata']);ax.set_ylabel('Interval width within [−4, 4]');ax.set_ylim(0,10);ax.set_yticks([0,2,4,6,8,10]);tidy(ax);fig.legend([Patch(fc=c,ec=K,lw=.6) for c in pc],pnames,loc='upper center',bbox_to_anchor=(.56,.985),ncol=3,fontsize=7.8,columnspacing=1.2,handlelength=1.1);fig.text(.115,.025,'n = 60/cell · 95% bootstrap intervals · paired tests, Holm-adjusted',fontsize=7.2)
save(fig,'g5','G5_R8_diagnostic_value','R21','paired benefit of adaptive acquisition versus fixed same-budget actions',str(P/'g5/revision_r8/results/evaluation_all_policies.tsv'),['four scenario groups with three policies; hollow data points and bootstrap intervals retained','12/60 outcome-independent display points per bar; all 60 enter estimates and tests','explicit pair brackets replace the reference vertical p text; full unbounded intervals remain in data'])
pd.DataFrame(tests).to_csv(O/'SIGNIFICANCE_TESTS_R8.tsv',sep='\t',index=False)
(O/'configs/FIGURE_MANIFESTS.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('Generated',len(manifest),'panels; awaiting visual review')
