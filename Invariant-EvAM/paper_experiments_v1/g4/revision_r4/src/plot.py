from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
G=Path(__file__).resolve().parents[1];R=G.parents[2]
sys.path.insert(0,str(R/'src'))
from invariant_evam.figure_style import apply_publication_style
apply_publication_style({'font.size':8,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'svg.fonttype':'none','pdf.fonttype':42})
C=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7'];CAT=json.loads((R/'configs/figure_reference_catalog.json').read_text())['records'];MAN=[]
F=json.loads((G/'configs/EVALUATION_FREEZE.json').read_text());CAL=F['primary_calibration']
D=pd.read_csv(G/'results/evaluation_raw.tsv',sep='\t');DES=['REDUNDANT','HETEROGENEOUS','CROSSVIEW_FAILURE'];DN=['Redundant','Heterogeneous','Missing joint path']
def canvas(title,ylabel,labels,colors):
    f,a=plt.subplots(figsize=(158/25.4,112/25.4));f.subplots_adjust(left=.14,right=.975,bottom=.23,top=.77)
    a.spines[['top','right']].set_visible(False);a.set_ylabel(ylabel);a.axhline(0,c='#555555',lw=.65,zorder=1)
    f.text(.14,.956,title,fontsize=10,weight='bold')
    handles=[Line2D([],[],ls='',marker='o',ms=5,mfc=c,mec='#222222',mew=.6,label=l) for l,c in zip(labels,colors)]
    f.legend(handles=handles,loc='upper left',bbox_to_anchor=(.13,.907),ncol=len(labels),frameon=True,fancybox=False,edgecolor='#b6b6b6',fontsize=7,handletextpad=.3,columnspacing=.85,borderpad=.35)
    return f,a
def bar(a,x,v,c,width):
    v=v.dropna();mean=v.mean();err=1.96*v.std(ddof=1)/np.sqrt(len(v));a.bar(x,mean,width,fc=c,ec='#222222',lw=.65,zorder=2)
    a.errorbar(x,mean,yerr=err,fmt='none',ecolor='#222222',capsize=2,lw=.65,zorder=3)
    batch=v.groupby(v.index//10).mean();offset=np.linspace(-.028,.028,len(batch));a.scatter(x+offset,batch,s=9,fc='white',ec='#333333',lw=.5,zorder=4)
    return min(0,mean-err,batch.min()),max(0,mean+err,batch.max())
def finish_axes(a,extrema):
    low=min(v[0] for v in extrema);high=max(v[1] for v in extrema);span=high-low
    a.set_ylim(low-.08*span if low<0 else 0,high+.16*span);a.set_xticks(range(3),DN);a.set_xlim(-.53,2.53)
def save(f,name,data,ref,adapt,foot):
    f.text(.14,.075,foot,fontsize=6.6,linespacing=1.45)
    data.to_csv(G/f'figure_data/{name}.tsv',sep='\t',index=False)
    for ext in ['pdf','svg']:f.savefig(G.parent/f'final_figures/{name}.{ext}',dpi=600)
    p=Path('/tmp/invariant_g4_r4_previews');p.mkdir(exist_ok=True);f.savefig(p/f'{name}.png',dpi=150)
    source=next(x for x in CAT if x['id']==ref)['source_file_absolute']
    MAN.append(dict(figure=name,reference=ref,reference_file=source,reference_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),size_mm=(f.get_size_inches()*25.4).tolist(),dpi=600,palette=C,adaptations=adapt,visual_review='AWAITING_ACTUAL_EXPORT_REVIEW',user_approved=False));plt.close(f)
for cancer in ['LUAD','COAD']:
    q=D[D.cancer.eq(cancer)];labels=['Joint CTMC','Pooled','Hierarchical','Two-step'];methods=['CTMC','POOLED','HIERARCHICAL','TWO_STEP'];colors=[C[0],C[1],C[2],C[3]]
    f,a=canvas(f'G4B | Equal-cost bridge measurements · {cancer}','Hidden log-score gain (millinats / record)',labels,colors);ext=[];rows=[]
    for k,design in enumerate(DES):
        w=q[q.design.eq(design)].pivot(index='replicate',columns='method',values='log_score')
        for j,m in enumerate(methods):
            v=1000*(w['MORE_'+m]-w['BRIDGE_'+m]);ext.append(bar(a,k+(j-1.5)*.185,v,colors[j],.156))
            rows.extend(dict(cancer=cancer,design=design,method=m,replicate=int(r),gain_millinats=float(x),display_batch=int(r//10)) for r,x in v.items())
    finish_axes(a,ext);ns=q[q.method.eq('BASE_CTMC')].groupby('design').size();n='/'.join(str(ns.get(x,0)) for x in DES)
    save(f,f'G4_R4_B_gain_{cancer}',pd.DataFrame(rows),'R21','Narrow outlined bars, open fixed-index batch points, black capped errors and top single-row boxed legend. Signed gain includes zero; no invented P values. Raw fusion remains in result tables, not hidden by axis truncation.',f'Both arms: 638 gene assays; bridge n=80 versus routine expansion n=106.\nCompleted paired seeds: {n} of 50/50/50. Caps: 1.96 seed-MCSE.\nOpen points: fixed batches of 10 seeds; all completed seeds contribute.')
    methods=['BASE_CTMC',CAL,'BRIDGE_CTMC','AUDIT_MARGINAL_1'];labels=['Base CTMC','Audit calibration','Joint refit','Marginal only'];colors=[C[1],C[2],C[0],C[3]]
    f,a=canvas(f'G4C | Hidden-event probability calibration · {cancer}','Calibration error (percentage points) ↓',labels,colors);ext=[]
    for k,design in enumerate(DES):
        for j,m in enumerate(methods):
            z=q[q.design.eq(design)&q.method.eq(m)].set_index('replicate').ece*100;ext.append(bar(a,k+(j-1.5)*.185,z,colors[j],.156))
    finish_axes(a,ext)
    save(f,f'G4_R4_C_calibration_{cancer}',q[q.method.isin(methods)],'R21','Same grouped-bar geometry with conditional probability calibration as endpoint. Constant marginal control retained; ECE alone does not establish useful prediction.',f'Five probability bins; paired seeds {n}; caps: 1.96 seed-MCSE.\nBase cost 560; each audit method pays 78 extra assays (total 638).\nMarginal-only can lower ECE while worsening proper scores; see score-control figure.')
# All-design proper-score control prevents calibration-only success claims.
methods=['MORE_CTMC',CAL,'BRIDGE_CTMC','AUDIT_DIRECT_1','AUDIT_MARGINAL_1'];labels=['More records','Audit calibration','Joint refit','Direct cells','Marginal only'];colors=[C[1],C[2],C[0],C[5],C[3]]
f,a=canvas('G4C | Proper-score controls across all three designs','Log-score gain over base (millinats / record)',labels,colors);ext=[];rr=[]
for k,cancer in enumerate(['LUAD','COAD']):
    q=D[D.cancer.eq(cancer)];w=q.pivot(index=['replicate','design'],columns='method',values='log_score')
    for j,m in enumerate(methods):
        v=1000*(w.BASE_CTMC-w[m]).unstack().dropna().mean(axis=1);ext.append(bar(a,k+(j-2)*.14,v,colors[j],.12));rr.extend(dict(cancer=cancer,method=m,replicate=int(r),gain_millinats=float(x)) for r,x in v.items())
low=min(t[0] for t in ext);high=max(t[1] for t in ext);a.set_ylim(low-.09*(high-low),high+.18*(high-low));a.set_xticks([0,1],['LUAD','COAD']);a.set_xlim(-.5,1.5)
save(f,'G4_R4_C_proper_score_controls',pd.DataFrame(rr),'R21','Five narrow bars per cancer, paired differences rather than nearly touching absolute losses. Equal design means use seeds with all three designs available; failures separately retained. Negative controls remain visible.','Positive values mean better predictions; zero is the base CTMC (cost 560).\nEvery candidate costs 638. Caps: seed-MCSE; open points: fixed batches of 10.\nEqual mean of all three designs; incomplete paired-seed blocks are reported in ledger.')
if (G/'results/bootstrap_raw.tsv').exists():
    b=pd.read_csv(G/'results/bootstrap_raw.tsv',sep='\t');rows=[];series=[]
    for cancer in ['LUAD','COAD']:
        w=b[b.cancer.eq(cancer)].pivot(index=['replicate','design'],columns='method',values='log_score')
        for label,ref,cand in [('B','MORE_CTMC','BRIDGE_CTMC'),('C','BASE_CTMC',CAL)]:
            v=1000*(w[ref]-w[cand]).unstack().dropna().mean(axis=1);series.append((f'{label} · {cancer}',v));rows.extend(dict(comparison=label,cancer=cancer,replicate=int(r),gain_millinats=float(x),display_point=bool(r%10==0)) for r,x in v.items())
    f,a=plt.subplots(figsize=(158/25.4,112/25.4));f.subplots_adjust(left=.14,right=.80,bottom=.24,top=.85);f.text(.14,.952,'G4B–C | Patient-resampling uncertainty',fontsize=10,weight='bold');a.set_ylabel('Log-score gain (millinats / record)');a.axhline(0,c='#555555',lw=.7,ls=(0,(3,3)))
    for k,(label,v) in enumerate(series):
        y=np.linspace(v.min(),v.max(),180);den=gaussian_kde(v)(y);den=den/den.max()*.28;a.fill_betweenx(y,k,k+den,fc=C[[0,2,1,5][k]],ec='#444444',lw=.6,alpha=.8)
        a.boxplot([v.values],positions=[k],widths=.075,patch_artist=True,showfliers=False,boxprops=dict(fc='white',ec='#333333',lw=.7),medianprops=dict(c='#333333',lw=.8),whiskerprops=dict(c='#333333',lw=.6),capprops=dict(c='#333333',lw=.6))
        z=v[v.index%10==0];jit=np.random.default_rng(902+k).uniform(.09,.27,len(z));a.scatter(k-jit,z,s=8,c=C[[0,2,1,5][k]],alpha=.7,lw=0,zorder=3)
    a.set_xticks(range(4),[f'{label}\nn={len(v)}' for label,v in series]);a.set_xlim(-.38,3.4);a.spines[['top','right']].set_color('#bbbbbb');a.spines[['top','right']].set_linewidth(.55)
    f.legend(handles=[Patch(fc=C[[0,2,1,5][k]],ec='#333333',lw=.5,label=label) for k,(label,v) in enumerate(series)],loc='center left',bbox_to_anchor=(.80,.54),fontsize=7,handlelength=1,handletextpad=.4)
    save(f,'G4_R4_BC_patient_bootstrap',pd.DataFrame(rows),'R27','Left sampled points, narrow white central box and right half density, closed light frame and right legend. All bootstrap values determine density/box; displayed points use fixed replicate modulo10. Signed endpoint requires negative axis. B=bridge vs more; C=frozen calibration vs base.','200 role-wise patient bootstraps per cell; complete three-design blocks shown.\nEvery fit is rerun; dots: fixed every-tenth replicate; box/density: all valid blocks.\nDistributions combine patient resampling and one fresh perturbation per bootstrap.')
(G/'configs/FIGURE_MANIFEST.json').write_text(json.dumps(MAN,indent=2))
