"""Stage three scoped biological-support figures; never publish or alter results.

References actually inspected: R29 hollow bars/individual points/caps and
R04 colored lines/black-edged circles/black caps/top legend/two data insets.
The phone UI, tutorial heading, arrows and unsupported stars are not copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import font_manager

BASE = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
from invariant_evam.figure_style import apply_publication_style

PALETTE = ["#0072B2", "#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
METHODS = ["CLONAL_ONLY", "BRANCH_DIVERSE", "RANDOM_MARKERS", "INFORMATION_ADAPTIVE"]
LABELS = {"CLONAL_ONLY": "Clonal", "BRANCH_DIVERSE": "Diverse",
          "RANDOM_MARKERS": "Random", "INFORMATION_ADAPTIVE": "Adaptive"}
COLORS = {"CLONAL_ONLY": PALETTE[1], "BRANCH_DIVERSE": PALETTE[2],
          "RANDOM_MARKERS": PALETTE[3], "INFORMATION_ADAPTIVE": PALETTE[0]}
BLACK = "#222222"
REFERENCES = {
    "R29": Path("/Users/oukang/Desktop/投稿资料/IQE/nature图例/fb5b46eb9066195f9667702674a8791b.png"),
    "R04": Path("/Users/oukang/Desktop/投稿资料/IQE/nature图例/14ac6b8bd93184b6034ef103c4d80d13.jpg"),
    "R21": Path("/Users/oukang/Desktop/投稿资料/IQE/nature图例/b7f665c0644dd5d35226c0b3e29df4f0.jpg"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_mean(values: np.ndarray, endpoint_key: str) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Invalid patient vector for {endpoint_key}")
    # Visualization uncertainty only: fixed2000 patient bootstrap, no method,
    # patient, budget, seed or scientific-endpoint selection from these draws.
    salt = int(hashlib.sha256(endpoint_key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng([20261009, salt])
    boots = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
    lo, hi = np.quantile(boots, [.025, .975])
    return float(values.mean()), float(lo), float(hi)


def bool_values(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lower().eq("true").to_numpy(dtype=float)


def tidy(ax, inset: bool = False) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(.8 if inset else 1.)
    ax.tick_params(direction="out", length=2.2 if inset else 3., width=.75,
                   pad=1.8, labelsize=6.8 if inset else 8)
    ax.set_facecolor("white")
    ax.grid(False)


def export(fig, stem: Path) -> dict:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    # Reduce external margins only after the endpoint/annotation geometry is
    # fixed. Both vector exports use the same actual physical bounding box.
    bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(.045)
    for ext in ("pdf", "svg"):
        fig.savefig(stem.with_suffix("." + ext), dpi=600, bbox_inches=bbox,
                    pad_inches=0, facecolor="white")
    width_mm, height_mm = bbox.width * 25.4, bbox.height * 25.4
    plt.close(fig)
    return dict(output_paths=[str(stem.with_suffix(".pdf")), str(stem.with_suffix(".svg"))],
        width_mm=width_mm, height_mm=height_mm, font="Arial", dpi=600,
        pdf_sha256=sha(stem.with_suffix(".pdf")), svg_sha256=sha(stem.with_suffix(".svg")))


def plot_g4(out: Path) -> tuple[dict, str]:
    name = "G4_BIO_01_later_lineage"
    summary_path = BASE / "results/metastasis/SUMMARY.tsv"
    raw_path = BASE / "results/metastasis/confirmation_patients.tsv"
    summaries = pd.read_csv(summary_path, sep="\t")
    rows = pd.read_csv(raw_path, sep="\t")
    summaries = summaries[(summaries.population == "ALL_NSCLC") &
        (summaries.method == "BEST_FIXED") & (summaries.overhead == 2)].sort_values("budget")
    rows = rows[(rows.population == "ALL_NSCLC") & (rows.method == "BEST_FIXED") &
        (rows.overhead == 2)].copy()
    assert summaries.budget.tolist() == [10, 14, 20]
    assert summaries.n_future_patients.eq(14).all() and summaries.n_evaluable_patients.eq(9).all()
    evaluable = rows[rows.future_branch_recall.notna()]
    assert evaluable.groupby("budget").patient_id.nunique().eq(9).all()
    ids = sorted(evaluable.patient_id.unique())
    offsets = dict(zip(ids, np.linspace(-.165, .165, len(ids))))
    fig, ax = plt.subplots(figsize=(3.3, 3.25))
    fig.subplots_adjust(left=.23, right=.975, bottom=.19, top=.90)
    plot_rows = []
    for i, record in enumerate(summaries.itertuples(index=False)):
        color = PALETTE[[0, 1, 2][i]]
        mean, lo, hi = np.array([record.mean, record.lo, record.hi]) * 100
        vals = evaluable[evaluable.budget.eq(record.budget)].sort_values("patient_id")
        assert np.isclose(vals.future_branch_recall.mean(), record.mean)
        ax.bar(i, mean, width=.56, facecolor="white", edgecolor=color,
               linewidth=1.7, zorder=2)
        ax.errorbar(i, mean, yerr=[[mean-lo], [hi-mean]], fmt="none",
            ecolor=BLACK, elinewidth=.85, capsize=3, capthick=.85, zorder=4)
        x = np.array([i + offsets[p] for p in vals.patient_id])
        ax.scatter(x, vals.future_branch_recall.to_numpy()*100, s=17,
                   color=color, edgecolor="white", linewidth=.25, zorder=5)
        plot_rows.append(dict(component="mean_ci", population="ALL_NSCLC", method="BEST_FIXED",
            budget=record.budget, overhead=2, n_future=14, n_evaluable=9,
            patient_id="", value=record.mean, lo=record.lo, hi=record.hi, x=i))
        for point, xx in zip(vals.itertuples(index=False), x):
            plot_rows.append(dict(component="patient", population="ALL_NSCLC", method="BEST_FIXED",
                budget=record.budget, overhead=2, n_future=14, n_evaluable=9,
                patient_id=point.patient_id, value=point.future_branch_recall, lo=np.nan, hi=np.nan, x=xx))
    ax.set_xticks([0, 1, 2], ["10", "14", "20"])
    ax.set_xlim(-.6, 2.6)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Incremental budget (cost units)", labelpad=4)
    ax.set_ylabel("Later lineage recovery (%)", labelpad=5)
    tidy(ax)
    ax.text(.5, 1.035, "Best fixed strategy; n=9 evaluable / 14 future",
        ha="center", va="bottom", transform=ax.transAxes, fontsize=7.4)
    plot_data = BASE / "publication_data/g4" / f"{name}.tsv"
    plot_data.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(plot_rows).to_csv(plot_data, sep="\t", index=False)
    metadata = export(fig, out / "g4" / name)
    metadata.update(panel=name, group="g4", primary_reference_id="R29",
        scientific_message="Increasing the frozen best-fixed classical acquisition budget recovers more primary-tumour marker lineages later observed in actual post-recurrence samples; not diagnostic superiority.",
        estimand="Patient-averaged recovery of supported primary nontrunk marker clusters with real metachronous sample support",
        unit_of_analysis="Independent confirmation patient, five spatial-start perturbations averaged first",
        n_definition="14 confirmation patients have future samples;9 have an evaluable supported reference;5 retain unknown/no-reference status outside this recovery denominator",
        plot_data_path=str(plot_data), data_sha256=sha(plot_data),
        data_source=[str(summary_path), str(raw_path)], source_hashes={str(p):sha(p) for p in [summary_path, raw_path]},
        adaptations=["Three hollow outlined bars, every nine patient points and black capped95%bootstrap intervals reproduce R29 geometry",
            "Numeric acquisition budgets replace treatment group names; no unsupported comparison stars/brackets",
            "Reference is actual later-sample marker support, not generator-hazard or independently verified ancestry truth"],
        statistical_annotations="No significance annotation; no newly registered paired budget test",
        review_status="awaiting_actual_export_review", user_approved=False)
    caption = """## G4_BIO_01_later_lineage

真实后期谱系标记的同预算获取支持实验。ALL_NSCLC 确认集中14名患者具有真正复发/进展后的样本，其中9名有可评价的主肿瘤非主干标记簇参照；另外5名维持未知/无可评价参照，不冒充生物阴性，也不列入恢复率分母。显示开发数据冻结的 BEST_FIXED 经典采集策略，在额外预算10、14、20和新区获取开销2的预定合同下，实际后期样本支持的主肿瘤非主干谱系标记恢复率。三个均值为54.44%、54.44%、77.78%；空心柱、全部9个患者点、患者级2000次 bootstrap95%区间。每患者五次初始区域旋转先平均；九个点是九名患者，而不是额外区域、簇或重复。

原区窄测2、完整WES8和新区域开销均为冻结的假设成本单位，不是实际货币价格或前瞻性新采集。区域按冻结随机顺序开放；后期样本身份仅用于独立评分，未用于选择动作。标记簇至少有3个分支自有变异满足实际测量支持；该指标不证明迁移方向、克隆适应度或原CTMC theta。此图支持固定经典补测预算增长的生物桥接，不支持诊断策略超过最佳固定方案；没有为未注册预算比较添加星号。参考R29的空心柱、实际患者点、端帽及简洁轴线，使用规定配色、Arial、600dpi PDF+SVG。
"""
    return metadata, caption


def plot_g5(out: Path) -> tuple[dict, str]:
    name = "G5_BIO_01_clone_tracking"
    patient_path = BASE / "results/ctdna/PATIENT_RECALL.tsv"
    population_path = BASE / "results/ctdna/POPULATION_SUMMARY.tsv"
    normal_path = BASE / "results/ctdna/NORMAL_FPR_SUMMARY.tsv"
    raw = pd.read_csv(patient_path, sep="\t")
    raw = raw[(raw.split == "confirmation") & (raw.scenario == "INTRINSIC_QC")]
    summaries = pd.read_csv(population_path, sep="\t")
    summaries = summaries[(summaries.population == "ALL197_CT_DNA") &
        (summaries.split == "confirmation") & (summaries.scenario == "INTRINSIC_QC")]
    assert set(summaries.method) == set(METHODS)
    fig = plt.figure(figsize=(5.25, 4.15))
    main = fig.add_axes([.145, .125, .80, .525])
    left = fig.add_axes([.20, .715, .32, .16])
    right = fig.add_axes([.61, .715, .32, .16])
    plot_rows = []
    for method in METHODS:
        curve = []
        for budget in [20, 50, 100]:
            data = raw[(raw.method == method) & (raw.nominal_K == budget)]
            scored = data[data.reference_recall.notna()]
            assert len(data) == 74 and len(scored) == 71
            mean, lo, hi = bootstrap_mean(scored.reference_recall.to_numpy(), f"CTDNA_MAIN_{budget}_{method}")
            frozen_mean = summaries[(summaries.method == method) & (summaries.nominal_K == budget)].recall_mean.iloc[0]
            assert np.isclose(mean, frozen_mean)
            curve.append((budget, mean*100, lo*100, hi*100))
            plot_rows.append(dict(component="curve_mean_ci", method=method, nominal_K=budget,
                patient_id="", n=71, whole_reference_positive_n=14, total_confirmation_n=74,
                with_postoperative_samples_n=52, value=mean, lo=lo, hi=hi, count=np.nan))
            for patient in scored.itertuples(index=False):
                plot_rows.append(dict(component="curve_patient", method=method, nominal_K=budget,
                    patient_id=patient.patient_id, n=71, whole_reference_positive_n=14,
                    total_confirmation_n=74, with_postoperative_samples_n=52,
                    value=patient.reference_recall, lo=np.nan, hi=np.nan, count=np.nan))
        values = np.array(curve)
        main.plot(values[:,0], values[:,1], color=COLORS[method], lw=1.8, zorder=3)
        main.errorbar(values[:,0], values[:,1],
            yerr=np.vstack([values[:,1]-values[:,2], values[:,3]-values[:,1]]),
            fmt="o", markersize=4.5, markerfacecolor=COLORS[method],
            markeredgecolor=BLACK, markeredgewidth=.85, color=COLORS[method],
            ecolor=BLACK, elinewidth=.7, capsize=2.6, capthick=.7, zorder=5)
        # K100 numeric endpoints distinguish genuinely close overlapping means;
        # marker locations and budget distances remain unchanged.
        label_y = {"CLONAL_ONLY": 1.7, "BRANCH_DIVERSE": 21.,
                   "RANDOM_MARKERS": 17.1, "INFORMATION_ADAPTIVE": 12.4}[method]
        main.annotate(f"{values[-1,1]:.1f}", xy=(100,values[-1,1]),
            xytext=(108,label_y), color=COLORS[method], fontsize=7.2,
            va="center", ha="left", arrowprops=dict(arrowstyle="-", color=COLORS[method], lw=.6), zorder=7)
        k50 = raw[(raw.method == method) & (raw.nominal_K == 50)]
        positive = k50[k50.whole_reference_branches_detected > 0]
        assert len(positive) == 14
        branch = bootstrap_mean(positive.reference_recall.to_numpy(), f"CTDNA_POSITIVE50_{method}")
        recurrence = k50[k50.confirmed_recurrence_day.notna()]
        assert len(recurrence) == 39 and recurrence.postoperative_samples.gt(0).sum() == 31
        before = bool_values(recurrence.MRD_before_recurrence)
        detect = bootstrap_mean(before, f"CTDNA_RECURRENCE50_{method}")
        j = METHODS.index(method)
        y = 3-j
        for ax, values_ci in [(left,branch), (right,detect)]:
            mean, lo, hi = np.array(values_ci)*100
            ax.barh(y, mean, height=.47, facecolor=COLORS[method], alpha=.27,
                edgecolor=BLACK, linewidth=.8, zorder=2)
            # Draw outlines separately so their opacity is not diluted by fill.
            ax.barh(y, mean, height=.47, facecolor="none", edgecolor=BLACK, linewidth=.7,zorder=3)
            ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt="none",ecolor=BLACK,
                elinewidth=.65,capsize=2,capthick=.65,zorder=4)
        # Actual14 positive-patient points reproduce the small R04 inset clouds.
        jitter = np.linspace(-.12,.12,len(positive))
        left.scatter(positive.reference_recall.to_numpy()*100, y+jitter,
                     s=5.5,color=COLORS[method],linewidth=0,zorder=5,alpha=.8)
        left.text(109,y,f"{branch[0]*100:.1f}%",ha="left",va="center",fontsize=6.8,color=BLACK)
        right.text(109,y,f"{int(before.sum())}/39",ha="left",va="center",fontsize=6.8,color=BLACK)
        for component, ci, n, count in [("inset_positive_mean_ci",branch,14,np.nan),
                                       ("inset_recurrence_mean_ci",detect,39,int(before.sum()))]:
            plot_rows.append(dict(component=component,method=method,nominal_K=50,
                patient_id="",n=n,whole_reference_positive_n=14,total_confirmation_n=74,
                with_postoperative_samples_n=14 if n==14 else 31,
                value=ci[0],lo=ci[1],hi=ci[2],count=count))
        for patient in positive.itertuples(index=False):
            plot_rows.append(dict(component="inset_positive_patient",method=method,nominal_K=50,
                patient_id=patient.patient_id,n=14,whole_reference_positive_n=14,total_confirmation_n=74,
                with_postoperative_samples_n=14,value=patient.reference_recall,lo=np.nan,hi=np.nan,count=np.nan))
        for patient, value in zip(recurrence.itertuples(index=False),before):
            plot_rows.append(dict(component="inset_recurrence_patient",method=method,nominal_K=50,
                patient_id=patient.patient_id,n=39,whole_reference_positive_n=14,total_confirmation_n=74,
                with_postoperative_samples_n=31,value=value,lo=np.nan,hi=np.nan,count=np.nan))
    main.set_xlim(13, 116)
    main.set_ylim(-2, 33)
    main.set_xticks([20, 50, 100])
    main.set_yticks([0, 10, 20, 30])
    main.set_xlabel("Nominal marker budget",labelpad=4)
    main.set_ylabel("Reference branch score (%)",labelpad=5)
    main.axvline(50,color=BLACK,lw=.9,ls=(0,(3,2)),zorder=1)
    main.text(.015,.98,"n=71; 14 reference-positive",transform=main.transAxes,
              ha="left",va="top",fontsize=7.4)
    tidy(main)
    for ax,title in [(left,"Branch recall (n=14)"),(right,"Before recurrence (n=39)")]:
        ax.set_xlim(-4, 121)
        ax.set_ylim(-.5, 3.65)
        ax.set_xticks([0,50,100])
        ax.set_yticks([])
        ax.set_title(title,fontsize=7.6,pad=3,loc="left")
        ax.set_xlabel("%",fontsize=7,labelpad=1)
        tidy(ax,True)
    fig.text(.565,.925,"K=50",ha="center",va="center",fontsize=7.4)
    handles=[Patch(facecolor=COLORS[m],edgecolor=BLACK,linewidth=.7) for m in METHODS]
    fig.legend(handles,[LABELS[m] for m in METHODS],loc="upper center",
        bbox_to_anchor=(.55,1.),ncol=4,fontsize=8,frameon=False,
        handlelength=1.5,handletextpad=.4,columnspacing=1.2)
    plot_data = BASE / "publication_data/g5" / f"{name}.tsv"
    plot_data.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(plot_rows).to_csv(plot_data,sep="\t",index=False)
    metadata=export(fig,out/"g5"/name)
    metadata.update(panel=name,group="g5",primary_reference_id="R04",
        scientific_message="Surgery-derived nontrunk marker coverage restores branches beyond clonal-only tracking, but adaptive selection does not outperform strong fixed diversity and is not proof of clinical reliability.",
        estimand="Availability-inclusive contracted wholePSP-reference branch score; conditional reference-positive recall and recurrence-preceding detection in separate K50 insets",
        unit_of_analysis="Independent confirmation patient; markers, samples and clones are not independent patients",
        n_definition="74confirmation;52withpostop plasma.71branch-scoreeligible=51withplasma+20missingplasma contractzeros;14whole-referencepositive.39true recurrence=31withplasma+8missingplasma.",
        data_source=[str(patient_path),str(population_path),str(normal_path)],
        source_hashes={str(p):sha(p) for p in [patient_path,population_path,normal_path]},
        plot_data_path=str(plot_data),data_sha256=sha(plot_data),
        adaptations=["R04 top single-row colored rectangle legend, four colored trajectories, black-edged filled circular markers and black capped patient-bootstrap intervals",
            "Two actual-data K50 insets reproduce horizontal bars and reference-positive patient clouds; true recurrence counts replace unsupported stars",
            "Vertical K50 dashed line denotes registered primary marker budget, not a treatment phase; no stage shading",
            "Budget positions retain numeric distances; nearly equal K100 endpoints get numerical labels without moving the data points",
            "No screenshot phone UI/tutorial headings/arrows and no unregistered significance annotations"],
        statistical_annotations="2000patientbootstrap95%intervals; no star/ns annotations",
        review_status="awaiting_actual_export_review",user_approved=False)
    caption="""## G5_BIO_01_clone_tracking

手术组织标记信息与术后非主干克隆追踪的公开计数回放。四个冻结策略完整保留：Clonal=CLONAL_ONLY、Diverse=BRANCH_DIVERSE经典强对照、Random=RANDOM_MARKERS、Adaptive=INFORMATION_ADAPTIVE。主曲线为 ALL197_CT_DNA 确认集、INTRINSIC_QC 合同，在名义标记预算20、50、100下的患者平均 wholePSP参照分支得分；圆点和同色线为均值，黑描边圆点与黑端帽为患者级2000次 bootstrap95%区间。虚线50只标记预登记主预算，不是处理阶段；曲线连接三个实际预算点，不是时间轨迹或平滑拟合。右侧数值为K100均值，解决接近终点的重叠；没有移动点的真实坐标或添加未注册比较星号。

分母必须按合同解释：74名确认患者中52名有术后血浆；71名有分支评分身份，其中51名有血浆、20名缺术后样本按冻结合同贡献0分，而不是71名全部实测。3名未知树保留MRD/容量资料但不冒充0分支真值。14名有wholePSP实际检出的非主干参照分支。K50左侧插图仅展示这14名whole-reference-positive患者的条件恢复率（全部14个真实患者点）：Clonal0%、Diverse83.93%、Random26.79%、Adaptive45.83%。右侧插图为39名具有明确真实复发事件和日期的患者，31名有血浆、8名缺样本按已冻结合同为未检出；复发前检出依次22/39、23/39、20/39、23/39。图中指标是含资料可用性的回放合同得分，不是对缺样本患者进行了真实测量。

标记选择只使用手术时可得组织信息和primary-only参考，不读取术后血浆或未来复发作选择。该实验使用已有完整患者特异面板的处理后UMI计数和误差/QC注释，未重新测序或FASTQ重跑，不证明自然演化theta、耐药机制、治疗获益或可部署小面板的独立误差校准。Adaptive没有超过固定Diverse强基线，整体MRD和复发前检出也不显示其独有优势。

重要边界保留：K50 Clonal有10/74标记容量不足；名义预算相同不等于每患者实际可用标记数完全相等。10个独立确认正常来源中，Adaptive/Diverse平均回放阳性率8.89%，Random10%，Clonal0；非恶性淋巴细胞体细胞突变混杂集中于CRUK0498，病例与错误均保留。因此分支覆盖增加不能直接解释为可靠临床MRD或患者获益。14个参照阳性和39个复发患者与本项目TRACERx组织资料存在患者重叠，不是新增独立临床队列。

参考R04的单行顶部图例、两真实指标插图、彩色线、黑描边圆点、黑色端帽；不同终点保留独立明确分母，不复制无实验意义的阶段阴影或星号。Arial、规定六色内配色、600dpi PDF+SVG，全部源患者与作图数据保留。
"""
    return metadata,caption


def plot_g5_future(out: Path) -> tuple[dict, str]:
    name = "G5_BIO_02_future_clone_tracking"
    patient_path = BASE / "results/ctdna/FUTURE_TISSUE_PATIENT_SCORES.tsv"
    summary_path = BASE / "results/ctdna/FUTURE_TISSUE_SUMMARY.tsv"
    receipt_path = BASE / "results/ctdna/FUTURE_TISSUE_SCORER_RECEIPT.json"
    raw = pd.read_csv(patient_path, sep="\t")
    raw = raw[(raw.split == "confirmation") & (raw.scenario == "INTRINSIC_QC") &
        (raw.nominal_K.eq(50) | raw.method.eq("WHOLE_PSP_REFERENCE")) &
        raw.future_nontrunk_targets.gt(0)].copy()
    summary = pd.read_csv(summary_path, sep="\t")
    summary = summary[(summary.population == "ALL197_CT_DNA") &
        (summary.split == "confirmation") & (summary.scenario == "INTRINSIC_QC") &
        (summary.nominal_K.eq(50) | summary.method.eq("WHOLE_PSP_REFERENCE"))]
    methods = METHODS + ["WHOLE_PSP_REFERENCE"]
    assert set(raw.method) == set(methods)
    assert raw.groupby("method").patient_id.nunique().eq(10).all()
    assert summary.future_target_patient_n.eq(10).all()
    assert summary.future_target_plasma_available_patient_n.eq(10).all()
    assert summary.true_recurrence_date_target_patient_n.eq(10).all()
    assert summary.future_target_n.eq(20).all()
    ids = sorted(raw.patient_id.unique())
    # Appearance-only subsample: no outcome is used, and all10 remain in every
    # mean/interval and the delivered plot data. The same6 are shown for every
    # policy and both endpoints, chosen once by this fixed patient-ID hash.
    display_ids = sorted(ids, key=lambda p: sha_text(name + "|display|" + p))[:6]
    offsets = dict(zip(sorted(display_ids), np.linspace(-.034, .034, 6)))
    labels = {**LABELS, "WHOLE_PSP_REFERENCE": "Whole PSP"}
    colors = {**COLORS, "WHOLE_PSP_REFERENCE": PALETTE[4]}
    fig, ax = plt.subplots(figsize=(4.6, 2.65))
    fig.subplots_adjust(left=.16, right=.985, bottom=.16, top=.84)
    endpoints = [
        ("Postoperative", "future_target_recall", "future_target_recall_mean"),
        ("Before recurrence", "future_target_recall_before_recurrence",
         "future_target_recall_before_recurrence_mean")]
    centers = [0., 1.1]
    width, spacing = .132, .165
    plot_rows = []
    for group, (label, column, summary_column) in enumerate(endpoints):
        for j, method in enumerate(methods):
            data = raw[raw.method.eq(method)].sort_values("patient_id")
            assert len(data) == 10 and data[column].notna().all()
            mean, lo, hi = bootstrap_mean(data[column].to_numpy(), f"FUTURE_TISSUE50_{column}_{method}")
            expected = summary[summary.method.eq(method)][summary_column].iloc[0]
            assert np.isclose(mean, expected)
            x = centers[group] + (j-2) * spacing
            color = colors[method]
            whole = method == "WHOLE_PSP_REFERENCE"
            ax.bar(x, mean*100, width=width, facecolor="white" if whole else color,
                   edgecolor=color if whole else BLACK, linewidth=1.2 if whole else .8, zorder=2)
            ax.errorbar(x, mean*100, yerr=[[(mean-lo)*100],[(hi-mean)*100]],
                fmt="none", ecolor=BLACK, elinewidth=.75, capsize=2., capthick=.75, zorder=4)
            visible = data[data.patient_id.isin(display_ids)]
            xx = np.array([x + offsets[p] for p in visible.patient_id])
            ax.scatter(xx, visible[column].to_numpy()*100, s=12, facecolor="white",
                edgecolor=color if whole else BLACK, linewidth=.65, zorder=5, clip_on=False)
            plot_rows.append(dict(component="mean_ci", endpoint=column, method=method,
                nominal_K=0 if whole else 50, extra_information_reference=whole,
                patient_id="", n=10, display_n=6, value=mean, lo=lo, hi=hi, x=x,
                displayed=False, target_n=20))
            for patient in data.itertuples(index=False):
                shown = patient.patient_id in display_ids
                plot_rows.append(dict(component="patient", endpoint=column, method=method,
                    nominal_K=0 if whole else 50, extra_information_reference=whole,
                    patient_id=patient.patient_id, n=10, display_n=6,
                    value=getattr(patient, column), lo=np.nan, hi=np.nan,
                    x=x+offsets[patient.patient_id] if shown else np.nan,
                    displayed=shown, target_n=patient.future_nontrunk_targets))
    ax.set_xlim(-.56, 1.66)
    ax.set_ylim(0, 107)
    ax.set_yticks([0,25,50,75,100])
    ax.set_xticks(centers,[e[0] for e in endpoints])
    ax.set_ylabel("Future lineage target recall (%)",labelpad=5)
    tidy(ax)
    ax.text(.5,1.025,"K=50; n=10 future-target patients",transform=ax.transAxes,
        ha="center",va="bottom",fontsize=7.4)
    handles = [Patch(facecolor="white" if m=="WHOLE_PSP_REFERENCE" else colors[m],
        edgecolor=colors[m] if m=="WHOLE_PSP_REFERENCE" else BLACK,linewidth=.8) for m in methods]
    fig.legend(handles,[labels[m] for m in methods],loc="upper center",ncol=5,
        bbox_to_anchor=(.565,.997),fontsize=7.2,frameon=False,handlelength=1.15,
        handletextpad=.30,columnspacing=.65)
    fig.text(.575,.015,"Whole PSP: extra-information reference",ha="center",va="bottom",fontsize=7)
    plot_data = BASE / "publication_data/g5" / f"{name}.tsv"
    pd.DataFrame(plot_rows).to_csv(plot_data,sep="\t",index=False)
    display_path = plot_data.with_name(name+"_POINT_DISPLAY.json")
    display_path.write_text(json.dumps(dict(rule="lowest6SHA256(figure_id+'|display|'+patient_id)",
        all_patient_ids=ids,display_patient_ids=display_ids,outcome_used=False,
        all_patients_in_statistics=True),ensure_ascii=False,indent=2)+"\n")
    metadata = export(fig,out/"g5"/name)
    metadata.update(panel=name,group="g5",primary_reference_id="R21",
        scientific_message="Frozen diversity-based surgery marker selection can recover actual future-tissue-supported nontrunk lineages in postoperative plasma and before recurrence, with all strategies retained; not adaptive superiority.",
        estimand="Patient-mean recovery of primary nontrunk multivariant marker lineages independently supported by actual future tissue; second endpoint restricts plasma evidence to before actual recurrence",
        unit_of_analysis="Ten independent confirmation patients, not20targets or multiple plasma samples",
        n_definition="74confirmation availability;12have paired future tissue;10have20evaluable future nontrunk targets and postoperative plasma;2paired but reference-ineligible remain unknown/no-reference outside target denominator",
        data_source=[str(patient_path),str(summary_path),str(receipt_path)],
        source_hashes={str(p):sha(p) for p in [patient_path,summary_path,receipt_path]},
        plot_data_path=str(plot_data),data_sha256=sha(plot_data),
        display_rule_path=str(display_path),display_rule_sha256=sha(display_path),
        adaptations=["R21 compact narrow grouped colored bars with black outlines, open patient circles and black capped intervals",
            "Two genuinely distinct biological endpoints replace mutation categories; no unregistered vertical P-value annotations",
            "Four equal nominal K50 strategies retained; hollow orange WholePSP is an extra-information reference, not an equal-budget control",
            "All10 patients determine means/intervals and delivered data; same6 ID-hash patients displayed for all10bars, without outcome-based selection"],
        statistical_annotations="Secondary descriptive endpoint;2000patientbootstrap95%intervals; no star/ns annotations or newly invented P-values",
        review_status="awaiting_actual_export_review",user_approved=False)
    caption = """## G5_BIO_02_future_clone_tracking

真正后期组织支持的谱系目标追踪。确认集74名患者中12名具有独立的复发/进展后组织；其中10名有可评价的20个主肿瘤非主干谱系标记目标，并均有术后血浆和真实复发日期。另外2名有配对组织但无可评价非主干参照，保留未知/无参照身份，不作为阴性。未来目标必须有至少3个分支自有变异在真实未来区域获得支持；未来组织只用于该新增次级评分，不改变手术时已冻结的面板选择。左组为术后任一血浆的未来目标恢复率，右组只计真实复发前的血浆证据。每患者先计算自身目标恢复比例，再对完整10名患者求均值；不是20个目标或多份血浆作为独立样本。

四个名义K50策略完整保留。术后/复发前患者均值分别为Clonal0%/0%、Diverse63.33%/33.33%、Random20%/6.67%、Adaptive23.33%/6.67%。空心橙色WholePSP整面板参照为63.33%/33.33%，属于额外信息参照，未与K50策略等成本；数值相同不能宣称节省75%费用或验证一款真实重新测序的小面板。3个未来目标低于原PSP多标记容量，仍在分母保留，不按能否被PSP检测剔除。

窄分组柱、黑描边、空心患者点和黑色端帽参照R21。均值与2000次患者bootstrap95%区间均由完整10名患者决定；为减少散点密度，每组仅显示由固定患者ID SHA256规则选出的同6名患者，所有10名患者和显示标记均在作图数据中保留。这是新增描述性生物桥接终点，没有注册比较检验，因此无星号或P值。此结果支持利用经典分支多样性获取策略选择可追踪的真实后期谱系目标；不支持Adaptive优于Diverse、不验证CTMC theta、迁移方向或治疗获益，也不改变正常来源体细胞突变混杂和处理后UMI回放的临床可靠性边界。Arial、规定六色、600dpi PDF+SVG。
"""
    return metadata,caption


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=BASE/"staging")
    args=parser.parse_args()
    font_manager.findfont("Arial",fallback_to_default=False)
    apply_publication_style({"font.family":"Arial","font.size":8,
        "axes.labelsize":8.5,"axes.titlesize":8,"pdf.fonttype":42,"svg.fonttype":"none"})
    manifest,captions=[],[]
    for plotter in (plot_g4,plot_g5,plot_g5_future):
        metadata,caption=plotter(args.output_dir)
        metadata.update(plotting_code=str(Path(__file__)),code_sha256=sha(Path(__file__)),
            palette=PALETTE,reference_source=str(REFERENCES[metadata["primary_reference_id"]]),
            reference_sha256=sha(REFERENCES[metadata["primary_reference_id"]]))
        manifest.append(metadata)
        captions.append(caption)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"FIGURE_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    (BASE/"publication_data/FIGURE_LEGENDS_CN.md").write_text("\n".join(captions))
    (args.output_dir/"ACTUAL_COMMAND.txt").write_text(".venv/bin/python paper_experiments_v1/biological_closure_20261009/src/plot_biological.py\n")
    print("Three PDF+SVG support figures staged; awaiting actual export review; no formal release change.")


if __name__=="__main__":
    main()
