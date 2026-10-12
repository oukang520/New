# G2当前正式图与实验代码对应

只以本包`paper_experiments_v1/g2/final_figures`的九张图选择实验。下面分开列出实验计算、结果汇总和实际作图代码；并非用绘图脚本代替实验代码。九张图的对应关系均已按当前正式数据与源脚本查证。当前PDF/SVG直接保留原文件，本次没有重跑或重绘。

| 当前单图 | 实验计算 | 汇总或推导 | 作图入口 |
|---|---|---|---|
| [G2_01](../paper_experiments_v1/g2/final_figures/G2_01.svg) | [g2_run.py](../paper_experiments_v1/g2/src/g2_run.py) | [g2_summarize.py](../paper_experiments_v1/g2/src/g2_summarize.py)、[g2_derive.py](../paper_experiments_v1/g2/src/g2_derive.py) | [redraw.py](../paper_experiments_v1/g2/figure_revision_v2/redraw.py) |
| [G2_02](../paper_experiments_v1/g2/final_figures/G2_02.svg) | [run_g2.py](../paper_experiments_v1/closure_r8/src/run_g2.py)、[refine_profile_v2.py](../paper_experiments_v1/closure_r8/src/refine_profile_v2.py)、[g2_fresh_confirmation.py](../paper_experiments_v1/closure_r8/src/g2_fresh_confirmation.py) | [summarize.py](../paper_experiments_v1/closure_r8/src/summarize.py) | [plot.py](../paper_experiments_v1/closure_r8/publication/plot.py)、[render_g2.py](../paper_experiments_v1/figure_delivery/bar_bracket_audit_20260929/render_g2.py) |
| [G2_03](../paper_experiments_v1/g2/final_figures/G2_03.svg) | [g2_run.py](../paper_experiments_v1/g2/src/g2_run.py) | [g2_summarize.py](../paper_experiments_v1/g2/src/g2_summarize.py)、[g2_derive.py](../paper_experiments_v1/g2/src/g2_derive.py) | [redraw.py](../paper_experiments_v1/g2/figure_revision_v3/redraw.py)、[redraw.py](../paper_experiments_v1/figure_delivery/reference_revision_20260925/redraw.py) |
| [G2_04](../paper_experiments_v1/g2/final_figures/G2_04.svg) | [g2_omhn.py](../paper_experiments_v1/g2/src/g2_omhn.py) | [g2_plot.py](../paper_experiments_v1/g2/src/g2_plot.py) | [redraw.py](../paper_experiments_v1/g2/figure_revision_v3/redraw.py) |
| [G2_05](../paper_experiments_v1/g2/final_figures/G2_05.svg) | [g2_run.py](../paper_experiments_v1/g2/src/g2_run.py) | [g2_summarize.py](../paper_experiments_v1/g2/src/g2_summarize.py)、[g2_derive.py](../paper_experiments_v1/g2/src/g2_derive.py) | [redraw.py](../paper_experiments_v1/g2/figure_revision_v2/redraw.py) |
| [G2_06](../paper_experiments_v1/g2/final_figures/G2_06.svg) | [g2_run.py](../paper_experiments_v1/g2/src/g2_run.py) | [g2_summarize.py](../paper_experiments_v1/g2/src/g2_summarize.py)、[g2_derive.py](../paper_experiments_v1/g2/src/g2_derive.py) | [redraw.py](../paper_experiments_v1/g2/figure_revision_v2/redraw.py) |
| [G2_07](../paper_experiments_v1/g2/final_figures/G2_07.svg) | [g2_omhn.py](../paper_experiments_v1/g2/src/g2_omhn.py) | 直接使用逐重复估计 | [redraw.py](../paper_experiments_v1/g2/figure_revision_v2/redraw.py) |
| [G2_R2_sign_strong](../paper_experiments_v1/g2/final_figures/G2_R2_sign_strong.svg) | [run_simulations.py](../paper_experiments_v1/reviewer_audit/redesign_r2/run_simulations.py) | [run_simulations.py](../paper_experiments_v1/reviewer_audit/redesign_r2/run_simulations.py) | [plot_r2.py](../paper_experiments_v1/reviewer_audit/redesign_r2/plot_r2.py) |
| [G2_R2_sign_weak](../paper_experiments_v1/g2/final_figures/G2_R2_sign_weak.svg) | [run_simulations.py](../paper_experiments_v1/reviewer_audit/redesign_r2/run_simulations.py) | [run_simulations.py](../paper_experiments_v1/reviewer_audit/redesign_r2/run_simulations.py) | [plot_r2.py](../paper_experiments_v1/reviewer_audit/redesign_r2/plot_r2.py) |

G2_01、03、05、06来自原冻结核心模拟。G2_02使用当前R8计算与profile细化；它与早期核心模拟是不同的正式终点。`g2_fresh_confirmation.py`提供独立确认记录，不能把确认结果混作主图原始评价。

G2_04与07使用官方oMHN匹配似然的800组配对估计。04需要`g2_plot.py`的`matched()`函数生成`B6_matched_summary.tsv`；该文件虽名为plot，内部含必要的统计汇总。07直接对四份`B6_matched_cell*.tsv`中的估计做配对pivot；`g2_b6_pairs.py`是同实验的额外配对指标汇总，不是07散点数据的直接生产者。

两张R2符号图由`reviewer_audit/redesign_r2/run_simulations.py`的`g2()`与`metrics()`产生；对应强/弱观察条件。这些源脚本在R2目录仍服务当前正式图，因此保留代码和有效输入。

各图的数据位于[正式作图数据](../paper_experiments_v1/g2/final_results/plot_data)，更详细的函数、终点和来源说明见[逐图机器清单](FIGURE_TO_CODE.json)。源文件可能包含其他绘图入口；不可整模块运行旧绘图`main()`来恢复本次正式图选择。

G2数值核心在[g2_engine.py](../paper_experiments_v1/g2/src/g2_engine.py)、[g2_certified_sign.py](../paper_experiments_v1/g2/src/g2_certified_sign.py)与[lbfgsb_direct.py](../paper_experiments_v1/g2/src/lbfgsb_direct.py)。原`ctmc_fast_v3.dylib`已收录；本机概率与导数检查通过。缺失的是原v3 C源文件，而非上述Python实验代码，跨平台重建范围见[复现说明](REPRODUCE_CN.md)。
