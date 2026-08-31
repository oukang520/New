"""Run experiments 1 and 2 for the selected Rel-ObsTQ-MHN cohorts.

Experiment 1:
    Data preparation, cancer-specific driver panel construction, event matrices,
    state tables, QC metrics, and publication-style QC figures.

Experiment 2:
    Stage/progression definition sensitivity before MHN training. This builds
    clinical, metastasis, mutation-burden, and pathway-burden schemes and
    quantifies their distributions, agreement, and state-space sparsity.

MHN-derived progression scores and R* ranking comparisons are intentionally
left as pending outputs because they require later MHN training.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

import figure_style
from scipy.stats import chi2_contingency


STAGE_ORDER = ["early", "local_advanced", "primary", "metastatic", "unknown"]
STAGE_COLORS = {
    "early": "#228833",
    "local_advanced": "#EEAA33",
    "primary": "#4477AA",
    "metastatic": "#CC6677",
    "unknown": "#BBBBBB",
}
SCHEME_COLORS = {
    "clinical_stage": "#4477AA",
    "metastasis_status": "#CC6677",
    "mutation_burden": "#228833",
    "pathway_burden": "#AA3377",
}
PASSENGER_EXACT = {
    "TTN",
    "MUC16",
    "MUC4",
    "MUC2",
    "MUC19",
    "OBSCN",
    "SYNE1",
    "HMCN1",
    "XIRP2",
    "CSMD1",
    "CSMD2",
    "CSMD3",
    "RYR1",
    "RYR2",
    "RYR3",
}
PASSENGER_PREFIXES = ("DNAH", "MUC", "CSMD", "RYR", "ANKRD")


def setup_logging(root: Path) -> None:
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_dir / "experiments_01_02.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def configure_plotting(config: dict) -> None:
    figure_style.configure_matplotlib(config)


def save_figure(
    fig: plt.Figure,
    base_path: Path,
    dpi: int,
    panel_names: list[str] | None = None,
) -> None:
    figure_style.save_figure_panels(
        fig,
        base_path,
        {"plot": {"dpi": dpi}},
        dpi=dpi,
        panel_names=panel_names,
    )


def save_square_figure(
    fig: plt.Figure,
    base_path: Path,
    dpi: int,
    panel_names: list[str] | None = None,
) -> None:
    figure_style.save_figure_panels(
        fig,
        base_path,
        {"plot": {"dpi": dpi}},
        dpi=dpi,
        panel_names=panel_names,
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def cohort_short_label(dataset_name: str) -> str:
    if dataset_name.startswith("AACR_"):
        return dataset_name.replace("AACR_", "", 1)
    return dataset_name


def read_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(input_dir / "analysis_metadata.csv", dtype=str, low_memory=False)
    mutations = pd.read_csv(input_dir / "mutations_long.csv", dtype=str, low_memory=False)
    metadata["analysis_id"] = metadata["analysis_id"].astype(str)
    mutations["analysis_id"] = mutations["analysis_id"].astype(str)
    mutations["gene"] = mutations["gene"].fillna("").astype(str).str.strip().str.upper()
    mutations = mutations[(mutations["gene"] != "") & mutations["analysis_id"].isin(set(metadata["analysis_id"]))]
    mutations = mutations.drop_duplicates(["analysis_id", "gene"])
    return metadata, mutations


def is_likely_passenger(gene: str) -> bool:
    if gene in PASSENGER_EXACT:
        return True
    return any(gene.startswith(prefix) for prefix in PASSENGER_PREFIXES)


def pathway_lookup(pathways: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for pathway, genes in pathways.items():
        for gene in genes:
            lookup.setdefault(str(gene).upper(), pathway)
    return lookup


def construct_event_panels(
    metadata: pd.DataFrame,
    mutations: pd.DataFrame,
    dataset_config: dict,
    panel_sizes: list[int],
    minimum_event_samples: int,
) -> tuple[dict[int, list[str]], pd.DataFrame]:
    n = metadata["analysis_id"].nunique()
    support_floor = max(minimum_event_samples, int(math.ceil(0.01 * n)))
    support = (
        mutations.groupby("gene")["analysis_id"]
        .nunique()
        .sort_values(ascending=False)
        .rename("sample_count")
    )
    frequency = support / max(n, 1)
    priority = [str(gene).upper() for gene in dataset_config.get("driver_priority", [])]
    priority_rank = {gene: idx + 1 for idx, gene in enumerate(priority)}
    pathways = pathway_lookup(dataset_config.get("pathways", {}))

    selected_rank: list[str] = []
    for gene in priority:
        if gene in support.index and int(support[gene]) >= support_floor:
            selected_rank.append(gene)

    for gene in support.index:
        if gene in selected_rank or int(support[gene]) < support_floor:
            continue
        if is_likely_passenger(gene):
            continue
        selected_rank.append(gene)

    for gene in support.index:
        if gene not in selected_rank and int(support[gene]) >= support_floor:
            selected_rank.append(gene)

    max_panel = max(panel_sizes)
    selected_rank = selected_rank[:max_panel]
    panels = {size: selected_rank[: min(size, len(selected_rank))] for size in panel_sizes}

    rows = []
    for gene in support.index:
        row = {
            "event": gene,
            "sample_count": int(support[gene]),
            "frequency": round(float(frequency[gene]), 6),
            "is_curated_driver": gene in priority_rank,
            "driver_priority_rank": priority_rank.get(gene, ""),
            "pathway": pathways.get(gene, "Other"),
            "likely_passenger_or_size_related": is_likely_passenger(gene),
            "minimum_support_required": support_floor,
            "panel_rank": selected_rank.index(gene) + 1 if gene in selected_rank else "",
            "selection_reason": (
                "curated_cancer_driver"
                if gene in priority_rank
                else ("frequency_fill" if gene in selected_rank else "not_selected")
            ),
        }
        for size in panel_sizes:
            row[f"selected_p{size}"] = gene in panels[size]
        rows.append(row)
    return panels, pd.DataFrame(rows)


def build_matrix(
    metadata: pd.DataFrame, mutations: pd.DataFrame, events: list[str]
) -> pd.DataFrame:
    ids = metadata["analysis_id"].drop_duplicates().astype(str)
    work = mutations[mutations["gene"].isin(events)][["analysis_id", "gene"]].drop_duplicates()
    work["_value"] = 1
    if work.empty:
        matrix = pd.DataFrame(0, index=ids, columns=events)
    else:
        matrix = work.pivot_table(
            index="analysis_id", columns="gene", values="_value", aggfunc="max", fill_value=0
        )
        matrix = matrix.reindex(index=ids, columns=events, fill_value=0).astype(int)
    matrix.index.name = "analysis_id"
    return matrix.reset_index()


def normalize_clinical_stage(series: pd.Series) -> pd.Series:
    values = series.fillna("unknown").astype(str).str.strip().str.lower()
    out = pd.Series("unknown", index=series.index, dtype=object)
    out[values.eq("early")] = "early"
    out[values.eq("local_advanced")] = "local_advanced"
    out[values.eq("primary")] = "primary"
    out[values.eq("metastatic")] = "metastatic"
    return out


def normalize_metastasis(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip().str.lower()
    out = pd.Series("unknown", index=series.index, dtype=object)
    out[values.str.contains("metast|distant", regex=True)] = "metastatic"
    out[values.str.contains("primary tumour|primary tumor|primary", regex=True)] = "primary"
    return out


def burden_groups(series: pd.Series, prefix: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    try:
        binned = pd.qcut(values, q=3, labels=False, duplicates="drop")
        if binned.nunique(dropna=True) >= 3:
            mapping = {0: f"{prefix}_low", 1: f"{prefix}_mid", 2: f"{prefix}_high"}
            return binned.map(mapping).fillna(f"{prefix}_unknown")
    except ValueError:
        pass

    positive = values[values > 0]
    median_positive = float(positive.median()) if not positive.empty else 0
    out = pd.Series(f"{prefix}_low", index=values.index, dtype=object)
    out[(values > 0) & (values <= median_positive)] = f"{prefix}_mid"
    out[values > median_positive] = f"{prefix}_high"
    return out


def build_state_table(
    metadata: pd.DataFrame,
    matrix: pd.DataFrame,
    stage_assignment: pd.Series,
    events: list[str],
    min_state_count: int,
    stage_column_name: str = "stage_group",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = metadata.merge(matrix, on="analysis_id", how="left")
    work[stage_column_name] = stage_assignment.reindex(metadata.index).values
    for event in events:
        work[event] = pd.to_numeric(work[event], errors="coerce").fillna(0).astype(int)
    work["event_count"] = work[events].sum(axis=1).astype(int)
    active = work[events].apply(
        lambda row: "+".join(sorted(row.index[row.astype(int).eq(1)].tolist())) or "WT",
        axis=1,
    )
    work["genotype_signature"] = active
    work["state_id"] = work[stage_column_name].astype(str) + "::" + active
    counts = work["state_id"].value_counts()
    work["state_count"] = work["state_id"].map(counts).astype(int)
    work["state_count_flag"] = np.where(
        work["state_count"] >= min_state_count, "valid_state", "rare_state"
    )
    work["usable_for_mhn"] = True
    work["usable_for_relobstq"] = (
        work[stage_column_name].ne("unknown") & work["state_count_flag"].eq("valid_state")
    )
    occupancy = (
        work.groupby(
            [stage_column_name, "genotype_signature", "state_id", "state_count_flag"],
            dropna=False,
        )
        .size()
        .reset_index(name="state_count")
        .sort_values(["state_count", "state_id"], ascending=[False, True])
    )
    occupancy["occupancy_fraction"] = occupancy["state_count"] / len(work)
    columns = [
        "analysis_id",
        "patient_id",
        "sample_id",
        "dataset_name",
        "cancer_code",
        "cancer_type",
        "cancer_type_detailed",
        "stage_raw",
        stage_column_name,
        "metastasis_status",
        "genotype_signature",
        "event_count",
        "state_id",
        "state_count",
        "state_count_flag",
        "usable_for_mhn",
        "usable_for_relobstq",
        "survival_time",
        "survival_event",
        "age",
        "sex",
    ]
    return work[columns], occupancy


def stage_event_frequency(
    matrix: pd.DataFrame, assignments: pd.Series, events: list[str]
) -> pd.DataFrame:
    work = matrix.copy()
    work["stage"] = assignments.values
    result = work.groupby("stage")[events].mean()
    order = [stage for stage in STAGE_ORDER if stage in result.index]
    remaining = [stage for stage in result.index if stage not in order]
    return result.reindex(order + sorted(remaining))


def cramer_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b)
    if table.empty or min(table.shape) < 2:
        return np.nan
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(math.sqrt(phi2corr / denom)) if denom > 0 else np.nan


def event_pathway_burden(
    mutations: pd.DataFrame, metadata: pd.DataFrame, pathways: dict[str, list[str]]
) -> pd.Series:
    gene_to_pathways: dict[str, set[str]] = {}
    for pathway, genes in pathways.items():
        for gene in genes:
            gene_to_pathways.setdefault(str(gene).upper(), set()).add(pathway)
    selected = mutations[mutations["gene"].isin(gene_to_pathways)].copy()
    if selected.empty:
        return pd.Series(0, index=metadata.index)
    selected["pathway"] = selected["gene"].map(
        lambda gene: sorted(gene_to_pathways.get(gene, {"Other"}))[0]
    )
    counts = selected.drop_duplicates(["analysis_id", "pathway"]).groupby("analysis_id")[
        "pathway"
    ].nunique()
    return metadata["analysis_id"].map(counts).fillna(0).astype(int)


def plot_experiment1_overview(
    dataset_name: str,
    display_name: str,
    metadata: pd.DataFrame,
    mutations: pd.DataFrame,
    matrix: pd.DataFrame,
    events: list[str],
    stage_freq: pd.DataFrame,
    output_base: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
    ax = axes[0, 0]
    counts = [
        len(metadata),
        mutations["analysis_id"].nunique(),
        int(normalize_clinical_stage(metadata["stage_group"]).ne("unknown").sum()),
        int(matrix[events].sum(axis=1).gt(0).sum()),
    ]
    labels = ["All tumor samples", "With functional mutation", "Known progression state", "≥1 panel event"]
    y = np.arange(len(labels))
    bars = ax.barh(y, counts, color=["#6B6B6B", "#4477AA", "#228833", "#AA3377"])
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Analysis units")
    ax.set_title("Cohort retention")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {count:,}", va="center", fontsize=7)
    sns.despine(ax=ax)
    panel_label(ax, "A")

    ax = axes[0, 1]
    event_counts = matrix.set_index("analysis_id")[events].sum().sort_values()
    ax.barh(event_counts.index, event_counts.values, color="#4477AA")
    ax.set_xlabel("Samples with event")
    ax.set_ylabel("")
    ax.set_title("Cancer-specific event panel")
    ax.tick_params(axis="y", labelsize=6.5)
    sns.despine(ax=ax)
    panel_label(ax, "B")

    ax = axes[1, 0]
    heat = stage_freq.loc[:, events]
    sns.heatmap(
        heat,
        ax=ax,
        cmap=sns.light_palette("#B2182B", as_cmap=True),
        vmin=0,
        vmax=max(0.01, float(heat.to_numpy().max())),
        cbar_kws={"label": "Event frequency", "shrink": 0.75},
        linewidths=0.25,
        linecolor="white",
    )
    ax.set_xlabel("Event")
    ax.set_ylabel("Progression state")
    ax.set_title("Stage-specific event frequencies")
    ax.tick_params(axis="x", rotation=55, labelsize=6.5)
    ax.tick_params(axis="y", rotation=0)
    panel_label(ax, "C")

    ax = axes[1, 1]
    per_sample = matrix[events].sum(axis=1)
    bins = np.arange(-0.5, max(3, int(per_sample.max())) + 1.5, 1)
    ax.hist(per_sample, bins=bins, color="#EEAA33", edgecolor="white", linewidth=0.5)
    ax.axvline(per_sample.median(), color="#222222", linestyle="--", linewidth=1)
    ax.set_xlabel("Panel events per sample")
    ax.set_ylabel("Samples")
    ax.set_title(f"Mutation burden (median={per_sample.median():.1f})")
    sns.despine(ax=ax)
    panel_label(ax, "D")

    fig.suptitle(f"{display_name}: Experiment 1 data preparation", fontweight="bold")
    save_figure(
        fig,
        output_base,
        dpi,
        panel_names=[
            "cohort_retention",
            "cancer_specific_event_panel",
            "stage_specific_event_frequencies",
            "mutation_burden",
        ],
    )


def plot_oncoprint(
    display_name: str,
    metadata: pd.DataFrame,
    matrix: pd.DataFrame,
    events: list[str],
    max_samples_per_stage: int,
    output_base: Path,
    dpi: int,
    include_title: bool = True,
) -> None:
    stages = normalize_clinical_stage(metadata["stage_group"])
    work = metadata[["analysis_id"]].copy()
    work["stage"] = stages
    work = work.merge(matrix, on="analysis_id", how="left")
    work["event_count"] = work[events].sum(axis=1)
    selected = []
    stage_order = [stage for stage in STAGE_ORDER if stage in set(work["stage"])]
    for stage in stage_order:
        part = work[work["stage"] == stage].sort_values(
            ["event_count", "analysis_id"], ascending=[False, True]
        )
        selected.append(part.head(max_samples_per_stage))
    shown = pd.concat(selected, ignore_index=True) if selected else work.head(0)
    if shown.empty:
        return

    data = shown[events].to_numpy(dtype=int).T
    fig = plt.figure(figsize=(12, 5.9))
    grid_top = 0.80 if include_title else 0.90
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.32, 5],
        hspace=0.08,
        left=0.09,
        right=0.98,
        bottom=0.10,
        top=grid_top,
    )
    ax_stage = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0])

    stage_codes = np.array([stage_order.index(s) for s in shown["stage"]])[None, :]
    stage_cmap = mcolors.ListedColormap([STAGE_COLORS[s] for s in stage_order])
    ax_stage.imshow(stage_codes, aspect="auto", cmap=stage_cmap, interpolation="nearest")
    ax_stage.set_xticks([])
    ax_stage.set_yticks([])
    for spine in ax_stage.spines.values():
        spine.set_visible(False)

    cmap = mcolors.ListedColormap(["#F2F2F2", "#2B6CB0"])
    ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(events)), events)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xticks([])
    ax.set_xlabel(f"Selected samples (up to {max_samples_per_stage} per progression state)")
    if include_title:
        fig.suptitle(
            f"{display_name}: cancer-specific oncoprint",
            y=0.88,
            fontsize=10.5,
            fontweight="bold",
        )
    for idx in range(1, len(shown)):
        if shown.loc[idx, "stage"] != shown.loc[idx - 1, "stage"]:
            ax.axvline(idx - 0.5, color="white", linewidth=1.5)
            ax_stage.axvline(idx - 0.5, color="white", linewidth=1.5)
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=STAGE_COLORS[s], label=s, markersize=7)
        for s in stage_order
    ]
    fig.legend(
        handles=handles,
        title="Progression state" if include_title else None,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985 if include_title else 0.975),
        ncol=len(handles),
        handletextpad=0.4,
        columnspacing=1.4,
    )
    save_figure(
        fig,
        output_base,
        dpi,
        panel_names=[
            "progression_state_definitions",
            "state_space_sensitivity",
            "pairwise_scheme_agreement",
            "pre_mhn_sensitivity_boundary",
        ],
    )


def plot_combined_e1_overview(
    dataset_names: list[str],
    experiment_config: dict,
    result_root: Path,
    output_base: Path,
    dpi: int,
) -> None:
    colors = figure_style.colors(experiment_config)
    cat = figure_style.categorical_palette(experiment_config)
    text_primary = colors.get("text", {}).get("primary", "#263238")
    text_secondary = colors.get("text", {}).get("secondary", "#4E5A5E")
    grid_color = colors.get("text", {}).get("grid", "#E6E6E6")
    display = {
        dataset: experiment_config["datasets"][dataset].get("short_name", cohort_short_label(dataset))
        for dataset in dataset_names
    }
    retention_rows = []
    stage_rows = []
    burden_rows = []
    panel_frequency = []
    panel_annotations = []

    for dataset in dataset_names:
        tables = (
            result_root
            / dataset
            / "experiment_01_data_preparation"
            / "tables"
        )
        inclusion = pd.read_csv(tables / "sample_inclusion_summary.csv")
        all_units = int(
            inclusion.loc[inclusion["step"].eq("All tumor analysis units"), "n"].iloc[0]
        )
        step_map = {
            "Functional mutation": "With ≥1 functional mutation",
            "Known state": "Known clinical/progression state",
            "≥1 panel event": "With ≥1 p15 event",
            "Valid state": "In valid p15 states",
        }
        for short_label, source_label in step_map.items():
            count = int(inclusion.loc[inclusion["step"].eq(source_label), "n"].iloc[0])
            retention_rows.append(
                {
                    "dataset": display[dataset],
                    "criterion": short_label,
                    "fraction": count / all_units,
                }
            )

        clinical = pd.read_csv(tables / "clinical_clean.csv", dtype=str)
        stage_counts = clinical["stage_group_experiment1"].value_counts()
        for stage, count in stage_counts.items():
            stage_rows.append(
                {
                    "dataset": display[dataset],
                    "stage": stage,
                    "fraction": count / len(clinical),
                }
            )

        burden = pd.read_csv(tables / "sample_event_counts_p15.csv")
        burden_rows.extend(
            {
                "dataset": display[dataset],
                "event_count": int(value),
            }
            for value in burden["event_count"]
        )

        panel = pd.read_csv(tables / "main_event_panel_p15.csv").sort_values(
            "panel_rank"
        )
        panel_frequency.append(panel["frequency"].astype(float).to_numpy())
        panel_annotations.append(panel["event"].astype(str).to_numpy())

    retention_df = pd.DataFrame(retention_rows)
    stage_df = pd.DataFrame(stage_rows)
    burden_df = pd.DataFrame(burden_rows)
    dataset_labels = [display[dataset] for dataset in dataset_names]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2))
    fig.subplots_adjust(
        left=0.080,
        right=0.965,
        bottom=0.085,
        top=0.895,
        wspace=0.300,
        hspace=0.380,
    )

    ax = axes[0, 0]
    criteria = retention_df["criterion"].drop_duplicates().tolist()
    criterion_colors = [
        cat.get("sky_blue", "#B2E6FD"),
        cat.get("sage", "#B8D2CC"),
        cat.get("lavender", "#B5AED5"),
        cat.get("pale_yellow", "#FEEBB9"),
    ]
    x = np.arange(len(dataset_labels))
    width = 0.18
    offsets = (np.arange(len(criteria)) - (len(criteria) - 1) / 2) * width
    retention_lookup = retention_df.set_index(["dataset", "criterion"])["fraction"]
    for idx, criterion in enumerate(criteria):
        values = [float(retention_lookup.get((dataset, criterion), np.nan)) for dataset in dataset_labels]
        ax.bar(
            x + offsets[idx],
            values,
            width=width * 0.92,
            color=criterion_colors[idx],
            edgecolor=text_primary,
            linewidth=0.35,
            alpha=0.96,
            label=criterion,
            zorder=3,
        )
    ax.set_xticks(x, dataset_labels)
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of analysis units")
    ax.set_title("Cohort retention and usability")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", color=grid_color, lw=0.35, zorder=0)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(-0.02, 1.02), fontsize=5.2, handlelength=1.0, columnspacing=0.8)
    sns.despine(ax=ax)
    panel_label(ax, "A")

    ax = axes[0, 1]
    stage_pivot = (
        stage_df.pivot(index="dataset", columns="stage", values="fraction")
        .fillna(0)
        .reindex(dataset_labels)
    )
    stage_columns = [
        stage for stage in STAGE_ORDER if stage in stage_pivot.columns
    ]
    bottoms = np.zeros(len(stage_pivot))
    stage_colors = {
        "early": cat.get("sage", "#B8D2CC"),
        "local_advanced": cat.get("pale_yellow", "#FEEBB9"),
        "primary": cat.get("sky_blue", "#B2E6FD"),
        "metastatic": cat.get("coral", "#E8B2A7"),
        "unknown": "#D6D6D6",
    }
    for stage in stage_columns:
        values = stage_pivot[stage].to_numpy()
        ax.bar(
            np.arange(len(stage_pivot)),
            values,
            bottom=bottoms,
            label=stage,
            color=stage_colors[stage],
            edgecolor=text_primary,
            linewidth=0.32,
            width=0.72,
            alpha=0.96,
            zorder=3,
        )
        bottoms += values
    ax.set_xticks(np.arange(len(stage_pivot)), stage_pivot.index, rotation=18)
    ax.tick_params(axis="x", rotation=0)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction of analysis units")
    ax.set_title("Clinical/progression-state composition")
    ax.grid(axis="y", color=grid_color, lw=0.35, zorder=0)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(-0.02, 1.02), fontsize=5.2, handlelength=1.0, columnspacing=0.7)
    sns.despine(ax=ax)
    panel_label(ax, "B")

    ax = axes[1, 0]
    sns.violinplot(
        data=burden_df,
        x="dataset",
        y="event_count",
        order=dataset_labels,
        color=cat.get("sky_blue", "#B2E6FD"),
        inner="quartile",
        cut=0,
        linewidth=0.65,
        ax=ax,
    )
    for collection in ax.collections:
        collection.set_alpha(0.82)
        collection.set_edgecolor(text_primary)
    ax.set_xlabel("")
    ax.set_ylabel("p15 events per sample")
    ax.set_title("Cancer-specific event burden")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", color=grid_color, lw=0.35, zorder=0)
    sns.despine(ax=ax)
    panel_label(ax, "C")

    ax = axes[1, 1]
    frequencies = np.vstack(panel_frequency)
    annotations = np.vstack(panel_annotations)
    sns.heatmap(
        frequencies,
        annot=annotations,
        fmt="",
        cmap=mcolors.LinearSegmentedColormap.from_list("panel_frequency", ["#F7F7F7", cat.get("coral", "#E8B2A7"), "#B2182B"]),
        vmin=0,
        vmax=max(0.01, float(frequencies.max())),
        annot_kws={"fontsize": 4.5, "rotation": 90},
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Event frequency", "shrink": 0.68},
        ax=ax,
    )
    ax.set_yticks(np.arange(len(dataset_labels)) + 0.5, dataset_labels, rotation=0)
    ax.set_xticks(np.arange(15) + 0.5, [f"Rank {i}" for i in range(1, 16)])
    ax.tick_params(axis="x", rotation=55, labelsize=5.2)
    ax.tick_params(axis="y", labelsize=5.7)
    ax.set_xlabel("Cancer-specific p15 panel rank")
    ax.set_ylabel("")
    ax.set_title("Driver panel composition and frequency")
    panel_label(ax, "D")

    fig.suptitle(
        "Experiment 1: integrated data preparation across four cancer cohorts",
        fontweight="bold",
        y=0.975,
        fontsize=10.2,
    )
    save_square_figure(fig, output_base, dpi)


def plot_combined_e1_state_sparsity(
    dataset_names: list[str],
    experiment_config: dict,
    result_root: Path,
    output_base: Path,
    dpi: int,
) -> None:
    colors = figure_style.colors(experiment_config)
    cat = figure_style.categorical_palette(experiment_config)
    grid_color = colors.get("text", {}).get("grid", "#E6E6E6")
    display = {
        dataset: experiment_config["datasets"][dataset].get("short_name", cohort_short_label(dataset))
        for dataset in dataset_names
    }
    metric_frames = []
    occupancy_frames = []
    for dataset in dataset_names:
        tables = (
            result_root
            / dataset
            / "experiment_01_data_preparation"
            / "tables"
        )
        metrics = pd.read_csv(tables / "panel_sensitivity_metrics.csv")
        metrics["dataset"] = display[dataset]
        metric_frames.append(metrics)
        occupancy = pd.read_csv(tables / "state_occupancy_p15.csv")
        occupancy["dataset"] = display[dataset]
        occupancy_frames.append(occupancy)
    metrics = pd.concat(metric_frames, ignore_index=True)
    occupancy = pd.concat(occupancy_frames, ignore_index=True)

    palette = dict(
        zip(
            [display[dataset] for dataset in dataset_names],
            [
                cat.get("lavender", "#B5AED5"),
                cat.get("sky_blue", "#B2E6FD"),
                cat.get("sage", "#B8D2CC"),
                cat.get("coral", "#E8B2A7"),
            ],
        )
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2))
    fig.subplots_adjust(
        left=0.085,
        right=0.965,
        bottom=0.085,
        top=0.865,
        wspace=0.270,
        hspace=0.350,
    )

    ax = axes[0, 0]
    sns.lineplot(
        data=metrics,
        x="panel_size",
        y="unique_states",
        hue="dataset",
        marker="o",
        palette=palette,
        linewidth=0.9,
        markersize=4.3,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_xticks([10, 15, 20, 25])
    ax.set_xlabel("Event panel size")
    ax.set_ylabel("All states (log scale)")
    ax.set_title("State-space growth")
    ax.grid(color=grid_color, lw=0.35, which="major")
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.despine(ax=ax)
    panel_label(ax, "A")

    ax = axes[0, 1]
    sns.lineplot(
        data=metrics,
        x="panel_size",
        y="valid_states",
        hue="dataset",
        marker="o",
        palette=palette,
        legend=False,
        linewidth=0.9,
        markersize=4.3,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_xticks([10, 15, 20, 25])
    ax.set_xlabel("Event panel size")
    ax.set_ylabel("Valid states (log scale)")
    ax.set_title("Stable-state support")
    ax.grid(color=grid_color, lw=0.35, which="major")
    sns.despine(ax=ax)
    panel_label(ax, "B")

    ax = axes[1, 0]
    sns.lineplot(
        data=metrics,
        x="panel_size",
        y="valid_unit_fraction",
        hue="dataset",
        marker="o",
        palette=palette,
        legend=False,
        linewidth=0.9,
        markersize=4.3,
        ax=ax,
    )
    ax.set_xticks([10, 15, 20, 25])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Event panel size")
    ax.set_ylabel("Samples in valid states")
    ax.set_title("Usable sample fraction")
    ax.grid(color=grid_color, lw=0.35)
    sns.despine(ax=ax)
    panel_label(ax, "C")

    ax = axes[1, 1]
    for dataset_label, group in occupancy.groupby("dataset"):
        counts = (
            pd.to_numeric(group["state_count"], errors="coerce")
            .dropna()
            .sort_values()
            .to_numpy()
        )
        y = np.arange(1, len(counts) + 1) / len(counts)
        ax.step(
            counts,
            y,
            where="post",
            label=dataset_label,
            color=palette[dataset_label],
            linewidth=0.95,
        )
    ax.axvline(5, color="#222222", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Samples per p15 state (log scale)")
    ax.set_ylabel("Cumulative fraction of states")
    ax.set_title("State occupancy distributions")
    ax.grid(color=grid_color, lw=0.35, which="major")
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    sns.despine(ax=ax)
    panel_label(ax, "D")

    fig.suptitle(
        "Experiment 1: state-space sensitivity across four cancer cohorts",
        y=0.975,
        fontweight="bold",
        fontsize=10.2,
    )
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=palette[label],
            marker="o",
            linewidth=1.2,
            markersize=4.2,
            label=label,
        )
        for label in [display[dataset] for dataset in dataset_names]
    ]
    fig.legend(
        handles=handles,
        title="Cancer cohort",
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.928),
        ncol=4,
        columnspacing=1.0,
        handletextpad=0.5,
        fontsize=5.4,
        title_fontsize=5.7,
    )
    save_square_figure(fig, output_base, dpi)


def inclusion_count(inclusion: pd.DataFrame, keywords: list[str]) -> int:
    """Return the first inclusion count whose step contains all keywords."""
    lowered = inclusion["step"].astype(str).str.lower()
    mask = pd.Series(True, index=inclusion.index)
    for keyword in keywords:
        mask &= lowered.str.contains(keyword.lower(), regex=False)
    if not mask.any():
        raise ValueError(f"Missing inclusion step containing {keywords}")
    return int(inclusion.loc[mask, "n"].iloc[0])


def plot_publication_single_figures(
    dataset_names: list[str],
    experiment_config: dict,
    result_root: Path,
    dpi: int,
) -> None:
    """Render atomic E1/E2 panels directly from saved experiment tables."""
    colors = figure_style.colors(experiment_config)
    cat = figure_style.categorical_palette(experiment_config)
    text_primary = colors.get("text", {}).get("primary", "#263238")
    grid_color = colors.get("text", {}).get("grid", "#E6E6E6")
    display = {
        dataset: experiment_config["datasets"][dataset].get("short_name", cohort_short_label(dataset))
        for dataset in dataset_names
    }
    dataset_labels = [display[dataset] for dataset in dataset_names]
    single_dir = result_root / "single_figures"
    single_dir.mkdir(parents=True, exist_ok=True)

    retention_rows = []
    stage_rows = []
    burden_rows = []
    panel_frequency = []
    panel_annotations = []
    metric_frames = []
    occupancy_frames = []

    for dataset in dataset_names:
        tables = result_root / dataset / "experiment_01_data_preparation" / "tables"
        inclusion = pd.read_csv(tables / "sample_inclusion_summary.csv")
        all_units = inclusion_count(inclusion, ["all tumor"])
        step_map = {
            "Functional mutation": ["functional mutation"],
            "Known state": ["known", "state"],
            "Panel event": ["p15", "event"],
            "Valid state": ["valid", "p15", "state"],
        }
        for label, keywords in step_map.items():
            count = inclusion_count(inclusion, keywords)
            retention_rows.append(
                {
                    "dataset": display[dataset],
                    "criterion": label,
                    "fraction": count / all_units,
                }
            )

        clinical = pd.read_csv(tables / "clinical_clean.csv", dtype=str)
        stage_counts = clinical["stage_group_experiment1"].value_counts()
        for stage, count in stage_counts.items():
            stage_rows.append(
                {
                    "dataset": display[dataset],
                    "stage": stage,
                    "fraction": count / len(clinical),
                }
            )

        burden = pd.read_csv(tables / "sample_event_counts_p15.csv")
        burden_rows.extend(
            {
                "dataset": display[dataset],
                "event_count": int(value),
            }
            for value in burden["event_count"]
        )

        panel = pd.read_csv(tables / "main_event_panel_p15.csv").sort_values("panel_rank")
        panel_frequency.append(panel["frequency"].astype(float).to_numpy())
        panel_annotations.append(panel["event"].astype(str).to_numpy())

        metrics = pd.read_csv(tables / "panel_sensitivity_metrics.csv")
        metrics["dataset"] = display[dataset]
        metric_frames.append(metrics)
        occupancy = pd.read_csv(tables / "state_occupancy_p15.csv")
        occupancy["dataset"] = display[dataset]
        occupancy_frames.append(occupancy)

    retention_df = pd.DataFrame(retention_rows)
    stage_df = pd.DataFrame(stage_rows)
    burden_df = pd.DataFrame(burden_rows)
    metrics = pd.concat(metric_frames, ignore_index=True)
    occupancy = pd.concat(occupancy_frames, ignore_index=True)
    cohort_palette = dict(
        zip(
            dataset_labels,
            [
                cat.get("lavender", "#B5AED5"),
                cat.get("sky_blue", "#B2E6FD"),
                cat.get("sage", "#B8D2CC"),
                cat.get("coral", "#E8B2A7"),
            ],
        )
    )

    fig, ax = plt.subplots(figsize=(3.35, 3.1))
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.18, top=0.96)
    criteria = retention_df["criterion"].drop_duplicates().tolist()
    criterion_colors = [
        cat.get("sky_blue", "#B2E6FD"),
        cat.get("sage", "#B8D2CC"),
        cat.get("lavender", "#B5AED5"),
        cat.get("pale_yellow", "#FEEBB9"),
    ]
    x = np.arange(len(dataset_labels))
    width = 0.17
    offsets = (np.arange(len(criteria)) - (len(criteria) - 1) / 2) * width
    retention_lookup = retention_df.set_index(["dataset", "criterion"])["fraction"]
    for idx, criterion in enumerate(criteria):
        values = [float(retention_lookup.get((dataset, criterion), np.nan)) for dataset in dataset_labels]
        ax.bar(
            x + offsets[idx],
            values,
            width=width * 0.9,
            color=criterion_colors[idx],
            edgecolor=text_primary,
            linewidth=0.32,
            label=criterion,
            zorder=3,
        )
    ax.set_xticks(x, dataset_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Fraction of analysis units")
    ax.grid(axis="y", color=grid_color, lw=0.35, zorder=0)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(-0.02, 1.03), fontsize=5.4, handlelength=1.0, columnspacing=0.8)
    sns.despine(ax=ax)
    save_figure(fig, single_dir / "Figure_E1_QC_overview__cohort_retention", dpi)

    fig, ax = plt.subplots(figsize=(3.35, 3.1))
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.18, top=0.94)
    stage_pivot = (
        stage_df.pivot(index="dataset", columns="stage", values="fraction")
        .fillna(0)
        .reindex(dataset_labels)
    )
    stage_columns = [stage for stage in STAGE_ORDER if stage in stage_pivot.columns]
    stage_colors = {
        "early": cat.get("sage", "#B8D2CC"),
        "local_advanced": cat.get("pale_yellow", "#FEEBB9"),
        "primary": cat.get("sky_blue", "#B2E6FD"),
        "metastatic": cat.get("coral", "#E8B2A7"),
        "unknown": "#D6D6D6",
    }
    bottoms = np.zeros(len(stage_pivot))
    for stage in stage_columns:
        values = stage_pivot[stage].to_numpy()
        ax.bar(
            np.arange(len(stage_pivot)),
            values,
            bottom=bottoms,
            label=stage,
            color=stage_colors[stage],
            edgecolor=text_primary,
            linewidth=0.32,
            width=0.70,
            zorder=3,
        )
        bottoms += values
    ax.set_xticks(np.arange(len(stage_pivot)), stage_pivot.index)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Fraction of analysis units")
    ax.grid(axis="y", color=grid_color, lw=0.35, zorder=0)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(-0.02, 1.03), fontsize=5.2, handlelength=1.0, columnspacing=0.7)
    sns.despine(ax=ax)
    save_figure(fig, single_dir / "Figure_E1_QC_overview__progression_state_composition", dpi)

    fig, ax = plt.subplots(figsize=(3.2, 3.1))
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.18, top=0.96)
    sns.violinplot(
        data=burden_df,
        x="dataset",
        y="event_count",
        hue="dataset",
        order=dataset_labels,
        hue_order=dataset_labels,
        palette=cohort_palette,
        inner="quartile",
        cut=0,
        linewidth=0.65,
        legend=False,
        ax=ax,
    )
    for collection in ax.collections:
        collection.set_alpha(0.82)
        collection.set_edgecolor(text_primary)
    ax.set_xlabel("")
    ax.set_ylabel("p15 events per sample")
    ax.grid(axis="y", color=grid_color, lw=0.35, zorder=0)
    sns.despine(ax=ax)
    save_figure(fig, single_dir / "Figure_E1_QC_overview__event_burden", dpi)

    fig, ax = plt.subplots(figsize=(4.6, 2.75))
    fig.subplots_adjust(left=0.12, right=0.94, bottom=0.34, top=0.94)
    frequencies = np.vstack(panel_frequency)
    annotations = np.vstack(panel_annotations)
    sns.heatmap(
        frequencies,
        annot=annotations,
        fmt="",
        cmap=mcolors.LinearSegmentedColormap.from_list(
            "panel_frequency_single",
            ["#F7F7F7", cat.get("coral", "#E8B2A7"), "#B2182B"],
        ),
        vmin=0,
        vmax=max(0.01, float(frequencies.max())),
        annot_kws={"fontsize": 4.6, "rotation": 90},
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "Event frequency", "shrink": 0.65},
        ax=ax,
    )
    ax.set_yticks(np.arange(len(dataset_labels)) + 0.5, dataset_labels, rotation=0)
    ax.set_xticks(np.arange(15) + 0.5, [str(i) for i in range(1, 16)])
    ax.tick_params(axis="x", labelsize=5.5, rotation=0)
    ax.tick_params(axis="y", labelsize=6.0)
    ax.set_xlabel("Cancer-specific p15 panel rank")
    ax.set_ylabel("")
    save_figure(fig, single_dir / "Figure_E1_QC_overview__driver_panel_frequency", dpi)

    for metric, ylabel, name, ylim in [
        ("unique_states", "All states", "state_space_growth", None),
        ("valid_states", "Valid states", "valid_state_support", None),
        ("valid_unit_fraction", "Samples in valid states", "valid_sample_fraction", (0, 1.02)),
    ]:
        fig, ax = plt.subplots(figsize=(3.2, 3.1))
        fig.subplots_adjust(left=0.20, right=0.96, bottom=0.18, top=0.96)
        sns.lineplot(
            data=metrics,
            x="panel_size",
            y=metric,
            hue="dataset",
            marker="o",
            palette=cohort_palette,
            linewidth=0.95,
            markersize=4.0,
            ax=ax,
        )
        if metric != "valid_unit_fraction":
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xticks([10, 15, 20, 25])
        ax.set_xlabel("Event panel size")
        ax.set_ylabel(ylabel)
        ax.grid(color=grid_color, lw=0.35, which="major")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        sns.despine(ax=ax)
        save_figure(fig, single_dir / f"Figure_E1_state_sparsity__{name}", dpi)

    fig, ax = plt.subplots(figsize=(3.2, 3.1))
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.18, top=0.96)
    for dataset_label, group in occupancy.groupby("dataset"):
        counts = (
            pd.to_numeric(group["state_count"], errors="coerce")
            .dropna()
            .sort_values()
            .to_numpy()
        )
        y = np.arange(1, len(counts) + 1) / len(counts)
        ax.step(
            counts,
            y,
            where="post",
            label=dataset_label,
            color=cohort_palette[dataset_label],
            linewidth=0.95,
        )
    ax.axvline(5, color="#222222", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Samples per p15 state")
    ax.set_ylabel("Cumulative fraction of states")
    ax.grid(color=grid_color, lw=0.35, which="major")
    ax.legend(frameon=False, loc="lower right", fontsize=5.5, handlelength=1.2)
    sns.despine(ax=ax)
    save_figure(fig, single_dir / "Figure_E1_state_sparsity__state_occupancy_ecdf", dpi)


def plot_state_sparsity(
    display_name: str,
    metrics: pd.DataFrame,
    occupancy: pd.DataFrame,
    output_base: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.36
    ax.bar(x - width / 2, metrics["unique_states"], width, label="All states", color="#4477AA")
    ax.bar(x + width / 2, metrics["valid_states"], width, label="Valid states", color="#228833")
    ax.set_xticks(x, [f"p={p}" for p in metrics["panel_size"]])
    ax.set_ylabel("Number of states")
    ax.set_title("State-space size across panels")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    panel_label(ax, "A")

    ax = axes[1]
    counts = pd.to_numeric(occupancy["state_count"], errors="coerce").dropna()
    bins = np.logspace(0, np.log10(max(5, counts.max())), 24)
    ax.hist(counts, bins=bins, color="#CC6677", edgecolor="white", linewidth=0.4)
    ax.axvline(5, color="#222222", linestyle="--", linewidth=1, label="valid-state threshold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Samples per state")
    ax.set_ylabel("States")
    ax.set_title("Main-panel state occupancy")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    panel_label(ax, "B")
    fig.suptitle(f"{display_name}: state-space sparsity", fontweight="bold")
    save_figure(fig, output_base, dpi)


def plot_experiment2_summary(
    display_name: str,
    assignments: pd.DataFrame,
    scheme_summary: pd.DataFrame,
    state_metrics: pd.DataFrame,
    agreement: pd.DataFrame,
    event_heatmaps: dict[str, pd.DataFrame],
    output_base: Path,
    heatmap_base: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
    ax = axes[0, 0]
    dist = scheme_summary.pivot(index="scheme", columns="group", values="fraction").fillna(0)
    bottoms = np.zeros(len(dist))
    groups = list(dist.columns)
    palette = sns.color_palette("colorblind", n_colors=max(len(groups), 3))
    for idx, group in enumerate(groups):
        values = dist[group].to_numpy()
        ax.bar(np.arange(len(dist)), values, bottom=bottoms, label=group, color=palette[idx])
        bottoms += values
    ax.set_xticks(np.arange(len(dist)), [s.replace("_", "\n") for s in dist.index])
    ax.set_ylabel("Fraction of samples")
    ax.set_ylim(0, 1)
    ax.set_title("Alternative progression-state definitions")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1)
    sns.despine(ax=ax)
    panel_label(ax, "A")

    ax = axes[0, 1]
    x = np.arange(len(state_metrics))
    width = 0.36
    ax.bar(x - width / 2, state_metrics["unique_states"], width, color="#4477AA", label="All states")
    ax.bar(x + width / 2, state_metrics["valid_states"], width, color="#228833", label="Valid states")
    ax.set_xticks(x, [s.replace("_", "\n") for s in state_metrics["scheme"]])
    ax.set_ylabel("Number of states")
    ax.set_title("State-space sensitivity")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    panel_label(ax, "B")

    ax = axes[1, 0]
    agree = agreement.pivot(index="scheme_a", columns="scheme_b", values="cramers_v")
    names = sorted(set(agreement["scheme_a"]) | set(agreement["scheme_b"]))
    full = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for _, row in agreement.iterrows():
        full.loc[row["scheme_a"], row["scheme_b"]] = row["cramers_v"]
        full.loc[row["scheme_b"], row["scheme_a"]] = row["cramers_v"]
    sns.heatmap(
        full,
        ax=ax,
        cmap=sns.light_palette("#5B2C83", as_cmap=True),
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Cramér's V", "shrink": 0.8},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Pairwise scheme agreement")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    panel_label(ax, "C")

    ax = axes[1, 1]
    ax.axis("off")
    pending = [
        "MHN progression-score scheme: pending Experiment 3",
        "R* rank correlation: pending MHN + inflow estimation",
        "Top-K R* overlap: pending MHN + inflow estimation",
        f"Current schemes evaluated: {len(state_metrics)}",
        f"Clinical-stage usable units: {int(assignments['clinical_stage'].ne('unknown').sum()):,}",
    ]
    ax.text(
        0.02,
        0.95,
        "Pre-MHN sensitivity boundary",
        fontsize=9.5,
        fontweight="bold",
        va="top",
    )
    ax.text(0.02, 0.83, "\n".join(f"• {line}" for line in pending), fontsize=8, va="top", linespacing=1.6)
    panel_label(ax, "D")

    fig.suptitle(f"{display_name}: Experiment 2 stage-definition sensitivity", fontweight="bold")
    save_figure(fig, output_base, dpi)

    schemes = list(event_heatmaps)
    fig, axes = plt.subplots(
        len(schemes),
        1,
        figsize=(10.5, max(2.4 * len(schemes), 5)),
        constrained_layout=True,
    )
    if len(schemes) == 1:
        axes = [axes]
    for idx, (scheme, heat) in enumerate(event_heatmaps.items()):
        ax = axes[idx]
        sns.heatmap(
            heat,
            ax=ax,
            cmap=sns.light_palette("#2166AC", as_cmap=True),
            vmin=0,
            vmax=max(0.01, float(heat.to_numpy().max())),
            cbar_kws={"label": "Event frequency", "shrink": 0.72},
            linewidths=0.2,
            linecolor="white",
        )
        ax.set_title(scheme.replace("_", " ").title(), loc="left")
        ax.set_xlabel("Event" if idx == len(schemes) - 1 else "")
        ax.set_ylabel("Group")
        ax.tick_params(axis="x", rotation=55, labelsize=6.5)
        ax.tick_params(axis="y", rotation=0)
        panel_label(ax, chr(ord("A") + idx))
    fig.suptitle(f"{display_name}: event profiles under alternative state definitions", fontweight="bold")
    save_figure(fig, heatmap_base, dpi)


def write_dataset_report(
    dataset_name: str,
    dataset_config: dict,
    exp1_metrics: dict,
    panel_table: pd.DataFrame,
    scheme_summary: pd.DataFrame,
    state_metrics: pd.DataFrame,
    report_path: Path,
) -> None:
    selected = panel_table[panel_table["selected_p15"]].sort_values("panel_rank")
    pathways = selected["pathway"].value_counts().to_dict()
    stage_groups = (
        scheme_summary[scheme_summary["scheme"] == "clinical_stage"]
        .set_index("group")["n"]
        .to_dict()
    )
    lines = [
        f"# Experiments 1–2: {dataset_config['display_name']}",
        "",
        "## Experiment 1",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Analysis units: {exp1_metrics['analysis_units']:,}",
        f"- Samples with functional mutation: {exp1_metrics['mutated_units']:,}",
        f"- Main event panel: p=15",
        f"- Sensitivity panels: p=10, 15, 20, 25",
        f"- Main-panel zero-event fraction: {exp1_metrics['zero_event_fraction']:.3f}",
        f"- Main-panel valid states: {exp1_metrics['valid_states']}",
        f"- Main-panel samples in valid states: {exp1_metrics['samples_in_valid_states']:,}",
        "",
        "### Cancer-Specific Event Panel",
        "",
        ", ".join(selected["event"].tolist()),
        "",
        "The panel was built by prioritizing established cancer drivers and pathways, then filling remaining slots with cohort-supported events. Likely size-related passenger genes were not used as frequency-fill events.",
        "",
        f"Pathway representation: {pathways}",
        "",
        "## Experiment 2",
        "",
        f"Clinical/progression groups: {stage_groups}",
        "",
        "Pre-MHN sensitivity schemes completed:",
        "",
        "- Clinical stage/progression group.",
        "- Primary/metastatic status.",
        "- Mutation-burden grouping.",
        "- Cancer-specific pathway-burden grouping.",
        "",
        "The MHN progression-score scheme, R* rank correlation, and Top-K R* overlap are deliberately pending because they require the subsequent MHN and inflow experiments.",
        "",
        "### State-Space Sensitivity",
        "",
    ]
    for _, row in state_metrics.iterrows():
        lines.append(
            f"- {row['scheme']}: {row['unique_states']} states, {row['valid_states']} valid states, "
            f"{row['valid_unit_fraction']:.3f} of units in valid states."
        )
    lines.extend(
        [
            "",
            "## Biological Interpretation Boundary",
            "",
            "These experiments establish a clean, cancer-specific input space and progression-state definitions. They do not yet infer transition probabilities, dwell indices, or clinical effects.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_dataset(
    dataset_name: str,
    input_dir: Path,
    dataset_config: dict,
    global_config: dict,
    result_root: Path,
) -> dict:
    logging.info("Starting Experiments 1-2 for %s", dataset_name)
    display_name = dataset_config["display_name"]
    metadata, mutations = read_inputs(input_dir)
    metadata = metadata.sort_values("analysis_id").reset_index(drop=True)
    mutations = mutations[mutations["analysis_id"].isin(set(metadata["analysis_id"]))]

    exp1_root = result_root / dataset_name / "experiment_01_data_preparation"
    exp2_root = result_root / dataset_name / "experiment_02_stage_sensitivity"
    exp1_tables = exp1_root / "tables"
    exp1_figures = exp1_root / "figures"
    exp2_tables = exp2_root / "tables"
    exp2_figures = exp2_root / "figures"
    for path in [exp1_tables, exp1_figures, exp2_tables, exp2_figures]:
        path.mkdir(parents=True, exist_ok=True)

    panel_sizes = [int(x) for x in global_config["sensitivity_panel_sizes"]]
    panels, panel_table = construct_event_panels(
        metadata,
        mutations,
        dataset_config,
        panel_sizes,
        int(global_config["minimum_event_samples"]),
    )
    panel_table.to_csv(exp1_tables / "event_panel_candidates.csv", index=False)

    clinical_stage = normalize_clinical_stage(metadata["stage_group"])
    panel_metrics = []
    matrices: dict[int, pd.DataFrame] = {}
    occupancies: dict[int, pd.DataFrame] = {}
    state_tables: dict[int, pd.DataFrame] = {}
    for size in panel_sizes:
        events = panels[size]
        matrix = build_matrix(metadata, mutations, events)
        state_table, occupancy = build_state_table(
            metadata,
            matrix,
            clinical_stage,
            events,
            int(global_config["minimum_state_count"]),
        )
        matrices[size] = matrix
        state_tables[size] = state_table
        occupancies[size] = occupancy
        matrix.to_csv(exp1_tables / f"event_matrix_p{size}.csv", index=False)
        matrix[events].to_csv(exp1_tables / f"mhn_training_matrix_p{size}.csv", index=False)
        pd.DataFrame(
            {
                "panel_rank": range(1, len(events) + 1),
                "event": events,
            }
        ).to_csv(exp1_tables / f"event_panel_p{size}.csv", index=False)
        state_table.to_csv(exp1_tables / f"state_table_p{size}.csv", index=False)
        occupancy.to_csv(exp1_tables / f"state_occupancy_p{size}.csv", index=False)
        panel_metrics.append(
            {
                "dataset_name": dataset_name,
                "panel_size": size,
                "events_retained": len(events),
                "unique_states": int(len(occupancy)),
                "valid_states": int(occupancy["state_count_flag"].eq("valid_state").sum()),
                "rare_states": int(occupancy["state_count_flag"].eq("rare_state").sum()),
                "samples_in_valid_states": int(state_table["state_count_flag"].eq("valid_state").sum()),
                "valid_unit_fraction": float(state_table["state_count_flag"].eq("valid_state").mean()),
                "zero_event_fraction": float(state_table["event_count"].eq(0).mean()),
            }
        )
    panel_metrics_df = pd.DataFrame(panel_metrics)
    panel_metrics_df.to_csv(exp1_tables / "panel_sensitivity_metrics.csv", index=False)

    primary_size = int(global_config["primary_panel_size"])
    primary_events = panels[primary_size]
    primary_matrix = matrices[primary_size]
    primary_state = state_tables[primary_size]
    primary_occupancy = occupancies[primary_size]
    clinical_clean = metadata.copy()
    clinical_clean["stage_group_experiment1"] = clinical_stage
    clinical_clean.to_csv(exp1_tables / "clinical_clean.csv", index=False)
    row_map = primary_state[
        [
            "analysis_id",
            "patient_id",
            "sample_id",
            "stage_group",
            "genotype_signature",
            "event_count",
            "state_id",
            "state_count_flag",
            "usable_for_mhn",
            "usable_for_relobstq",
        ]
    ].copy()
    row_map.insert(0, "row_index", range(len(row_map)))
    row_map.to_csv(exp1_tables / "mhn_row_index_map_p15.csv", index=False)

    stage_freq = stage_event_frequency(primary_matrix, clinical_stage, primary_events)
    stage_freq.to_csv(exp1_tables / "stage_event_frequency_p15.csv")
    sample_event_counts = primary_state[
        ["analysis_id", "patient_id", "sample_id", "stage_group", "event_count"]
    ]
    sample_event_counts.to_csv(exp1_tables / "sample_event_counts_p15.csv", index=False)

    inclusion = pd.DataFrame(
        [
            {"step": "All tumor analysis units", "n": len(metadata)},
            {"step": "With ≥1 functional mutation", "n": mutations["analysis_id"].nunique()},
            {"step": "Known clinical/progression state", "n": int(clinical_stage.ne("unknown").sum())},
            {"step": "With ≥1 p15 event", "n": int(primary_state["event_count"].gt(0).sum())},
            {"step": "In valid p15 states", "n": int(primary_state["state_count_flag"].eq("valid_state").sum())},
        ]
    )
    inclusion.to_csv(exp1_tables / "sample_inclusion_summary.csv", index=False)

    selected_panel = panel_table[panel_table["selected_p15"]].sort_values("panel_rank")
    selected_panel.to_csv(exp1_tables / "main_event_panel_p15.csv", index=False)

    exp1_metrics = {
        "dataset_name": dataset_name,
        "analysis_units": int(len(metadata)),
        "unique_patients": int(metadata["patient_id"].nunique()),
        "mutated_units": int(mutations["analysis_id"].nunique()),
        "functional_mutation_rows": int(len(mutations)),
        "main_panel_events": int(len(primary_events)),
        "known_stage_fraction": float(clinical_stage.ne("unknown").mean()),
        "zero_event_fraction": float(primary_state["event_count"].eq(0).mean()),
        "unique_states": int(len(primary_occupancy)),
        "valid_states": int(primary_occupancy["state_count_flag"].eq("valid_state").sum()),
        "samples_in_valid_states": int(primary_state["state_count_flag"].eq("valid_state").sum()),
        "mhn_format_binary": bool(
            primary_matrix[primary_events].isin([0, 1]).all().all()
        ),
        "mhn_rows_match_metadata": bool(len(primary_matrix) == len(metadata)),
    }
    pd.DataFrame([exp1_metrics]).to_csv(exp1_tables / "experiment_01_metrics.csv", index=False)

    dpi = int(global_config["plot"]["dpi"])
    plot_experiment1_overview(
        dataset_name,
        display_name,
        metadata,
        mutations,
        primary_matrix,
        primary_events,
        stage_freq,
        exp1_figures / "Figure_E1_QC_overview",
        dpi,
    )
    plot_oncoprint(
        display_name,
        metadata,
        primary_matrix,
        primary_events,
        int(global_config["oncoprint_samples_per_stage"]),
        exp1_figures / "Figure_E1_oncoprint",
        dpi,
    )
    plot_oncoprint(
        display_name,
        metadata,
        primary_matrix,
        primary_events,
        int(global_config["oncoprint_samples_per_stage"]),
        result_root / dataset_name / "single_figures" / "Figure_E1_oncoprint",
        dpi,
        include_title=False,
    )
    plot_state_sparsity(
        display_name,
        panel_metrics_df,
        primary_occupancy,
        exp1_figures / "Figure_E1_state_sparsity",
        dpi,
    )

    mutation_burden = primary_state["event_count"].astype(int)
    pathway_count = event_pathway_burden(
        mutations, metadata, dataset_config.get("pathways", {})
    )
    assignments = pd.DataFrame(
        {
            "analysis_id": metadata["analysis_id"],
            "clinical_stage": clinical_stage,
            "metastasis_status": normalize_metastasis(metadata["metastasis_status"]),
            "mutation_burden": burden_groups(mutation_burden, "burden"),
            "pathway_burden": burden_groups(pathway_count, "pathway"),
            "mhn_progression_score": "pending_mhn_training",
            "p15_event_count": mutation_burden,
            "pathway_count": pathway_count,
        }
    )
    assignments.to_csv(exp2_tables / "stage_scheme_assignments.csv", index=False)

    schemes = ["clinical_stage", "metastasis_status", "mutation_burden", "pathway_burden"]
    summary_rows = []
    state_metric_rows = []
    event_heatmaps: dict[str, pd.DataFrame] = {}
    scheme_state_tables: dict[str, pd.DataFrame] = {}
    for scheme in schemes:
        counts = assignments[scheme].value_counts(dropna=False)
        for group, count in counts.items():
            summary_rows.append(
                {
                    "dataset_name": dataset_name,
                    "scheme": scheme,
                    "group": group,
                    "n": int(count),
                    "fraction": float(count / len(assignments)),
                }
            )
        state_table, occupancy = build_state_table(
            metadata,
            primary_matrix,
            assignments[scheme],
            primary_events,
            int(global_config["minimum_state_count"]),
            stage_column_name=scheme,
        )
        scheme_state_tables[scheme] = state_table
        state_table.to_csv(exp2_tables / f"state_table_{scheme}.csv", index=False)
        occupancy.to_csv(exp2_tables / f"state_occupancy_{scheme}.csv", index=False)
        state_metric_rows.append(
            {
                "dataset_name": dataset_name,
                "scheme": scheme,
                "unique_groups": int(assignments[scheme].nunique()),
                "unknown_fraction": float(assignments[scheme].eq("unknown").mean()),
                "unique_states": int(len(occupancy)),
                "valid_states": int(occupancy["state_count_flag"].eq("valid_state").sum()),
                "rare_states": int(occupancy["state_count_flag"].eq("rare_state").sum()),
                "valid_unit_fraction": float(state_table["state_count_flag"].eq("valid_state").mean()),
            }
        )
        event_heatmaps[scheme] = stage_event_frequency(
            primary_matrix, assignments[scheme], primary_events
        )
        event_heatmaps[scheme].to_csv(exp2_tables / f"event_frequency_{scheme}.csv")

    scheme_summary = pd.DataFrame(summary_rows)
    state_metrics = pd.DataFrame(state_metric_rows)
    scheme_summary.to_csv(exp2_tables / "stage_scheme_summary.csv", index=False)
    state_metrics.to_csv(exp2_tables / "stage_scheme_state_metrics.csv", index=False)

    agreement_rows = []
    for i, scheme_a in enumerate(schemes):
        for scheme_b in schemes[i + 1 :]:
            valid = assignments[scheme_a].ne("unknown") & assignments[scheme_b].ne("unknown")
            agreement_rows.append(
                {
                    "dataset_name": dataset_name,
                    "scheme_a": scheme_a,
                    "scheme_b": scheme_b,
                    "n_compared": int(valid.sum()),
                    "cramers_v": cramer_v(
                        assignments.loc[valid, scheme_a],
                        assignments.loc[valid, scheme_b],
                    ),
                }
            )
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(exp2_tables / "stage_scheme_pairwise_agreement.csv", index=False)

    pending_rows = []
    for i, scheme_a in enumerate(schemes + ["mhn_progression_score"]):
        for scheme_b in (schemes + ["mhn_progression_score"])[i + 1 :]:
            pending_rows.append(
                {
                    "dataset_name": dataset_name,
                    "scheme_a": scheme_a,
                    "scheme_b": scheme_b,
                    "rstar_spearman": "",
                    "top10_overlap": "",
                    "status": "pending_mhn_training_and_rstar",
                }
            )
    pd.DataFrame(pending_rows).to_csv(
        exp2_tables / "rstar_stage_sensitivity_pending.csv", index=False
    )

    plot_experiment2_summary(
        display_name,
        assignments,
        scheme_summary,
        state_metrics,
        agreement,
        event_heatmaps,
        exp2_figures / "Figure_E2_stage_sensitivity",
        exp2_figures / "Figure_E2_stage_event_heatmaps",
        dpi,
    )

    write_dataset_report(
        dataset_name,
        dataset_config,
        exp1_metrics,
        panel_table,
        scheme_summary,
        state_metrics,
        result_root / dataset_name / "experiments_01_02_report.md",
    )

    checks = {
        "main_panel_has_15_events": len(primary_events) == 15,
        "nested_panels": all(
            set(panels[a]).issubset(set(panels[b]))
            for a, b in zip(panel_sizes[:-1], panel_sizes[1:])
        ),
        "binary_mhn_matrix": bool(primary_matrix[primary_events].isin([0, 1]).all().all()),
        "row_alignment": len(primary_matrix) == len(metadata) == len(primary_state),
        "no_duplicate_analysis_ids": metadata["analysis_id"].is_unique,
        "stage_assignments_complete": len(assignments) == len(metadata),
        "figures_png_pdf_present": True,
    }
    checks["figures_png_pdf_present"] = all(
        (base.with_suffix(".png").exists() and base.with_suffix(".pdf").exists())
        for base in [
            exp1_figures / "Figure_E1_QC_overview",
            exp1_figures / "Figure_E1_oncoprint",
            exp1_figures / "Figure_E1_state_sparsity",
            exp2_figures / "Figure_E2_stage_sensitivity",
            exp2_figures / "Figure_E2_stage_event_heatmaps",
        ]
    )
    (result_root / dataset_name / "validation.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )

    return {
        **exp1_metrics,
        "clinical_groups": int(assignments["clinical_stage"].nunique()),
        "metastasis_groups": int(assignments["metastasis_status"].nunique()),
        "mutation_burden_groups": int(assignments["mutation_burden"].nunique()),
        "pathway_burden_groups": int(assignments["pathway_burden"].nunique()),
        "all_validation_checks_passed": bool(all(checks.values())),
    }


def write_overall_summary(results: list[dict], result_root: Path) -> None:
    summary = pd.DataFrame(results)
    summary.to_csv(result_root / "experiments_01_02_summary.csv", index=False)
    columns = [
        "dataset_name",
        "analysis_units",
        "unique_patients",
        "main_panel_events",
        "zero_event_fraction",
        "valid_states",
        "known_stage_fraction",
        "all_validation_checks_passed",
    ]
    lines = [
        "# Experiments 1–2 Overall Summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in summary.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    lines.extend(
        [
            "",
            "Experiment 1 and the pre-MHN portion of Experiment 2 were run independently for all four datasets.",
            "",
            "The MHN progression-score stage scheme and R* ranking sensitivity remain pending by design; they require the subsequent MHN transition and Rel-ObsTQ experiments.",
            "",
        ]
    )
    (result_root / "experiments_01_02_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rel-ObsTQ-MHN Experiments 1 and 2.")
    parser.add_argument(
        "--experiment-config", default="configs/experiments_01_02.yaml"
    )
    parser.add_argument(
        "--dataset-config", default="configs/selected_experiment_datasets.yaml"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(".").resolve()
    setup_logging(project_root)
    with Path(args.experiment_config).open("r", encoding="utf-8") as handle:
        experiment_config = yaml.safe_load(handle)
    with Path(args.dataset_config).open("r", encoding="utf-8") as handle:
        selection_config = yaml.safe_load(handle)
    configure_plotting(experiment_config)

    result_root = Path(experiment_config["experiment_root"]).resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    results = []
    dataset_names = []
    for entry in selection_config["included_datasets"]:
        dataset_name = entry["dataset_name"]
        dataset_names.append(dataset_name)
        input_dir = Path(entry["input_dir"]).resolve()
        dataset_config = experiment_config["datasets"][dataset_name]
        results.append(
            run_dataset(
                dataset_name,
                input_dir,
                dataset_config,
                experiment_config,
                result_root,
            )
        )
        print(f"Completed Experiments 1-2: {dataset_name}")
    combined_dir = result_root / "combined_figures"
    combined_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(experiment_config["plot"]["dpi"])
    plot_combined_e1_overview(
        dataset_names,
        experiment_config,
        result_root,
        combined_dir / "Figure_E1_QC_overview_three_cohorts",
        dpi,
    )
    plot_combined_e1_state_sparsity(
        dataset_names,
        experiment_config,
        result_root,
        combined_dir / "Figure_E1_state_sparsity_three_cohorts",
        dpi,
    )
    plot_publication_single_figures(dataset_names, experiment_config, result_root, dpi)
    write_overall_summary(results, result_root)
    print(f"Results written to {result_root}")


if __name__ == "__main__":
    main()
