"""Audit immutable TRACERx native inputs and construct scorer-only branch references.

No outcome selection, model fitting, protocol mutation, or publication occurs here.
Branch detection means multiple markers assigned to an author-reconstructed clone
are observed; it does not mean that a region establishes ancestry or a new tree.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_NATIVE = Path('/Users/oukang/Desktop/投稿资料/IQE/iqe_project/data/metadata/tracerx_native_20260922T132136')
DEFAULT_OUT = Path(__file__).resolve().parents[1] / 'data' / 'native_audit'
FILES = ['patients.tsv.gz', 'tumours.tsv.gz', 'mutations.tsv.gz',
         'region_mutations.tsv.gz', 'tree_mutation_clusters.tsv.gz',
         'tree_inventory.tsv.gz', 'tree_edges.tsv.gz']


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def boolean(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False})


def write(df: pd.DataFrame, out: Path, name: str) -> None:
    df.to_csv(out / name, sep='\t', index=False)


def audit(native: Path, out: Path, min_markers: int = 3, min_depth: int = 20,
          min_alt: int = 3) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    source = {f: {'path': str(native / f), 'bytes': (native / f).stat().st_size,
                  'sha256': digest(native / f)} for f in FILES}
    pa = pd.read_csv(native / FILES[0], sep='\t')
    tu = pd.read_csv(native / FILES[1], sep='\t')
    assert not pa.cruk_id.duplicated().any()
    assert not tu.tumour_id_muttable_cruk.duplicated().any()
    assert set(tu.cruk_id).issubset(set(pa.cruk_id))
    mu = pd.read_csv(native / FILES[2], sep='\t', low_memory=False,
                     usecols=['mutation_id', 'patient_id', 'tumour_id', 'chr', 'start', 'ref',
                              'var', 'Hugo_Symbol', 'DriverMut', 'cleanCluster_SC',
                              'PyCloneCluster_SC_TB_corrected', 'exonic.func', 'func'])
    assert not mu.mutation_id.duplicated().any()
    mu['variant_key'] = (mu.tumour_id + ':' + mu.chr.str.replace(r'^chr', '', regex=True)
                         + ':' + mu.start.astype(str) + ':' + mu.ref + ':' + mu['var'])
    assert not mu[['tumour_id', 'variant_key']].duplicated().any()
    tc = pd.read_csv(native / FILES[4], sep='\t', dtype={'cluster': str})
    # Multi-tumour patients use a patient prefix in some author tree variant keys.
    # Join within the explicitly supplied tumour using full variant coordinates.
    tc['author_variant_key'] = tc.variant_key
    tc['variant_key'] = tc.tumour_id + ':' + tc.variant_key.str.split(':', n=1).str[1]
    tc['cluster'] = tc.cluster.str.replace(r'\.0$', '', regex=True)
    assert not tc[['tumour_id', 'variant_key']].duplicated().any()
    mm = tc.merge(mu, on=['tumour_id', 'variant_key'], how='left', validate='one_to_one', indicator=True)
    mapped = mm._merge.eq('both')
    mm = mm[mapped].copy()
    mm['clean'] = boolean(mm.cleanCluster_SC).fillna(False)
    inv = pd.read_csv(native / FILES[5], sep='\t', dtype={'trunk': str})
    # Avoid materializing the enormous collection of alternate-tree edge rows.
    es = []
    for c in pd.read_csv(native / FILES[6], sep='\t', chunksize=200000, dtype=str):
        es.append(c[c.kind.eq('Corrected_tree')])
    ed = pd.concat(es, ignore_index=True)
    roster = tu.rename(columns={'tumour_id_muttable_cruk': 'tumour_id', 'cruk_id': 'patient_id'})
    nodes, parents, trunks = {}, {}, {}
    for tid, e in ed.groupby('tumour_id'):
        nodes[tid] = set(e.parent) | set(e.child)
        parents[tid] = dict(zip(e.child, e.parent))
    for row in inv.itertuples(index=False):
        roots = set(str(row.trunk).split(';'))
        trunks[row.tumour_id] = roots
        nodes.setdefault(row.tumour_id, set()).update(roots)
    markers = mm[mm.clean & mm.apply(lambda r: r.cluster in nodes.get(r.tumour_id, set()), axis=1)].copy()
    markers['DriverMut'] = boolean(markers.DriverMut).fillna(False)
    markers['driver_gene'] = markers.Hugo_Symbol.where(markers.DriverMut, '')
    b = markers.groupby(['tumour_id', 'cluster'], observed=True).agg(
        reference_markers=('mutation_id', 'nunique'),
        driver_markers=('DriverMut', 'sum'),
        driver_genes=('driver_gene', lambda x: ';'.join(sorted(set(x) - {''})))).reset_index()
    children = ed.groupby('tumour_id').parent.apply(set).to_dict()
    b['is_trunk'] = [c in trunks.get(t, set()) for t, c in zip(b.tumour_id, b.cluster)]
    b['is_terminal'] = [c not in children.get(t, set()) for t, c in zip(b.tumour_id, b.cluster)]
    b['parent_cluster'] = [parents.get(t, {}).get(c, '') for t, c in zip(b.tumour_id, b.cluster)]
    b['eligible_multimarker_reference'] = b.reference_markers.ge(min_markers) & ~b.is_trunk
    b = b.merge(roster[['tumour_id', 'patient_id', 'histology_3']], on='tumour_id', validate='many_to_one')
    # Aggregate only branch-owned markers. Inherited trunk markers cannot retrieve a branch.
    key = markers[['tumour_id', 'mutation_id', 'cluster', 'DriverMut', 'Hugo_Symbol']]
    accum = []
    region_rosters = []
    duplicate_region_mutations = 0
    obskey = mu[['tumour_id', 'mutation_id', 'patient_id', 'Hugo_Symbol', 'chr', 'start', 'ref', 'var', 'exonic.func', 'func']].copy()
    # No clean-cluster, phylogenetic-cluster or hidden outcome fields enter observations.
    observation_path = out / 'region_variants.tsv.gz'
    core_path = out / 'region_core_variants.tsv.gz'
    observation_handle = gzip.open(observation_path, 'wt')
    core_handle = gzip.open(core_path, 'wt')
    reference_ccf_handle = gzip.open(out / 'scorer_region_CCF.tsv.gz', 'wt')
    first = first_core = True
    for r in pd.read_csv(native / FILES[3], sep='\t', chunksize=250000,
                         usecols=['tumour_id', 'RegionID', 'mutation_id', 'var_count', 'depth',
                                  'Is.present.region', 'CCF_phylo_SC']):
        duplicate_region_mutations += int(r[['RegionID', 'mutation_id']].duplicated().sum())
        region_rosters.append(r[['tumour_id', 'RegionID']].drop_duplicates())
        obs = r.merge(obskey, on=['tumour_id', 'mutation_id'], how='left', validate='many_to_one')
        assert obs.patient_id.notna().all()
        obs['positive'] = (boolean(obs['Is.present.region']).fillna(False)
                           & obs.depth.ge(min_depth) & obs.var_count.ge(min_alt))
        obs['evaluable'] = boolean(obs['Is.present.region']).notna() & obs.depth.ge(min_depth)
        obs['primary'] = obs.RegionID.str.contains(':SU_T', regex=False)
        reference_ccf = obs[['patient_id', 'tumour_id', 'RegionID', 'mutation_id', 'CCF_phylo_SC']].rename(columns={'CCF_phylo_SC': 'reference_CCF_phylo'})
        reference_ccf.to_csv(reference_ccf_handle, sep='\t', index=False, header=first)
        obs['observed_VAF'] = np.divide(obs.var_count, obs.depth, out=np.full(len(obs), np.nan), where=obs.depth.gt(0))
        obs.drop(columns=['CCF_phylo_SC'], inplace=True)
        obs.rename(columns={'Hugo_Symbol': 'gene', 'var': 'alt'}, inplace=True)
        obs.to_csv(observation_handle, sep='\t', index=False, header=first)
        first = False
        core = obs[obs.gene.isin(['TP53', 'KRAS', 'STK11', 'KEAP1'])]
        if len(core):
            core.to_csv(core_handle, sep='\t', index=False, header=first_core)
            first_core = False
        v = r.merge(key, on=['tumour_id', 'mutation_id'], how='inner', validate='many_to_one')
        known = boolean(v['Is.present.region']).notna()
        v['evaluable'] = v.depth.ge(min_depth) & known
        v['positive'] = v.evaluable & boolean(v['Is.present.region']).fillna(False) & v.var_count.ge(min_alt)
        v['driver_positive'] = v.positive & v.DriverMut
        v['one'] = 1
        accum.append(v.groupby(['tumour_id', 'RegionID', 'cluster'], observed=True)[
            ['one', 'evaluable', 'positive', 'driver_positive']].sum().reset_index())
    observation_handle.close()
    core_handle.close()
    reference_ccf_handle.close()
    rr = pd.concat(region_rosters).drop_duplicates().merge(roster[['tumour_id', 'patient_id', 'histology_3']], on='tumour_id', validate='many_to_one')
    rr['region_class'] = np.select([rr.RegionID.str.contains(':SU_T', regex=False),
                                    rr.RegionID.str.contains(':SU_LN', regex=False)],
                                   ['SURGICAL_PRIMARY', 'SURGICAL_LYMPH_NODE'], default='UNCLASSIFIED')
    cm = pd.concat(accum).groupby(['tumour_id', 'RegionID', 'cluster'], observed=True)[
        ['one', 'evaluable', 'positive', 'driver_positive']].sum().reset_index()
    cm = cm.rename(columns={'one': 'marker_rows', 'evaluable': 'markers_evaluable',
                            'positive': 'markers_positive', 'driver_positive': 'driver_markers_positive'})
    cm = cm.merge(b, on=['tumour_id', 'cluster'], validate='many_to_one')
    cm = cm.merge(rr[['RegionID', 'region_class']], on='RegionID', validate='many_to_one')
    cm['detected'] = cm.markers_positive.ge(min_markers)
    cm['support_status'] = np.select([cm.detected,
        cm.markers_positive.eq(0) & cm.markers_evaluable.ge(min_markers)],
        ['MULTIMARKER_DETECTED', 'NOT_DETECTED_AT_REGISTERED_DEPTH'], default='UNCERTAIN')
    anydet = cm.groupby(['tumour_id', 'cluster'], observed=True).detected.any().rename('observed_multimarker_support').reset_index()
    b = b.merge(anydet, on=['tumour_id', 'cluster'], validate='one_to_one')
    # A region co-detection establishes marker retrieval, not tree reconstruction.
    patient_counts = rr[rr.region_class.eq('SURGICAL_PRIMARY')].groupby('patient_id').agg(
        primary_regions=('RegionID', 'nunique'), primary_tumours=('tumour_id', 'nunique')).reset_index()
    pc = pa.rename(columns={'cruk_id': 'patient_id'}).merge(patient_counts, on='patient_id', how='left', validate='one_to_one')
    primary_count = rr[rr.region_class.eq('SURGICAL_PRIMARY')].groupby(['patient_id', 'tumour_id']).RegionID.nunique().rename('primary_regions').reset_index()
    representative = primary_count.sort_values(['patient_id', 'primary_regions', 'tumour_id'], ascending=[True, False, True]).drop_duplicates('patient_id')
    representative = representative.merge(roster[['tumour_id', 'histology_3']], on='tumour_id', validate='one_to_one')
    representative['selection_rule'] = 'MAX_PRIMARY_REGION_COUNT_THEN_LEXICOGRAPHIC_TUMOUR_ID;NO_OUTCOME_USE'
    write(representative, out, 'patient_primary_tumour.tsv')
    write(representative[representative.histology_3.eq('LUAD')], out, 'primary_luad_patients.tsv')
    # Scorer/reference-only gene-level OR across the whole tumour. Zero is a proxy
    # no-report state, not independently verified whole-gene callability.
    gd = mu.copy()
    gd['is_driver'] = boolean(gd.DriverMut).fillna(False)
    # Native ANNOVAR labels include "stopgain SNV", "frameshift substitution",
    # "immediate-stopgain" and "nonsynonymous"; do not silently lose these.
    nonsyn = (gd['exonic.func'].fillna('').str.contains(
        r'nonsynonymous|stopgain|stoploss|frameshift|startloss', case=False, regex=True)
        | gd.func.fillna('').str.contains('splicing', case=False, regex=False))
    gd['is_nonsynonymous'] = nonsyn
    gene_rows = []
    for tid, pid in zip(roster.tumour_id, roster.patient_id):
        z = gd[gd.tumour_id.eq(tid)]
        for gene in ['TP53', 'KRAS', 'STK11', 'KEAP1']:
            v = z[z.Hugo_Symbol.eq(gene)]
            gene_rows.append({'patient_id': pid, 'tumour_id': tid, 'gene': gene,
                'union_driver_event': bool(v.is_driver.any()),
                'union_nonsynonymous_event': bool(v.is_nonsynonymous.any()),
                'negative_semantics': 'NO_REPORTED_UNION_EVENT_PROXY_NOT_VERIFIED_CALLABLE_WILDTYPE'})
    write(pd.DataFrame(gene_rows), out, 'gene_union_reference.tsv')
    write(pc, out, 'patients.tsv')
    write(rr, out, 'regions.tsv')
    write(ed, out, 'corrected_tree_edges.tsv')
    write(b, out, 'branch_reference.tsv')
    write(cm, out, 'cluster_region_support.tsv.gz')
    write(markers[['patient_id', 'tumour_id', 'variant_key', 'author_variant_key', 'mutation_id', 'cluster', 'Hugo_Symbol', 'DriverMut']], out, 'branch_owned_markers.tsv.gz')
    summary = dict(source_files=source, audit_thresholds={'minimum_branch_owned_markers': min_markers,
        'minimum_locus_depth': min_depth, 'minimum_alt_reads_for_positive': min_alt},
        threshold_status='AUDIT_REFERENCE_RULES_NOT_A_METHOD_OR_OUTCOME_SELECTED_PROTOCOL',
        patients=len(pa), tumours=len(tu), histology=tu.histology_3.value_counts().to_dict(),
        regions=len(rr), region_classes=rr.region_class.value_counts().to_dict(),
        patients_with_at_least_3_primary_regions=int(pc.primary_regions.ge(3).sum()),
        deduplicated_primary_luad_patients=int(representative.histology_3.eq('LUAD').sum()),
        deduplicated_primary_luad_with_at_least_3_regions=int((representative.histology_3.eq('LUAD') & representative.primary_regions.ge(3)).sum()),
        tree_tumours=int(inv.tumour_id.nunique()), corrected_edges=len(ed),
        tree_marker_rows=len(tc), tree_marker_unmapped=int((~mapped).sum()),
        branch_reference_rows=len(b), eligible_nontrunk_branches=int(b.eligible_multimarker_reference.sum()),
        supported_eligible_nontrunk_branches=int((b.eligible_multimarker_reference & b.observed_multimarker_support).sum()),
        local_duplicate_region_mutation_rows=duplicate_region_mutations,
        clinical_relapse_categories=pa.Relapse_cat_new.value_counts(dropna=False).to_dict(),
        clinical_dfs_indicator=pa.cens_dfs.value_counts(dropna=False).to_dict(),
        semantics={'branch_truth': 'AUTHOR_RECONSTRUCTED_CLUSTER_REFERENCE_WITH_MULTIPLE_BRANCH_OWNED_MARKERS; NOT_DIRECTLY_OBSERVED_ANCESTRY_OR_THETA_TRUTH',
          'region_truth': 'Marker support is measured in actual regions; an additional region alone never establishes a recovered branch.',
          'clinical': 'Relapse_cat_new defines site categories. DFS indicator includes more events than recurrence and must not be silently recoded as recurrence.',
          'deduplication': 'Split/resample/uncertainty must use patient_id; multiple tumours, regions, clusters and markers are not independent patients.',
          'primary': 'Only author naming SU_T* / SU_LN* is classified; no post-recurrence metastatic regions exist in this native region file.',
          'negative': 'Not detected at a measured known variant locus is not gene-level callability or never-acquired biological absence.',
          'scorer_only': 'Complete trees, branch markers, hidden regions, phylogenetic CCF and future labels must be isolated from acquisition policies. Raw var_count/depth VAF may be used only after that region is queried.'})
    (out / 'AUDIT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ['source_files', 'semantics']}, ensure_ascii=False, indent=2))
    return summary


def audit_metastasis(processed: Path, native_out: Path, out: Path,
                     min_markers: int = 3, min_depth: int = 20, min_alt: int = 3) -> dict:
    """Crosswalk author seeding labels to native markers, without fitting or scoring policies."""
    out.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(processed / 'sampleOverview.txt', sep='\t')
    assert not s.region.duplicated().any()
    s['local_region'] = [r[len(t) + 1:] if r.startswith(t + '_')
                         else r[len(p) + 1:] if r.startswith(p + '_') else r
                         for r, t, p in zip(s.region, s.tumour_id, s.patient_id)]
    s['RegionID'] = s.tumour_id + ':' + s.local_region
    write(s, out, 'metastasis_regions.tsv')
    files = ['sampleOverview.txt', 'seedingTable.txt', 'seedingTableFull.txt',
             'seedingRegionInfo.txt', 'mutTableAll.cloneInfo.20220726.txt', 'fullTreeOutputList.RDS']
    provenance = {f: {'path': str(processed / f), 'bytes': (processed / f).stat().st_size,
                       'sha256': digest(processed / f)} for f in files}
    columns = ['mutation_id', 'patient_id', 'tumour_id', 'chr', 'start', 'ref', 'var',
               'Hugo_Symbol', 'RegionSum', 'Is.present', 'PyCloneCluster_SC',
               'cleanCluster_SC', 'DriverMut', 'treeClones', 'seedingClones']
    m = pd.read_csv(processed / 'mutTableAll.cloneInfo.20220726.txt', sep='\t', usecols=columns, low_memory=False)
    m.rename(columns={'Is.present': 'author_presence_string'}, inplace=True)
    m['variant_key'] = (m.tumour_id + ':' + m.chr.str.replace(r'^chr', '', regex=True)
        + ':' + m.start.astype(str) + ':' + m.ref + ':' + m['var'])
    assert not m[['tumour_id', 'variant_key']].duplicated().any()
    for c in ['cleanCluster_SC', 'DriverMut', 'treeClones', 'seedingClones']:
        m[c] = boolean(m[c])
    nk = pd.read_csv(native_out / 'branch_owned_markers.tsv.gz', sep='\t',
                     usecols=['tumour_id', 'variant_key', 'cluster'])
    nk = nk.rename(columns={'cluster': 'native_cluster'})
    m = m.merge(nk, on=['tumour_id', 'variant_key'], how='left', validate='one_to_one')
    m['native_cluster'] = m.native_cluster.astype('Int64').astype(str).replace('<NA>', '')
    corrected_mapping = out / 'metastasis_tree_marker_clusters.tsv.gz'
    if not corrected_mapping.exists():
        raise FileNotFoundError('Decode fullTreeOutputList.RDS with --decode-metastasis-trees first; raw PyClone labels are not silently substituted.')
    fk = pd.read_csv(corrected_mapping, sep='\t', dtype={'corrected_future_cluster': str})
    m = m.merge(fk[['tumour_id', 'variant_key', 'corrected_future_cluster']],
                on=['tumour_id', 'variant_key'], how='left', validate='one_to_one')
    m['future_cluster'] = m.corrected_future_cluster.fillna('')
    provenance['corrected_tree_marker_crosswalk'] = {'path': str(corrected_mapping),
        'sha256': digest(corrected_mapping)}
    write(m.drop(columns=['RegionSum', 'author_presence_string']), out, 'metastasis_marker_reference.tsv.gz')
    obs_rows = []
    for row in m.itertuples(index=False):
        present = {}
        if isinstance(row.author_presence_string, str):
            for a in row.author_presence_string.split(';'):
                if ':' in a:
                    k, v = a.rsplit(':', 1)
                    present[k] = {'TRUE': True, 'FALSE': False}.get(v)
        if not isinstance(row.RegionSum, str):
            continue
        for a in row.RegionSum.split(';'):
            if ':' not in a or '/' not in a:
                continue
            local, counts = a.rsplit(':', 1)
            try:
                alt, depth = (int(x) for x in counts.split('/'))
            except (ValueError, TypeError):
                continue
            known = present.get(local)
            obs_rows.append({'patient_id': row.patient_id, 'tumour_id': row.tumour_id,
                'RegionID': row.tumour_id + ':' + local, 'mutation_id': row.mutation_id,
                'variant_key': row.variant_key, 'native_cluster': row.native_cluster,
                'future_cluster': row.future_cluster,
                'gene': row.Hugo_Symbol, 'alt_reads': alt, 'depth': depth,
                'author_present': known, 'evaluable': known is not None and depth >= min_depth,
                'positive': known is True and depth >= min_depth and alt >= min_alt,
                'author_seeding_marker': row.seedingClones,
                'author_tree_marker': row.treeClones,
                'clean': row.cleanCluster_SC, 'author_driver_marker': row.DriverMut})
    o = pd.DataFrame(obs_rows)
    assert not o[['RegionID', 'variant_key']].duplicated().any()
    o = o.merge(s[['RegionID', 'sampleType', 'sampleTypeDetail']], on='RegionID', how='left', validate='many_to_one')
    write(o, out, 'scorer_metastasis_marker_regions.tsv.gz')
    future = o[o.sampleTypeDetail.eq('metachronousMet') & o.native_cluster.ne('')
               & o.future_cluster.ne('') & o.clean.eq(True) & o.author_tree_marker.eq(True)].copy()
    future['seeding_positive'] = future.positive & future.author_seeding_marker.eq(True)
    f = future.groupby(['patient_id', 'tumour_id', 'native_cluster', 'future_cluster', 'RegionID'], observed=True).agg(
        shared_own_markers=('variant_key', 'nunique'), markers_evaluable=('evaluable', 'sum'),
        markers_positive=('positive', 'sum'), seeding_markers_positive=('seeding_positive', 'sum')).reset_index()
    nb = pd.read_csv(native_out / 'branch_reference.tsv', sep='\t', dtype={'cluster': str})
    nb = nb.rename(columns={'cluster': 'native_cluster', 'is_trunk': 'native_is_trunk'})
    f = f.merge(nb[['tumour_id', 'native_cluster', 'native_is_trunk']], on=['tumour_id', 'native_cluster'], validate='many_to_one')
    f['multimarker_future_lineage_support'] = f.shared_own_markers.ge(min_markers) & f.markers_positive.ge(min_markers)
    f['multimarker_author_seeding_support'] = f.multimarker_future_lineage_support & f.seeding_markers_positive.ge(min_markers)
    write(f, out, 'native_cluster_future_marker_support.tsv')
    capacity = s.loc[s.sampleTypeDetail.eq('metachronousMet'), ['patient_id']].drop_duplicates()
    roster = pd.read_csv(native_out / 'patient_primary_tumour.tsv', sep='\t')
    capacity = capacity.merge(roster, on='patient_id', how='left', validate='one_to_one')
    split_path = native_out.parents[1] / 'configs' / 'PATIENT_SPLIT.tsv'
    if split_path.exists():
        capacity = capacity.merge(pd.read_csv(split_path, sep='\t'), on='patient_id', how='left', validate='one_to_one')
    supported_patients = set(f.loc[f.multimarker_future_lineage_support & ~f.native_is_trunk, 'patient_id'])
    capacity['nontrunk_multimarker_reference_available'] = capacity.patient_id.isin(supported_patients)
    capacity['reference_capacity_status'] = capacity.nontrunk_multimarker_reference_available.map({
        True: 'NONTRUNK_REFERENCE_AVAILABLE', False: 'NO_NONTRUNK_REFERENCE_NOT_A_BIOLOGICAL_NEGATIVE'})
    write(capacity, out, 'future_patient_reference_capacity.tsv')
    source_region = pd.read_csv(processed / 'seedingRegionInfo.txt', sep='\t')
    write(source_region, out, 'scorer_author_seeding_regions.tsv')
    summary = dict(source_files=provenance, patients=int(s.patient_id.nunique()),
        tumours=int(s.tumour_id.nunique()), regions=len(s),
        sample_types=s.sampleType.value_counts().to_dict(),
        sample_details=s.sampleTypeDetail.value_counts().to_dict(),
        metachronous_patients=int(s.loc[s.sampleTypeDetail.eq('metachronousMet'), 'patient_id'].nunique()),
        variant_rows=len(m), native_mapped_variants=int(m.native_cluster.ne('').sum()),
        author_seeding_true=int(m.seedingClones.eq(True).sum()),
        author_seeding_false=int(m.seedingClones.eq(False).sum()),
        author_seeding_unknown=int(m.seedingClones.isna().sum()),
        observed_variant_region_rows=len(o), region_roster_unmapped=int(o.sampleType.isna().sum()),
        native_cluster_future_supported_rows=int(f.multimarker_future_lineage_support.sum()),
        native_cluster_future_supported_patients=int(f.loc[f.multimarker_future_lineage_support, 'patient_id'].nunique()),
        native_nontrunk_future_supported_patients=int(f.loc[f.multimarker_future_lineage_support & ~f.native_is_trunk, 'patient_id'].nunique()),
        future_patient_capacity_table=str(out / 'future_patient_reference_capacity.tsv'),
        scope='SCORER_ONLY_AUTHORS_RECONSTRUCTED_SEEDING_AND_MULTI_MARKER_FUTURE_LINEAGE_REFERENCE;NOT_CAUSAL_THETA_TRUTH;LN_NOT_FUTURE_METASTASIS')
    (out / 'AUDIT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'source_files'}, indent=2))
    return summary


def decode_metastasis_trees(processed: Path, out: Path) -> None:
    """Read author RDS objects, preserving corrected cluster identity and all trees."""
    import warnings
    import rdata
    out.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Missing constructor for R class "table"')
        trees = rdata.read_rds(processed / 'fullTreeOutputList.RDS')
    edges, maps, inventory = [], [], []
    for tid, tree in trees.items():
        graph = tree['graph_pyclone']
        e = np.asarray(graph['Corrected_tree'])
        # Some R character matrices lose dim during generic conversion. Restore
        # the R column-major parent/child order explicitly, not a row-wise guess.
        if e.ndim == 1:
            assert e.size % 2 == 0
            e = e.reshape((-1, 2), order='F')
        assert e.shape[1] == 2
        trunk = [str(x) for x in np.asarray(graph['trunk']).ravel()]
        nodes = set(str(x) for x in e.ravel()) | set(trunk)
        for parent, child in e:
            edges.append({'tumour_id': tid, 'parent': str(parent), 'child': str(child)})
        cf = tree['ccf_table_pyclone_clean']
        assert isinstance(cf, pd.DataFrame)
        cluster_col = next(c for c in cf.columns if c.lower() == 'pyclonecluster')
        for key, cluster in zip(cf.index, cf[cluster_col]):
            c = str(int(cluster)) if isinstance(cluster, (int, float, np.number)) and pd.notna(cluster) else str(cluster)
            if c not in nodes:
                continue
            canonical = tid + ':' + str(key).split(':', 1)[1]
            maps.append({'tumour_id': tid, 'variant_key': canonical,
                         'author_variant_key': str(key), 'corrected_future_cluster': c})
        inventory.append({'tumour_id': tid, 'trunk': ';'.join(trunk),
                          'corrected_edges': len(e), 'corrected_nodes': len(nodes)})
    mapping = pd.DataFrame(maps)
    assert not mapping[['tumour_id', 'variant_key']].duplicated().any()
    write(pd.DataFrame(edges), out, 'metastasis_corrected_tree_edges.tsv')
    write(mapping, out, 'metastasis_tree_marker_clusters.tsv.gz')
    write(pd.DataFrame(inventory), out, 'metastasis_tree_inventory.tsv')
    (out / 'TREE_DECODE.json').write_text(json.dumps({'rdata_version': rdata.__version__,
        'source_sha256': digest(processed / 'fullTreeOutputList.RDS'),
        'trees': len(trees), 'edges': len(edges), 'mapped_markers': len(mapping),
        'semantics': 'AUTHOR_CORRECTED_TREES_REFERENCE_ONLY;ALTERNATIVE_TOPOLOGIES_NOT_CLAIMED_RESOLVED'}, indent=2))
    print('DECODED_METASTASIS_TREES', len(trees), 'EDGES', len(edges), 'MARKERS', len(mapping))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--native', type=Path, default=DEFAULT_NATIVE)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--min-markers', type=int, default=3)
    p.add_argument('--min-depth', type=int, default=20)
    p.add_argument('--min-alt', type=int, default=3)
    p.add_argument('--metastasis', type=Path)
    p.add_argument('--metastasis-only', action='store_true')
    p.add_argument('--decode-metastasis-trees', action='store_true')
    a = p.parse_args()
    if not a.metastasis_only:
        audit(a.native, a.out, a.min_markers, a.min_depth, a.min_alt)
    if a.metastasis:
        if a.decode_metastasis_trees:
            decode_metastasis_trees(a.metastasis, a.out.parent / 'metastasis_audit')
        audit_metastasis(a.metastasis, a.out, a.out.parent / 'metastasis_audit', a.min_markers, a.min_depth, a.min_alt)
