#!/usr/bin/env python3
"""Acquire and audit public TRACERx ctDNA supplementary inputs; no model fitting."""
import argparse
import concurrent.futures
import csv
import datetime
import gzip
import hashlib
import json
import re
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import openpyxl
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ctdna"
PAPER = "https://www.nature.com/articles/s41586-023-05776-4"
MEDIA = "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-023-05776-4/MediaObjects/"
SOURCES = {
    "ctdna_article.html": PAPER,
    "41586_2023_5776_MOESM4_ESM.xlsx": MEDIA + "41586_2023_5776_MOESM4_ESM.xlsx",
    "41586_2023_5776_MOESM5_ESM.docx": MEDIA + "41586_2023_5776_MOESM5_ESM.docx",
    "41586_2023_5776_MOESM1_ESM.pdf": MEDIA + "41586_2023_5776_MOESM1_ESM.pdf",
}
EXPORT_SHEETS = ["Table_1_Variant_comp", "Table_2_sample_cfDNA", "Table_4_ddPCR_data",
                 "Table_8_patient_data", "Table_13_sample_ctDNA", "Table_14_relapsesite",
                 "Table_15_treatment", "Table_16_scan_data", "Table_17_patient_mutation_data",
                 "Table_19_PBMC_samples", "Table_20_PBMC_variant_data", "Table_21_EGA_sample_identifiers"]


def find_header(ws):
    for rownum, row in enumerate(ws.iter_rows(max_row=100, values_only=True), 1):
        values = [v for v in row if v is not None]
        if len(values) < 3:
            continue
        if row[0] in ("PublicationID", "tracerx_id") and "Patient identifier" not in values:
            last = max(i for i, v in enumerate(row) if v is not None)
            return rownum, [str(v) if v is not None else "unnamed_" + str(i) for i, v in enumerate(row[:last + 1])]
    raise ValueError("No explicit data header found in " + ws.title)


def export_and_count(wb):
    manifest = []
    for name in EXPORT_SHEETS:
        ws = wb[name]
        header_row, header = find_header(ws)
        path = DATA / "processed" / (name + ".tsv.gz")
        count = 0
        with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)
            for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
                if row[0] is None:
                    continue
                writer.writerow(["" if v is None else serializable(v) for v in row[:len(header)]])
                count += 1
        record = {"sheet": name, "source_header_row": header_row, "rows": count,
                  "columns": header, "path": str(path), "sha256": sha256(path)}
        manifest.append(record)
        print(json.dumps({"export": name, "rows": count, "header_row": header_row}), flush=True)
    (DATA / "audit" / "EXTRACTED_TABLES.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    clinical = pd.read_csv(DATA / "processed" / "Table_8_patient_data.tsv.gz", sep="\t")
    samples = pd.read_csv(DATA / "processed" / "Table_13_sample_ctDNA.tsv.gz", sep="\t")
    markers = pd.read_csv(DATA / "processed" / "Table_17_patient_mutation_data.tsv.gz", sep="\t", low_memory=False)
    normal = pd.read_csv(DATA / "processed" / "Table_20_PBMC_variant_data.tsv.gz", sep="\t")
    summary = {"clinical_patients": int(clinical.PublicationID.nunique()),
               "clinical_records": len(clinical), "sample_records": len(samples),
               "sample_patients": int(samples.PublicationID.nunique()),
               "sample_snp_match_counts": samples.matches_wes.value_counts(dropna=False).astype(int).to_dict(),
               "marker_records": len(markers), "marker_patients": int(markers.PublicationID.nunique()),
               "marker_samples": int(markers.tracerx_id.nunique()),
               "marker_sample_key_duplicates": int(markers.duplicated(["tracerx_id", "chromosome", "position", "reference", "alternate"]).sum()),
               "marker_missing_dao": int(markers.dao.isna().sum()), "marker_missing_ddp": int(markers.ddp.isna().sum()),
               "marker_missing_days": int(markers.days_post_surgery.isna().sum()),
               "marker_day_min": float(markers.days_post_surgery.min()), "marker_day_max": float(markers.days_post_surgery.max()),
               "marker_patient_not_in_clinical": sorted(set(markers.PublicationID) - set(clinical.PublicationID)),
               "normal_records": len(normal), "normal_controls": int(normal.tracerx_id.nunique()),
               "normal_patient_ids": sorted(normal.PublicationID.unique()),
               "notes": ["Data audit only; no model fitting, threshold calibration or endpoint-based selection.",
                         "doi:10.1038/s41586-023-05776-4 public processed count tables; original raw sequencing remains EGA-controlled."]}
    (DATA / "audit" / "DATA_GRAIN_AUDIT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("GRAIN\n" + json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    audit_marker_metadata(markers, normal, samples)


def audit_marker_metadata(markers=None, normal=None, samples=None):
    if markers is None:
        markers = pd.read_csv(DATA / "processed" / "Table_17_patient_mutation_data.tsv.gz", sep="\t", low_memory=False)
        normal = pd.read_csv(DATA / "processed" / "Table_20_PBMC_variant_data.tsv.gz", sep="\t")
        samples = pd.read_csv(DATA / "processed" / "Table_13_sample_ctDNA.tsv.gz", sep="\t")
    targets = markers.drop_duplicates(["PublicationID", "chromosome", "position", "reference", "alternate"])
    metadata = {"marker_target_count": len(targets),
                "tissue_keys": sorted(set(region.split(":", 1)[0] for s in targets.var_ref_counts.dropna() for region in s.split(";"))),
                "tissue_marker_nonSU_count": int(targets.var_ref_counts.fillna("").str.contains(r"(?:^|;)(?!SU_)[^;:]+:", regex=True).sum()),
                "clonality_counts": targets.PyCloneClonal_SC.fillna("missing").value_counts().astype(int).to_dict(),
                "normal_ids": normal.groupby(["PublicationID", "tracerx_id"]).size().reset_index(name="markers").to_dict("records")}
    (DATA / "audit" / "MARKER_METADATA_AUDIT.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    filters = {"clinical_filters_counts": {str(k): int(v) for k, v in markers["failed.filters"].value_counts(dropna=False).items()},
               "normal_filters_counts": {str(k): int(v) for k, v in normal["failed filters"].value_counts(dropna=False).items()}}
    (DATA / "audit" / "QUALITY_FILTER_METADATA.json").write_text(json.dumps(filters, indent=2), encoding="utf-8")
    panels = pd.read_csv(DATA / "processed" / "Table_1_Variant_comp.tsv.gz", sep="\t")
    capacity = {"table1_patient_n": len(panels),
                "capacity_by_budget": {str(k): {"patients_clonal_below": int((panels.Clonal < k).sum()),
                                               "patients_total_below": int((panels[["Clonal", "Subclonal", "Unknown"]].sum(axis=1) < k).sum())} for k in [20, 50, 100]},
                "sample_total": len(samples),
                "sample_snp_match_counts": {str(k): int(v) for k, v in samples.matches_wes.value_counts(dropna=False).items()},
                "sample_snp_matched_patient_n": int(samples.loc[samples.matches_wes == True, "PublicationID"].nunique()),
                "day_missing": int(samples.days_post_surgery.isna().sum()),
                "day0_samples": int((samples.days_post_surgery == 0).sum()),
                "patient_ids_without_day0": sorted(set(samples.PublicationID) - set(samples.loc[samples.days_post_surgery == 0, "PublicationID"]))}
    (DATA / "audit" / "EARLY_CAPACITY_AUDIT.json").write_text(json.dumps(capacity, indent=2), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def fetch(item):
    name, url = item
    dest = DATA / "raw" / name
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".part")
        response = requests.get(url, stream=True, timeout=(15, 120))
        response.raise_for_status()
        with tmp.open("wb") as f:
            for b in response.iter_content(1024 * 1024):
                f.write(b)
        tmp.replace(dest)
        http = {"status": response.status_code, "final_url": response.url,
                "content_type": response.headers.get("Content-Type")}
    else:
        http = {"status": "existing_input_verified_by_hash", "final_url": url}
    receipt = {"source_url": url, "path": str(dest), "size_bytes": dest.stat().st_size,
               "sha256": sha256(dest), "retrieved_or_verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **http}
    (DATA / "raw" / (name + ".receipt.json")).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"input": name, "bytes": receipt["size_bytes"], "sha256": receipt["sha256"]}), flush=True)
    return receipt


def serializable(value):
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    return value


def audit():
    book = DATA / "raw" / "41586_2023_5776_MOESM4_ESM.xlsx"
    wb = openpyxl.load_workbook(book, read_only=True, data_only=True)
    sheets = []
    for ws in wb.worksheets:
        rows = []
        for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row or 100, 100), values_only=True):
            rows.append([serializable(v) for v in row])
        item = {"sheet": ws.title, "declared_max_row": ws.max_row, "declared_max_column": ws.max_column,
                "first_100_rows": rows}
        sheets.append(item)
        print(json.dumps({k: item[k] for k in ["sheet", "declared_max_row", "declared_max_column"]}), flush=True)
    (DATA / "audit" / "WORKBOOK_SCHEMA.json").write_text(json.dumps(sheets, ensure_ascii=False, indent=2), encoding="utf-8")
    export_and_count(wb)
    wb.close()
    doc = DATA / "raw" / "41586_2023_5776_MOESM5_ESM.docx"
    with zipfile.ZipFile(doc) as z:
        xml = ET.fromstring(z.read("word/document.xml"))
    guide = "\n".join("".join(p.itertext()) for p in xml.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"))
    (DATA / "audit" / "SUPPLEMENTARY_TABLE_GUIDE.txt").write_text(guide, encoding="utf-8")
    print("GUIDE\n" + guide, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    for d in [DATA / "raw", DATA / "audit", DATA / "processed"]:
        d.mkdir(parents=True, exist_ok=True)
    if args.metadata_only:
        audit_marker_metadata()
        return
    if not args.audit_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            receipts = list(pool.map(fetch, SOURCES.items()))
        (DATA / "SOURCE_FILES.json").write_text(json.dumps(receipts, indent=2), encoding="utf-8")
    audit()


if __name__ == "__main__":
    main()
