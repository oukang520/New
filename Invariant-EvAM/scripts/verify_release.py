"""Read-only verification of the selected current release, using standard library."""
from pathlib import Path
import csv,hashlib,json,sys
root=Path(__file__).resolve().parents[1];p=root/"paper_experiments_v1"
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def nested(v):
    if isinstance(v,dict):
        if "path" in v and "sha256" in v:yield v["path"],v["sha256"]
        for x in v.values():yield from nested(x)
    elif isinstance(v,list):
        for x in v:yield from nested(x)
errors=[];panels=0
for group,count in {"g1":8,"g2":9,"g3":10,"g4":18,"g5":14}.items():
    f=p/group/"final_results";m=json.loads((f/"MANIFEST.json").read_text());expect=set()
    if len(m["panels"])!=count:errors.append(group+":panel count")
    panels+=len(m["panels"])
    for row in m["panels"]:
        for key,hash_key in [("pdf","pdf_sha256"),("svg","svg_sha256"),("plot_data","data_sha256")]:
            target=(f/row[key]).resolve()
            if not target.is_file() or sha(target)!=row[hash_key]:errors.append(group+":"+row["panel"]+":"+key)
            if key in ["pdf","svg"]:expect.add(target)
    actual={x.resolve() for x in (p/group/"final_figures").iterdir()}
    if actual!=expect:errors.append(group+":extra/missing figure files")
    for path,digest in nested(m):
        target=f/path
        if not target.is_file() or sha(target)!=digest:errors.append(group+":"+path)
    registry=json.loads((p/group/"configs/FINAL_FIGURES_MANIFEST.json").read_text())
    for row in registry["files"]:
        target=p/row["destination"]
        if not target.is_file() or sha(target)!=row["sha256"]:errors.append(group+":registry:"+row["destination"])
with (root/"provenance/SOURCE_MAP.tsv").open(newline="") as f:
    for row in csv.DictReader(f,delimiter="\t"):
        target=root/row["package_path"]
        if not target.is_file() or sha(target)!=row["package_sha256"]:errors.append("source map:"+row["package_path"])
print(json.dumps({"status":"PASS" if not errors else "FAIL","panels":panels,"figure_files":panels*2,"errors":errors},ensure_ascii=False,indent=2))
sys.exit(bool(errors))
