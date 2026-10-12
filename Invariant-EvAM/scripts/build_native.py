"""Build optional local native artifacts; never silently replace the frozen G2v3."""
from pathlib import Path
import argparse,subprocess,sys
root=Path(__file__).resolve().parents[1]/"paper_experiments_v1"
parser=argparse.ArgumentParser()
parser.add_argument("--build-g3-g5",action="store_true")
parser.add_argument("--build-g2-v4",action="store_true")
parser.add_argument("--cc",default="cc")
args=parser.parse_args()
targets=[]
if args.build_g3_g5:
    targets += [("g3","ctmc_generic.c","ctmc_generic.dylib"),("g5","ctmc_g5.c","ctmc_g5.dylib")]
if args.build_g2_v4:
    targets += [("g2","ctmc_fast_v4.c","ctmc_fast_v4.dylib")]
if not targets:parser.error("Choose --build-g3-g5 or --build-g2-v4")
for group,source,name in targets:
    folder=root/group/"src"
    cmd=[args.cc,"-O3","-shared","-fPIC",str(folder/source),"-o",str(folder/name),"-lm"]
    print(" ".join(cmd),flush=True)
    subprocess.run(cmd,check=True)
print("Built local native artifacts. These are not frozen source-run binary hashes.")
