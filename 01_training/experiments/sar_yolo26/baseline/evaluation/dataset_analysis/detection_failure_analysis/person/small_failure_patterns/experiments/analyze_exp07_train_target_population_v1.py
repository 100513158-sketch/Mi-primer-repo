from __future__ import annotations
import csv, math, sys
from collections import Counter
from pathlib import Path
from PIL import Image

PERSON_CLASS_ID = 0
SMALL_AREA_THRESHOLD = 256.0
EXTREME_SMALL_THRESHOLD = 16.0
DENSE_PERSON_COUNT = 25
NEIGHBOR_DISTANCE_FACTOR = 2.0
EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

SCRIPT = Path(__file__).resolve()
BASELINE = next((x for x in [SCRIPT.parent,*SCRIPT.parents] if x.name.lower()=="baseline"), None)
if BASELINE is None:
    raise RuntimeError("No se encontró baseline.")

ROOT = next((p for p in [BASELINE, *BASELINE.parents] if p.name == "SARC-Drone"), BASELINE)
DATASET = ROOT/"00_datasets"/"SAR_DATASET_STUDIO"/"processed"/"sar"/"cleaned"/"VisDrone_SAR_2CLASS_V1"
TRAIN_I = DATASET/"train"/"images"
TRAIN_L = DATASET/"train"/"labels"
OUT = BASELINE/"evaluation"/"dataset_analysis"/"detection_failure_analysis"/"person"/"small_failure_patterns"/"experiments"/"exp07_train_target_population_analysis_v1"/"reports"
OBJ = OUT/"exp07_train_target_population_objects_v1.csv"
IMG = OUT/"exp07_train_target_population_images_v1.csv"
SUM = OUT/"EXP07_TRAIN_TARGET_POPULATION_ANALYSIS_V1_SUMMARY.txt"

def xywhn(xc,yc,w,h,iw,ih):
    cx,cy,bw,bh=xc*iw,yc*ih,w*iw,h*ih
    return [max(0,cx-bw/2),max(0,cy-bh/2),min(iw,cx+bw/2),min(ih,cy+bh/2)]
def dist(a,b):
    return math.hypot((a[0]+a[2])-(b[0]+b[2]), (a[1]+a[3])-(b[1]+b[3]))/2
def load(label,iw,ih):
    out=[]
    if not label.exists(): return out
    try: lines=label.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError: lines=label.read_text(encoding="latin-1").splitlines()
    for i,line in enumerate(lines):
        q=line.split()
        if len(q)<5: continue
        try: cls=int(float(q[0])); xc,yc,w,h=map(float,q[1:5])
        except ValueError: continue
        if cls!=PERSON_CLASS_ID or w<=0 or h<=0: continue
        area=w*h*iw*ih
        if area>=SMALL_AREA_THRESHOLD: continue
        out.append({"gt_index":i,"box":xywhn(xc,yc,w,h,iw,ih),"area":area,"size_sqrt":math.sqrt(area)})
    return out
def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    for p in (TRAIN_I,TRAIN_L):
        if not p.exists(): raise FileNotFoundError(str(p))
    images=sorted(x for x in TRAIN_I.iterdir() if x.is_file() and x.suffix.lower() in EXTS)
    objects=[]; image_rows=[]; type_counts=Counter(); target_images=triple_images=0
    for n,img in enumerate(images,1):
        label=TRAIN_L/f"{img.stem}.txt"
        if not label.exists(): continue
        with Image.open(img) as im: iw,ih=im.width,im.height
        persons=load(label,iw,ih)
        if not persons: continue
        dense=len(persons)>=DENSE_PERSON_COUNT
        tcount=0; triple=0
        for person in persons:
            ds=[dist(person["box"],o["box"]) for o in persons if o["gt_index"]!=person["gt_index"]]
            nearest=min(ds) if ds else float("inf")
            extreme=person["size_sqrt"]<EXTREME_SMALL_THRESHOLD
            close=nearest<=NEIGHBOR_DISTANCE_FACTOR*max(person["size_sqrt"],1.0)
            target=extreme and (dense or close)
            tri=extreme and dense and close
            if target: tcount+=1; type_counts["TRIPLE" if tri else ("EXTREME_SMALL_DENSE" if dense else "EXTREME_SMALL_NEIGHBOR")]+=1
            if tri: triple+=1
            objects.append({"image":img.name,"gt_index":person["gt_index"],"area":round(person["area"],4),"size_sqrt":round(person["size_sqrt"],4),"person_count":len(persons),"dense_scene":int(dense),"extreme_small":int(extreme),"nearest_distance":"" if not math.isfinite(nearest) else round(nearest,4),"close_neighbors":int(close),"target_any":int(target),"target_triple":int(tri),"target_type":"TRIPLE" if tri else ("EXTREME_SMALL_DENSE" if target and dense else "EXTREME_SMALL_NEIGHBOR" if target else "SMALL_OTHER")})
        if tcount: target_images+=1
        if triple: triple_images+=1
        image_rows.append({"image":img.name,"small_person_count":len(persons),"target_any_count":tcount,"target_triple_count":triple,"dense_scene":int(dense),"has_target":int(tcount>0),"has_triple":int(triple>0)})
        if n%500==0 or n==len(images): print(f"Procesadas {n:,}/{len(images):,} | SMALL={len(objects):,} | TARGET={sum(r['target_any'] for r in objects):,} | TRIPLE={sum(r['target_triple'] for r in objects):,}")
    small=len(objects); target=sum(r["target_any"] for r in objects); tri=sum(r["target_triple"] for r in objects)
    OUT.mkdir(parents=True,exist_ok=True); write_csv(OBJ,objects); write_csv(IMG,image_rows)
    lines=["="*72,"SAR YOLO26 - EXP07 TRAIN TARGET POPULATION ANALYSIS V1","="*72,"",
           f"TRAIN images: {len(images):,}",f"SMALL PERSON: {small:,}",f"TARGET UNION: {target:,}",f"TARGET TRIPLE: {tri:,}",
           f"TARGET/SMALL: {100*target/small:.2f}%" if small else "TARGET/SMALL: 0.00%",
           f"TRIPLE/SMALL: {100*tri/small:.2f}%" if small else "TRIPLE/SMALL: 0.00%",
           f"TRIPLE/TARGET: {100*tri/target:.2f}%" if target else "TRIPLE/TARGET: 0.00%","",
           "TARGET TYPES"]
    lines += [f"{k}: {v:,}" for k,v in type_counts.most_common()]
    lines += ["","DECISION","Use at most one crop per GT target. Prioritize TRIPLE first; expand to UNION only if necessary.",
              "No crops generated. Dataset, labels and official YAML unchanged."]
    SUM.write_text("\n".join(lines),encoding="utf-8")
    print("="*72); print("# RESULTADO EXP07 TRAIN TARGET POPULATION"); print("="*72)
    print(f"SMALL PERSON: {small:,}"); print(f"TARGET UNION: {target:,}"); print(f"TARGET TRIPLE: {tri:,}")
    print(f"TARGET/SMALL: {100*target/small:.2f}%" if small else "TARGET/SMALL: 0.00%")
    print(f"TRIPLE/SMALL: {100*tri/small:.2f}%" if small else "TRIPLE/SMALL: 0.00%")
    print(f"[OK] {OBJ}"); print(f"[OK] {IMG}"); print(f"[OK] {SUM}")

if __name__=="__main__":
    try: main()
    except Exception as e: print("[ERROR EXP07]",e); sys.exit(1)
