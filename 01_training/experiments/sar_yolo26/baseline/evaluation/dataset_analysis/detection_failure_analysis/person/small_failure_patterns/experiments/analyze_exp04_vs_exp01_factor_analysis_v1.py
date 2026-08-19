from __future__ import annotations
import csv, math, sys
from pathlib import Path
from collections import defaultdict
from PIL import Image
from ultralytics import YOLO

PERSON_CLASS_ID = 0
SMALL_AREA_THRESHOLD = 256.0
EDGE_MARGIN_RATIO = 0.10
DENSE_PERSON_COUNT = 25
NEIGHBOR_DISTANCE_FACTOR = 2.0
MATCH_IOU_THRESHOLD = 0.50
EVAL_IMAGE_SIZE = 1536
CONF_THRESHOLD = 0.25
IMAGE_EXTENSIONS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

def find_project_root():
    for p in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if p.name.lower() == "sarc-drone":
            return p
    raise RuntimeError("No se pudo localizar C:\\SARC-Drone.")

PROJECT_ROOT = find_project_root()
BASELINE_DIR = PROJECT_ROOT / "01_training" / "experiments" / "sar_yolo26" / "baseline"
DATASET_ROOT = PROJECT_ROOT / "00_datasets" / "SAR_DATASET_STUDIO" / "processed" / "sar" / "cleaned" / "VisDrone_SAR_2CLASS_V1"
TEST_IMAGES_DIR = DATASET_ROOT / "test_dev" / "images"
TEST_LABELS_DIR = DATASET_ROOT / "test_dev" / "labels"

EXP01_MODEL = BASELINE_DIR / "training" / "runs" / "baseline_v1" / "weights" / "best.pt"
EXP04_MODEL = BASELINE_DIR / "training" / "experiments" / "exp04_dense_scene_targeted_crops_v1" / "runs" / "exp04_dense_scene_targeted_crops" / "weights" / "best.pt"

REPORTS_DIR = BASELINE_DIR / "evaluation" / "dataset_analysis" / "detection_failure_analysis" / "person" / "small_failure_patterns" / "experiments" / "exp04_vs_exp01_factor_analysis_v1" / "reports"
OBJECTS_CSV = REPORTS_DIR / "exp04_vs_exp01_objects_v1.csv"
FACTORS_CSV = REPORTS_DIR / "exp04_vs_exp01_factor_metrics_v1.csv"
INTERACTIONS_CSV = REPORTS_DIR / "exp04_vs_exp01_interactions_v1.csv"
SIZE_CSV = REPORTS_DIR / "exp04_vs_exp01_size_metrics_v1.csv"
SUMMARY_TXT = REPORTS_DIR / "EXP04_VS_EXP01_FACTOR_ANALYSIS_V1_SUMMARY.txt"

def safe_div(a,b): return 0.0 if b == 0 else a/b

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1=max(ax1,bx1),max(ay1,by1); ix2,iy2=min(ax2,bx2),min(ay2,by2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    aa=max(0,ax2-ax1)*max(0,ay2-ay1); ab=max(0,bx2-bx1)*max(0,by2-by1)
    u=aa+ab-inter
    return 0.0 if u<=0 else inter/u

def xywhn_to_xyxy(xc,yc,w,h,iw,ih):
    cx,cy=xc*iw,yc*ih; bw,bh=w*iw,h*ih
    return [max(0,cx-bw/2),max(0,cy-bh/2),min(iw,cx+bw/2),min(ih,cy+bh/2)]

def center_distance(a,b):
    return math.hypot((a[0]+a[2])/2-(b[0]+b[2])/2,(a[1]+a[3])/2-(b[1]+b[3])/2)

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def load_small_person_gt(label_path,iw,ih):
    out=[]
    if not label_path.exists(): return out
    try: lines=label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError: lines=label_path.read_text(encoding="latin-1").splitlines()
    for idx,line in enumerate(lines):
        p=line.strip().split()
        if len(p)<5: continue
        try: cls=int(float(p[0])); xc,yc,w,h=map(float,p[1:5])
        except ValueError: continue
        if cls!=PERSON_CLASS_ID or w<=0 or h<=0: continue
        area=w*h*iw*ih
        if area>=SMALL_AREA_THRESHOLD: continue
        out.append({"gt_index":idx,"box":xywhn_to_xyxy(xc,yc,w,h,iw,ih),"area":area,"size_sqrt":math.sqrt(area)})
    return out

def factor_flags(persons,target,iw,ih):
    x1,y1,x2,y2=target["box"]; mx,my=iw*EDGE_MARGIN_RATIO,ih*EDGE_MARGIN_RATIO
    edge=int(x1<=mx or y1<=my or iw-x2<=mx or ih-y2<=my)
    extreme=int(target["size_sqrt"]<16)
    distances=[center_distance(target["box"],o["box"]) for o in persons if o["gt_index"]!=target["gt_index"]]
    nearest=min(distances) if distances else float("inf")
    close=int(nearest<=NEIGHBOR_DISTANCE_FACTOR*max(target["size_sqrt"],1.0))
    dense=int(len(persons)>=DENSE_PERSON_COUNT)
    sides=[]
    if edge:
        vals={"LEFT":x1,"TOP":y1,"RIGHT":iw-x2,"BOTTOM":ih-y2}
        sides=[min(vals,key=vals.get)]
    return {"EXTREME_SMALL":extreme,"EDGE_LOCATION":edge,"CLOSE_NEIGHBORS":close,"DENSE_SCENE":dense,"EDGE_SIDE":sides[0] if sides else "NONE","nearest_distance":nearest}

def predict(model,image_path):
    res=model.predict(source=str(image_path),imgsz=EVAL_IMAGE_SIZE,conf=CONF_THRESHOLD,device=0,verbose=False,save=False)
    if not res or res[0].boxes is None: return [],[]
    b=res[0].boxes; boxes=b.xyxy.cpu().tolist(); confs=b.conf.cpu().tolist(); classes=b.cls.cpu().tolist()
    keep=[(box,float(c)) for box,c,cls in zip(boxes,confs,classes) if int(cls)==PERSON_CLASS_ID]
    return [x[0] for x in keep],[x[1] for x in keep]

def match(gt,preds,confs):
    used=set(); out=[]
    for g in gt:
        best=0.0; bi=None
        for j,p in enumerate(preds):
            if j in used: continue
            sc=iou(g["box"],p)
            if sc>best: best,bi=sc,j
        ok=bi is not None and best>=MATCH_IOU_THRESHOLD
        if ok: used.add(bi)
        out.append({"tp":int(ok),"iou":best,"conf":confs[bi] if ok else 0.0})
    return out

def build_rows():
    m01=YOLO(str(EXP01_MODEL)); m04=YOLO(str(EXP04_MODEL))
    images=sorted(p for p in TEST_IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    rows=[]
    for i,img in enumerate(images,1):
        label=TEST_LABELS_DIR/f"{img.stem}.txt"
        try:
            with Image.open(img) as im: iw,ih=im.width,im.height
        except Exception as exc:
            print(f"[WARNING] {img.name}: {exc}"); continue
        gt=load_small_person_gt(label,iw,ih)
        if not gt: continue
        flags=[factor_flags(gt,g,iw,ih) for g in gt]
        p01,c01=predict(m01,img); p04,c04=predict(m04,img)
        x01=match(gt,p01,c01); x04=match(gt,p04,c04)
        for g,f,a,b in zip(gt,flags,x01,x04):
            rows.append({
                "image":img.name,"gt_index":g["gt_index"],"area":round(g["area"],6),"size_sqrt":round(g["size_sqrt"],6),
                "edge_side":f["EDGE_SIDE"],"nearest_distance":"" if not math.isfinite(f["nearest_distance"]) else round(f["nearest_distance"],6),
                "EXTREME_SMALL":f["EXTREME_SMALL"],"EDGE_LOCATION":f["EDGE_LOCATION"],"CLOSE_NEIGHBORS":f["CLOSE_NEIGHBORS"],"DENSE_SCENE":f["DENSE_SCENE"],
                "EXP01_TP":a["tp"],"EXP01_iou":round(a["iou"],6),"EXP01_conf":round(a["conf"],6),
                "EXP04_TP":b["tp"],"EXP04_iou":round(b["iou"],6),"EXP04_conf":round(b["conf"],6),
                "tp_delta":b["tp"]-a["tp"],
                "transition":"FN_TO_TP" if a["tp"]==0 and b["tp"]==1 else "TP_TO_FN" if a["tp"]==1 and b["tp"]==0 else "TP_TO_TP" if a["tp"]==1 else "FN_TO_FN"
            })
        if i%100==0 or i==len(images): print(f"Analizadas: {i:,}/{len(images):,} | SMALL GT: {len(rows):,}")
    if not rows: raise RuntimeError("No se obtuvieron SMALL PERSON.")
    return rows

def metrics(rows):
    gt=len(rows); tp01=sum(r["EXP01_TP"] for r in rows); tp04=sum(r["EXP04_TP"] for r in rows)
    return {
        "gt":gt,"EXP01_tp":tp01,"EXP01_fn":gt-tp01,"EXP01_recall_pct":safe_div(tp01,gt)*100,
        "EXP04_tp":tp04,"EXP04_fn":gt-tp04,"EXP04_recall_pct":safe_div(tp04,gt)*100,
        "delta_pp":safe_div(tp04-tp01,gt)*100,"tp_gain":tp04-tp01,
        "fn_to_tp":sum(r["EXP01_TP"]==0 and r["EXP04_TP"]==1 for r in rows),
        "tp_to_fn":sum(r["EXP01_TP"]==1 and r["EXP04_TP"]==0 for r in rows)
    }

def factor_report(rows):
    factors=["OVERALL","EXTREME_SMALL","EDGE_LOCATION","CLOSE_NEIGHBORS","DENSE_SCENE"]; out=[]
    for f in factors:
        sub=rows if f=="OVERALL" else [r for r in rows if r[f]==1]
        m=metrics(sub); m["factor"]=f; m["subset_pct"]=safe_div(len(sub),len(rows))*100; out.append(m)
    return out

def interaction_report(rows):
    fs=["EXTREME_SMALL","EDGE_LOCATION","CLOSE_NEIGHBORS","DENSE_SCENE"]; out=[]
    for i,a in enumerate(fs):
        for b in fs[i+1:]:
            sub=[r for r in rows if r[a] and r[b]]
            if not sub: continue
            m=metrics(sub); m["interaction"]=f"{a} + {b}"; m["subset_pct"]=safe_div(len(sub),len(rows))*100; out.append(m)
    out.sort(key=lambda r:r["delta_pp"],reverse=True); return out

def size_report(rows):
    buckets=[("<16",lambda s:s<16),("16-32",lambda s:16<=s<32),("32-64",lambda s:32<=s<64),("64-128",lambda s:64<=s<128),("128-256",lambda s:128<=s<256)]
    out=[]
    for name,fn in buckets:
        sub=[r for r in rows if fn(float(r["size_sqrt"]))]
        if sub:
            m=metrics(sub); m["size_bucket"]=name; out.append(m)
    return out

def main():
    print("="*72); print("# SAR YOLO26 - EXP04 VS EXP01 FACTOR ANALYSIS V1"); print("="*72)
    REPORTS_DIR.mkdir(parents=True,exist_ok=True)
    for name,p in {"EXP01_MODEL":EXP01_MODEL,"EXP04_MODEL":EXP04_MODEL,"DATASET_ROOT":DATASET_ROOT,"TEST_IMAGES_DIR":TEST_IMAGES_DIR,"TEST_LABELS_DIR":TEST_LABELS_DIR}.items():
        if not p.exists(): raise FileNotFoundError(f"No se encontró {name}:\n{p}")
        print(f"[OK] {name}\n     {p}")
    rows=build_rows(); factors=factor_report(rows); interactions=interaction_report(rows); sizes=size_report(rows)
    write_csv(OBJECTS_CSV,rows); write_csv(FACTORS_CSV,factors); write_csv(INTERACTIONS_CSV,interactions); write_csv(SIZE_CSV,sizes)
    overall=factors[0]; dense=next(r for r in factors if r["factor"]=="DENSE_SCENE"); extreme=next(r for r in factors if r["factor"]=="EXTREME_SMALL"); edge=next(r for r in factors if r["factor"]=="EDGE_LOCATION")
    lines=["="*72,"SAR YOLO26 - EXP04 VS EXP01 FACTOR ANALYSIS V1","="*72,"","OBJETIVO","Explicar de dónde procede el +3.04 pp de EXP04 frente a EXP01.","",
           f"GLOBAL EXP01: {overall['EXP01_recall_pct']:.2f}%",f"GLOBAL EXP04: {overall['EXP04_recall_pct']:.2f}%",f"DELTA: {overall['delta_pp']:+.2f} pp",f"TP GAIN: {overall['tp_gain']:+d}",f"FN->TP: {overall['fn_to_tp']:,}",f"TP->FN: {overall['tp_to_fn']:,}","",
           "FACTORES","-"*72]
    for r in factors:
        lines.append(f"{r['factor']:<20} GT={r['gt']:>7,} EXP01={r['EXP01_recall_pct']:>7.2f}% EXP04={r['EXP04_recall_pct']:>7.2f}% Delta={r['delta_pp']:+7.2f} pp TPgain={r['tp_gain']:+d}")
    lines+=["","INTERACCIONES","-"*72]
    for r in interactions:
        lines.append(f"{r['interaction']:<40} GT={r['gt']:>6,} EXP01={r['EXP01_recall_pct']:>7.2f}% EXP04={r['EXP04_recall_pct']:>7.2f}% Delta={r['delta_pp']:+7.2f} pp TPgain={r['tp_gain']:+d}")
    lines+=["","TAMAÑO","-"*72]
    for r in sizes:
        lines.append(f"{r['size_bucket']:>8} GT={r['gt']:>7,} EXP01={r['EXP01_recall_pct']:>7.2f}% EXP04={r['EXP04_recall_pct']:>7.2f}% Delta={r['delta_pp']:+7.2f} pp TPgain={r['tp_gain']:+d}")
    lines+=["","LECTURA INICIAL","-"*72]
    lines.append("DENSE_SCENE muestra mejora fuerte." if dense["delta_pp"]>2 else "DENSE_SCENE no explica por sí solo toda la mejora.")
    lines.append("EXTREME_SMALL muestra mejora fuerte." if extreme["delta_pp"]>2 else "EXTREME_SMALL no concentra una mejora equivalente.")
    lines.append("EDGE_LOCATION mejora claramente." if edge["delta_pp"]>1 else "EDGE_LOCATION no parece ser el principal origen de la mejora.")
    lines+=["","SIGUIENTE PASO","-"*72,"Usar las interacciones con mayor TP gain/delta para definir la intervención final.","","IMPORTANTE: no se modificaron dataset, labels ni YAML."]
    SUMMARY_TXT.write_text("\n".join(lines),encoding="utf-8")
    print("="*72); print("# RESULTADO EXP04 VS EXP01"); print("="*72)
    print(f"GLOBAL: EXP01={overall['EXP01_recall_pct']:.2f}% EXP04={overall['EXP04_recall_pct']:.2f}% Delta={overall['delta_pp']:+.2f} pp")
    print(f"DENSE:  EXP01={dense['EXP01_recall_pct']:.2f}% EXP04={dense['EXP04_recall_pct']:.2f}% Delta={dense['delta_pp']:+.2f} pp")
    print(f"SMALL:  EXP01={extreme['EXP01_recall_pct']:.2f}% EXP04={extreme['EXP04_recall_pct']:.2f}% Delta={extreme['delta_pp']:+.2f} pp")
    print(f"EDGE:   EXP01={edge['EXP01_recall_pct']:.2f}% EXP04={edge['EXP04_recall_pct']:.2f}% Delta={edge['delta_pp']:+.2f} pp")
    print("REPORTS:")
    for p in [OBJECTS_CSV,FACTORS_CSV,INTERACTIONS_CSV,SIZE_CSV,SUMMARY_TXT]: print(f"[OK] {p}")

if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: print("[CANCELADO]"); sys.exit(130)
    except Exception as exc: print("[ERROR]"); print(str(exc)); sys.exit(1)
