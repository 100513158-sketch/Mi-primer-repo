from __future__ import annotations
import csv, sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
BASE = next((p for p in [SCRIPT.parent, *SCRIPT.parents] if p.name.lower()=="baseline"), None)
if BASE is None: raise RuntimeError("No se encontró baseline.")

SRC = BASE/"evaluation"/"dataset_analysis"/"detection_failure_analysis"/"person"/"small_failure_patterns"/"experiments"/"exp04_vs_exp01_factor_analysis_v1"/"reports"/"exp04_vs_exp01_objects_v1.csv"
OUT = BASE/"evaluation"/"dataset_analysis"/"detection_failure_analysis"/"person"/"small_failure_patterns"/"experiments"/"exp07_target_population_analysis_v1"/"reports"
POP = OUT/"exp07_target_population_v1.csv"
TRANS = OUT/"exp07_target_population_transitions_v1.csv"
SUMMARY = OUT/"EXP07_TARGET_POPULATION_ANALYSIS_V1_SUMMARY.txt"

def write_csv(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w",encoding="utf-8-sig",newline="") as f:
        if not rows: return
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def load():
    if not SRC.exists(): raise FileNotFoundError(f"No se encontró:\n{SRC}")
    with SRC.open("r",encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    req={"EXTREME_SMALL","DENSE_SCENE","CLOSE_NEIGHBORS","EXP01_TP","EXP04_TP","transition"}
    miss=req-set(rows[0]) if rows else req
    if miss: raise ValueError("Faltan columnas: "+", ".join(sorted(miss)))
    return rows

def agg(rows,name,fn):
    s=[r for r in rows if fn(r)]
    gt=len(s); t1=sum(int(r["EXP01_TP"]) for r in s); t4=sum(int(r["EXP04_TP"]) for r in s)
    f2t=sum(int(r["EXP01_TP"])==0 and int(r["EXP04_TP"])==1 for r in s)
    t2f=sum(int(r["EXP01_TP"])==1 and int(r["EXP04_TP"])==0 for r in s)
    r1=100*t1/gt if gt else 0; r4=100*t4/gt if gt else 0
    return {"population":name,"gt":gt,"share_pct":100*gt/len(rows) if rows else 0,
            "EXP01_tp":t1,"EXP01_fn":gt-t1,"EXP01_recall_pct":r1,
            "EXP04_tp":t4,"EXP04_fn":gt-t4,"EXP04_recall_pct":r4,
            "delta_pp":r4-r1,"tp_gain":t4-t1,"fn_to_tp":f2t,"tp_to_fn":t2f,
            "net_recovery":f2t-t2f}

def main():
    rows=load(); OUT.mkdir(parents=True,exist_ok=True)
    specs=[
      ("SMALL_PERSON_ALL",lambda r:True),
      ("EXTREME_SMALL",lambda r:int(r["EXTREME_SMALL"])==1),
      ("EXTREME_SMALL + DENSE_SCENE",lambda r:int(r["EXTREME_SMALL"])==1 and int(r["DENSE_SCENE"])==1),
      ("EXTREME_SMALL + CLOSE_NEIGHBORS",lambda r:int(r["EXTREME_SMALL"])==1 and int(r["CLOSE_NEIGHBORS"])==1),
      ("EXTREME_SMALL + DENSE_SCENE + CLOSE_NEIGHBORS",lambda r:int(r["EXTREME_SMALL"])==1 and int(r["DENSE_SCENE"])==1 and int(r["CLOSE_NEIGHBORS"])==1),
      ("EXTREME_SMALL + (DENSE_SCENE OR CLOSE_NEIGHBORS)",lambda r:int(r["EXTREME_SMALL"])==1 and (int(r["DENSE_SCENE"])==1 or int(r["CLOSE_NEIGHBORS"])==1)),
      ("DENSE_SCENE",lambda r:int(r["DENSE_SCENE"])==1),
      ("CLOSE_NEIGHBORS",lambda r:int(r["CLOSE_NEIGHBORS"])==1),
      ("EDGE + EXTREME_SMALL + DENSE_SCENE",lambda r:int(r["EDGE_LOCATION"])==1 and int(r["EXTREME_SMALL"])==1 and int(r["DENSE_SCENE"])==1),
      ("EDGE + EXTREME_SMALL + CLOSE_NEIGHBORS",lambda r:int(r["EDGE_LOCATION"])==1 and int(r["EXTREME_SMALL"])==1 and int(r["CLOSE_NEIGHBORS"])==1),
    ]
    pop=[agg(rows,n,fn) for n,fn in specs]
    trans=[]
    for n,fn in specs[2:6]:
        s=[r for r in rows if fn(r)]
        d={"population":n,"gt":len(s)}
        for k in ("FN_TO_TP","TP_TO_FN","TP_TO_TP","FN_TO_FN"):
            d[k]=sum(1 for r in s if {"FN_TO_TP":"FN_TO_TP","TP_TO_FN":"TP_TO_FN","TP_TO_TP":"TP_TO_TP","FN_TO_FN":"FN_TO_FN"}[k]==r["transition"])
        d["net_gain"]=d["FN_TO_TP"]-d["TP_TO_FN"]; trans.append(d)
    write_csv(POP,pop); write_csv(TRANS,trans)
    core=next(r for r in pop if r["population"].startswith("EXTREME_SMALL + DENSE_SCENE + CLOSE_NEIGHBORS"))
    union=next(r for r in pop if r["population"].startswith("EXTREME_SMALL + (DENSE_SCENE OR"))
    dense=next(r for r in pop if r["population"]=="EXTREME_SMALL + DENSE_SCENE")
    neigh=next(r for r in pop if r["population"]=="EXTREME_SMALL + CLOSE_NEIGHBORS")
    txt=["="*72,"SAR YOLO26 - EXP07 TARGET POPULATION ANALYSIS V1","="*72,"",
         "OBJETIVO","Medir el volumen real de ejemplos para una intervención dirigida a EXTREME_SMALL en escenas densas y/o con vecinos.","",
         "CORE TRIPLE",f"GT={core['gt']:,}","Share={core['share_pct']:.2f}%",f"EXP01={core['EXP01_recall_pct']:.2f}%",f"EXP04={core['EXP04_recall_pct']:.2f}%",f"Delta={core['delta_pp']:+.2f} pp",f"TP gain={core['tp_gain']:+d}",f"FN->TP={core['fn_to_tp']:,}",f"TP->FN={core['tp_to_fn']:,}","",
         "UNIÓN EXTREME_SMALL + DENSE/NEIGHBOR",f"GT={union['gt']:,}",f"Share={union['share_pct']:.2f}%",f"EXP01={union['EXP01_recall_pct']:.2f}%",f"EXP04={union['EXP04_recall_pct']:.2f}%",f"Delta={union['delta_pp']:+.2f} pp",f"TP gain={union['tp_gain']:+d}",f"FN->TP={union['fn_to_tp']:,}",f"TP->FN={union['tp_to_fn']:,}","",
         "COMPARACIÓN",f"Extreme+Dense GT={dense['gt']:,}, Δ={dense['delta_pp']:+.2f} pp, TPgain={dense['tp_gain']:+d}",f"Extreme+Neighbors GT={neigh['gt']:,}, Δ={neigh['delta_pp']:+.2f} pp, TPgain={neigh['tp_gain']:+d}","",
         "DECISIÓN","No entrenar todavía con crops masivos. Usar la población medida aquí para diseñar EXP07 de forma controlada.","","IMPORTANTE: dataset, labels y YAML oficial no modificados."]
    SUMMARY.write_text("\n".join(txt),encoding="utf-8")
    print("="*72); print("# RESULTADO EXP07 TARGET POPULATION ANALYSIS V1"); print("="*72)
    for r in pop[2:6]:
        print(f"{r['population']}: GT={r['gt']:,} EXP01={r['EXP01_recall_pct']:.2f}% EXP04={r['EXP04_recall_pct']:.2f}% Δ={r['delta_pp']:+.2f} pp TPgain={r['tp_gain']:+d} FN->TP={r['fn_to_tp']:,} TP->FN={r['tp_to_fn']:,}")
    print(f"[OK] {POP}"); print(f"[OK] {TRANS}"); print(f"[OK] {SUMMARY}")

if __name__=="__main__":
    try: main()
    except Exception as e: print("[ERROR EXP07]",e); sys.exit(1)
