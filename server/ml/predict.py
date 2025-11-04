#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed prediction script:
- Loads artifacts saved by train.py.
- Computes current GPA/CGPA, final CGPA prediction (ensemble) and next-sem GPA.
- Prints __RESULT__ JSON and writes a detailed JSON to --out-file.
"""

import argparse, os, sys, json, numpy as np, datetime, joblib
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns  # optional
    from matplotlib.lines import Line2D
except Exception:
    plt = None
    sns = None
    Line2D = None

def semester_gpa(sem, GP):
    pts = []
    for k, g in sem.items():
        if k != "attendancePercentage" and g in GP:
            pts.append(GP[g])
    return float(np.mean(pts)) if pts else None

def build_features_for_final(student, GP):
    semesters = student.get("semesters", {})
    if not semesters: return None, None
    sem_nums = sorted(map(int, semesters.keys()))
    if len(sem_nums) < 2: return None, None
    upto = sem_nums[-2]
    att, sem_gpas, loads = [], [], []
    for s in sem_nums:
        if s <= upto:
            sem = semesters[str(s)]
            if "attendancePercentage" in sem: att.append(sem["attendancePercentage"])
            g = semester_gpa(sem, GP)
            if g is not None: sem_gpas.append(g)
            loads.append(sum(1 for k in sem if k!="attendancePercentage"))
    avg_att = float(np.mean(att)) if att else 0.0
    gpa_trend = (sem_gpas[-1]-sem_gpas[-2]) if len(sem_gpas)>=2 else 0.0
    avg_load = float(np.mean(loads)) if loads else 0.0
    X = [
        float(student.get("ssc_gpa", 0.0)),
        float(student.get("hsc_gpa", 0.0)),
        1 if str(student.get("gender","")).lower()=="female" else 0,
        int(student.get("birth_year", 0)),
        avg_att, gpa_trend, avg_load
    ]
    return X, None

def build_features_for_next(student, GP):
    semesters = student.get("semesters", {})
    if not semesters: return None
    sem_nums = sorted(map(int, semesters.keys()))
    att, sem_gpas, loads = [], [], []
    for s in sem_nums:
        sem = semesters[str(s)]
        if "attendancePercentage" in sem: att.append(sem["attendancePercentage"])
        g = semester_gpa(sem, GP)
        if g is not None: sem_gpas.append(g)
        loads.append(sum(1 for k in sem if k!="attendancePercentage"))
    avg_att = float(np.mean(att)) if att else 0.0
    gpa_trend = (sem_gpas[-1]-sem_gpas[-2]) if len(sem_gpas)>=2 else 0.0
    avg_load = float(np.mean(loads)) if loads else 0.0
    X = [
        float(student.get("ssc_gpa", 0.0)),
        float(student.get("hsc_gpa", 0.0)),
        1 if str(student.get("gender","")).lower()=="female" else 0,
        int(student.get("birth_year", 0)),
        avg_att, gpa_trend, avg_load
    ]
    return X

def compute_current(student, GP):
    sems = student.get("semesters", {})
    if not sems: return None, None, None
    sem_nums = sorted(map(int, sems.keys()))
    last = sem_nums[-1]
    sem = sems[str(last)]
    last_sem_gpa = semester_gpa(sem, GP)
    # cgpa to date
    tot=0.0; cnt=0
    for s in sems.values():
        for k,g in s.items():
            if k!="attendancePercentage" and g in GP:
                tot += GP[g]; cnt += 1
    cgpa = (tot/cnt) if cnt>0 else None
    return last_sem_gpa, cgpa, last

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org-id", required=True)
    ap.add_argument("--student-json", required=True)
    ap.add_argument("--artifacts-dir", required=True)
    ap.add_argument("--out-file", required=True)
    args = ap.parse_args()

    org_id = args.org_id
    student = json.load(open(args.student_json,"r"))
    art_dir = Path(args.artifacts_dir)
    out_file = Path(args.out_file)

    meta = json.load(open(art_dir/"metadata.json", "r"))
    GP = meta.get("grade_points", {
        "A+":4.0,"A":3.75,"A-":3.5,"B+":3.25,"B":3.0,"B-":2.75,"C":2.5,"C-":2.25,"D":2.0,"F":0.0
    })
    max_gpa = float(meta.get("max_gpa", max(GP.values())))
    best_model = meta.get("best_model")

    # Static asset helper
    storage_root = None
    for idx, part in enumerate(out_file.parts):
        if part == "storage":
            storage_root = Path(*out_file.parts[:idx + 1])
            break

    def to_static_path(path_obj: Path) -> str:
        if storage_root is not None:
            try:
                rel = path_obj.relative_to(storage_root)
                return "/static/" + rel.as_posix()
            except Exception:
                pass
        return str(path_obj)

    print(f"[INFO] org={org_id} student={student.get('student_id')} max_gpa={max_gpa}")

    # Load models
    models = {}
    for name in ["DecisionTree","RandomForest","LightGBM","SVR"]:
        p = art_dir/f"{name}.joblib"
        if p.exists():
            models[name] = joblib.load(p)

    # MLP + scaler
    import torch, torch.nn as nn
    class MLP(nn.Module):
        def __init__(self, in_dim, hid=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hid), nn.ReLU(),
                nn.Linear(hid, hid), nn.ReLU(),
                nn.Linear(hid, 1)
            )
        def forward(self, x): return self.net(x)

    feat_order = meta["feature_order"]
    mlp_model = MLP(in_dim=len(feat_order), hid=64)
    mlp_model.load_state_dict(torch.load(art_dir/"MLP.pt", map_location="cpu"))
    mlp_model.eval()
    mlp_scaler = joblib.load(art_dir/"MLP_Scaler.joblib")

    risk_clf = joblib.load(art_dir/"RiskClassifier.joblib")

    # Features
    Xf_new, _ = build_features_for_final(student, GP)
    Xn_new = build_features_for_next(student, GP)

    # Current
    cur_sem_gpa, cur_cgpa, last_sem_idx = compute_current(student, GP)
    if cur_sem_gpa is not None:
        print(f"[INFO] current last-sem GPA (sem {last_sem_idx}) = {cur_sem_gpa:.2f}")
    if cur_cgpa is not None:
        print(f"[INFO] current CGPA (to sem {last_sem_idx}) = {cur_cgpa:.2f}")

    # Predict final CGPA (5 models)
    preds_final = {}
    if Xf_new is not None:
        Xf = np.array(Xf_new, float).reshape(1,-1)
        for name, model in models.items():
            try:
                preds_final[name] = float(model.predict(Xf)[0])
            except Exception as e:
                print(f"[WARN] {name} prediction failed: {e}")
        Xs = mlp_scaler.transform(Xf)
        with torch.no_grad():
            preds_final["MLP"] = float(mlp_model(torch.tensor(Xs, dtype=torch.float32)).numpy().reshape(-1)[0])
    else:
        print("[WARN] Not enough semesters for final CGPA prediction (needs ≥ 2).")

    # Predict next semester GPA (RF+LGBM+SVR)
    preds_next = {}
    if Xn_new is not None:
        Xn = np.array(Xn_new, float).reshape(1,-1)
        if "LightGBM" in models:   preds_next["LightGBM"]   = float(models["LightGBM"].predict(Xn)[0])
        if "RandomForest" in models: preds_next["RandomForest"]= float(models["RandomForest"].predict(Xn)[0])
        if "SVR" in models:        preds_next["SVR"]        = float(models["SVR"].predict(Xn)[0])

    # Ensemble mean
    ens_final = float(np.mean([v for v in preds_final.values()])) if preds_final else None
    ens_next  = float(np.mean([v for v in preds_next.values()])) if preds_next else None

    # Risk
    try:
        if Xn_new is not None:
            risk_label = str(risk_clf.predict(np.array(Xn_new, float).reshape(1,-1))[0])
        else:
            risk_label = "Unknown"
    except Exception as e:
        print(f"[WARN] risk prediction failed: {e}")
        risk_label = "Unknown"

    plots = {}
    if plt is not None:
        try:
            if preds_final:
                fig, ax = plt.subplots(figsize=(6, 4))
                models = list(preds_final.keys())
                values = [preds_final[m] for m in models]
                colors = ['#0ea5e9'] * len(models)
                if best_model:
                    for i, model in enumerate(models):
                        if model.lower() == str(best_model).lower():
                            colors[i] = '#22c55e'
                ax.bar(models, values, color=colors)
                ax.set_ylabel("Predicted Final CGPA")
                ax.set_title("Final CGPA Predictions by Model")
                ax.tick_params(axis='x', rotation=30)
                if best_model and Line2D is not None:
                    ax.legend(handles=[Line2D([0], [0], color='#22c55e', lw=8, label='Best Model')])
                plt.tight_layout()
                final_plot = out_file.with_name(out_file.stem + "_final_cgpa.png")
                plt.savefig(final_plot)
                plt.close(fig)
                plots["final_cgpa_comparison"] = to_static_path(final_plot)

            if preds_next:
                fig, ax = plt.subplots(figsize=(6, 4))
                models = list(preds_next.keys())
                values = [preds_next[m] for m in models]
                ax.bar(models, values, color='#6366f1')
                ax.set_ylabel("Predicted Next Sem GPA")
                ax.set_title("Next Semester GPA Predictions")
                ax.tick_params(axis='x', rotation=30)
                plt.tight_layout()
                next_plot = out_file.with_name(out_file.stem + "_next_sem_gpa.png")
                plt.savefig(next_plot)
                plt.close(fig)
                plots["next_sem_gpa_comparison"] = to_static_path(next_plot)
        except Exception as e:
            print(f"[WARN] prediction plotting failed: {e}")

    result_payload = {
        "student_id": student.get("student_id"),
        "current": {"last_sem_index": last_sem_idx, "last_sem_gpa": cur_sem_gpa, "current_cgpa": cur_cgpa},
        "predictions": {
            "final_cgpa": preds_final,
            "next_sem_gpa": preds_next,
            "ensemble": {"final_cgpa_mean": ens_final, "next_sem_gpa_mean": ens_next}
        },
        "risk": risk_label,
        "created_at": datetime.datetime.utcnow().isoformat()+"Z",
        "grade_points_used": GP,
        "max_gpa": max_gpa,
        "best_model": best_model,
        "plots": plots
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result_payload, f, indent=2)
    print(f"[INFO] wrote {out_file}")

    hist_path = Path(art_dir) / "analysis_history.json"
    try: hist = json.load(open(hist_path, "r"))
    except Exception: hist = []
    hist.append(result_payload)
    with open(hist_path, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"[INFO] appended to {hist_path}")

    print("__RESULT__" + json.dumps({
        "status": "ok",
        "predictions": result_payload["predictions"],
        "risk": risk_label,
        "current": result_payload["current"],
        "outFile": str(out_file),
        "max_gpa": max_gpa,
        "plots": plots,
        "bestModel": best_model
    }))
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        print("__RESULT__" + json.dumps({"status":"error","error":str(e)}))
        sys.exit(1)
