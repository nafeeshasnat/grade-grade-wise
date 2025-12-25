#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed training script:
- Trains DT, RF, (optional) SVR, LightGBM, and a PyTorch MLP.
- Trains a risk classifier with SMOTE.
- **Always saves artifacts** even if plotting fails.
- Writes metadata.json and prints __RESULT__ JSON used by the Node backend.
"""

import argparse, os, sys, json, datetime, math
from pathlib import Path

import numpy as np
import pandas as pd

# Threading hints
os.environ.setdefault("OMP_NUM_THREADS", "4")

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns  # optional
except Exception:
    plt = None
    sns = None

# ------------------------- Bounds -------------------------
BOUNDS = {
    "RF_TREES": (50, 1000),
    "LGBM_N_ESTIMATORS": (200, 4000),
    "MLP_HIDDEN": (16, 256),
    "MLP_EPOCHS": (50, 600),
    "MLP_PATIENCE": (10, 100),
    "TEST_SIZE": (0.10, 0.30),
    "THREADS": (2, 8),
}

# ------------------------- GPA helpers -------------------------
def validate_grade_points(grade_points: dict):
    if not isinstance(grade_points, dict) or len(grade_points) < 2 or len(grade_points) > 30:
        raise ValueError("GRADE_POINTS must be an object with 2..30 entries")
    vals = []
    for k, v in grade_points.items():
        if not isinstance(k, str): raise ValueError("GRADE_POINTS keys must be strings")
        if not isinstance(v, (int, float)): raise ValueError("GRADE_POINTS values must be numbers")
        if v < 0 or v > 10: raise ValueError("GRADE_POINTS values must be within [0,10]")
        vals.append(float(v))
    return float(max(vals))

def semester_gpa(sem, GP):
    pts = []
    for k, g in sem.items():
        if k == "attendancePercentage":
            continue
        if g in GP:
            pts.append(GP[g])
    return float(np.mean(pts)) if pts else None

def cumulative_cgpa(semesters, GP):
    tot = 0.0; cnt = 0
    for sem in semesters.values():
        for k, g in sem.items():
            if k == "attendancePercentage":
                continue
            if g in GP:
                tot += GP[g]; cnt += 1
    return (tot/cnt) if cnt > 0 else None

def build_features_for_final(student, GP):
    semesters = student.get("semesters", {})
    if not semesters: return None, None
    sem_nums = sorted(map(int, semesters.keys()))
    if len(sem_nums) < 2: return None, None
    y = cumulative_cgpa(semesters, GP)
    if y is None: return None, None

    upto = sem_nums[-2]
    att, sem_gpas, loads = [], [], []
    for s in sem_nums:
        if s <= upto:
            sem = semesters[str(s)]
            if "attendancePercentage" in sem: att.append(sem["attendancePercentage"])
            g = semester_gpa(sem, GP)
            if g is not None: sem_gpas.append(g)
            loads.append(sum(1 for k in sem if k != "attendancePercentage"))
    avg_att = float(np.mean(att)) if att else 0.0
    gpa_trend = (sem_gpas[-1] - sem_gpas[-2]) if len(sem_gpas) >= 2 else 0.0
    avg_load = float(np.mean(loads)) if loads else 0.0

    X = [
        float(student.get("ssc_gpa", 0.0)),
        float(student.get("hsc_gpa", 0.0)),
        1 if str(student.get("gender","")).lower()=="female" else 0,
        int(student.get("birth_year", 0)),
        avg_att, gpa_trend, avg_load
    ]
    return X, float(y)

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
        loads.append(sum(1 for k in sem if k != "attendancePercentage"))
    avg_att = float(np.mean(att)) if att else 0.0
    gpa_trend = (sem_gpas[-1] - sem_gpas[-2]) if len(sem_gpas) >= 2 else 0.0
    avg_load = float(np.mean(loads)) if loads else 0.0

    X = [
        float(student.get("ssc_gpa", 0.0)),
        float(student.get("hsc_gpa", 0.0)),
        1 if str(student.get("gender","")).lower()=="female" else 0,
        int(student.get("birth_year", 0)),
        avg_att, gpa_trend, avg_load
    ]
    return X

# ------------------------- Main -------------------------
def main():
    import warnings; warnings.filterwarnings("ignore")

    ap = argparse.ArgumentParser()
    ap.add_argument("--org-id", required=True)
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--config-json", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    org_id = args.org_id
    out_dir = Path(args.out_dir); plots_dir = out_dir/"plots"
    out_dir.mkdir(parents=True, exist_ok=True); plots_dir.mkdir(parents=True, exist_ok=True)

    # Load config & clamp bounds
    cfg = json.load(open(args.config_json))
    def clamp(name, default):
        lo, hi = BOUNDS.get(name, (None,None))
        v = cfg.get(name, default)
        try: v = float(v)
        except Exception: v = default
        if lo is not None: v = max(lo, v)
        if hi is not None: v = min(hi, v)
        return v

    RANDOM_SEED = int(cfg.get("RANDOM_SEED", 42))
    THREADS     = int(clamp("THREADS", 4))
    TEST_SIZE   = float(clamp("TEST_SIZE", 0.2))
    RF_TREES    = int(clamp("RF_TREES", 400))
    LGBM_N_EST  = int(clamp("LGBM_N_ESTIMATORS", 2000))
    MLP_HIDDEN  = int(clamp("MLP_HIDDEN", 64))
    MLP_EPOCHS  = int(clamp("MLP_EPOCHS", 300))
    MLP_PATIENCE= int(clamp("MLP_PATIENCE", 40))
    SVR_ENABLE  = bool(cfg.get("SVR_ENABLE", True))
    RISK_HIGH_MAX = float(cfg.get("RISK_HIGH_MAX", 3.30))
    RISK_MED_MAX  = float(cfg.get("RISK_MED_MAX", 3.50))
    GRADE_POINTS = cfg.get("GRADE_POINTS", {
        "A+":4.0,"A":3.75,"A-":3.5,"B+":3.25,"B":3.0,"B-":2.75,"C+":2.5,"C":2.25,"D":2.0,"F":0.0
    })
    max_gpa = validate_grade_points(GRADE_POINTS)

    # Optional: set torch threads
    try:
        import torch; torch.set_num_threads(THREADS)
    except Exception:
        pass

    # Load data
    train_path = args.train_json
    assert os.path.exists(train_path), f"Training file not found: {train_path}"
    data = json.load(open(train_path))
    print(f"[INFO] org={org_id} students={len(data)} max_gpa={max_gpa}")

    # Build datasets
    X_final, y_final = [], []
    X_next, y_next = [], []   # y_next not used for supervised next-sem label here; kept for symmetry

    for student in data:
        Xf, yf = build_features_for_final(student, GRADE_POINTS)
        if Xf is not None and yf is not None:
            X_final.append(Xf); y_final.append(yf)
        Xn = build_features_for_next(student, GRADE_POINTS)
        if Xn is not None:
            X_next.append(Xn)

    X_final = np.array(X_final, float) if len(X_final) else np.empty((0,7))
    y_final = np.array(y_final, float) if len(y_final) else np.empty((0,))
    X_next  = np.array(X_next,  float) if len(X_next)  else np.empty((0,7))

    feat_names = ["ssc_gpa","hsc_gpa","gender_bin","birth_year","avg_attendance","gpa_trend","avg_course_load"]

    # Models
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

    # Split for evaluation
    from sklearn.model_selection import train_test_split as _tts
    Xf_tr, Xf_te, yf_tr, yf_te = _tts(X_final, y_final, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    # Train classical models
    dt  = DecisionTreeRegressor(random_state=RANDOM_SEED)
    rf  = RandomForestRegressor(n_estimators=RF_TREES, random_state=RANDOM_SEED, n_jobs=-1)
    svr = SVR(kernel="rbf", C=10.0, gamma="scale") if SVR_ENABLE else None
    lgbm = lgb.LGBMRegressor(
        n_estimators=LGBM_N_EST,
        learning_rate=0.03,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    dt.fit(Xf_tr, yf_tr)
    rf.fit(Xf_tr, yf_tr)
    if SVR_ENABLE: svr.fit(Xf_tr, yf_tr)

    # LightGBM with early stopping
    Xtr_lgb, Xval_lgb, ytr_lgb, yval_lgb = _tts(Xf_tr, yf_tr, test_size=0.2, random_state=RANDOM_SEED)
    evals_result = {}
    lgbm.fit(
        Xtr_lgb, ytr_lgb,
        eval_set=[(Xtr_lgb, ytr_lgb), (Xval_lgb, yval_lgb)],
        eval_names=["train", "valid"],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False),
                   lgb.record_evaluation(evals_result)]
    )

    # MLP
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

    def train_mlp(Xtr, ytr, Xval, yval, epochs=300, lr=1e-3, patience=40, hid=64):
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr); Xval_s = scaler.transform(Xval)
        xt = torch.tensor(Xtr_s, dtype=torch.float32)
        yt = torch.tensor(ytr.reshape(-1,1), dtype=torch.float32)
        xv = torch.tensor(Xval_s, dtype=torch.float32)
        yv = torch.tensor(yval.reshape(-1,1), dtype=torch.float32)
        model = MLP(in_dim=Xtr.shape[1], hid=hid)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        best = math.inf; best_state = None; patience_ctr = 0
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            pred = model(xt); loss = loss_fn(pred, yt); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vloss = loss_fn(model(xv), yv).item()
            if vloss < best - 1e-6:
                best = vloss; best_state = {k:v.clone() for k,v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, scaler

    Xtr_mlp, Xte_mlp, ytr_mlp, yte_mlp = _tts(Xf_tr, yf_tr, test_size=0.2, random_state=RANDOM_SEED)
    mlp_model, mlp_scaler = train_mlp(Xtr_mlp, ytr_mlp, Xte_mlp, yte_mlp, epochs=MLP_EPOCHS, patience=MLP_PATIENCE, hid=MLP_HIDDEN)

    # Evaluate
    def eval_model(m, Xtr, ytr, Xte, yte, name):
        yhat_tr = m.predict(Xtr); yhat_te = m.predict(Xte)
        if isinstance(yhat_tr, (list, tuple)): yhat_tr = np.array(yhat_tr)
        if isinstance(yhat_te, (list, tuple)): yhat_te = np.array(yhat_te)
        r2_tr = r2_score(ytr, yhat_tr); r2_te = r2_score(yte, yhat_te)
        # Older sklearn may not support squared=False; take sqrt manually
        rmse_tr = math.sqrt(mean_squared_error(ytr, yhat_tr))
        rmse_te = math.sqrt(mean_squared_error(yte, yhat_te))
        return {"name":name,"r2_tr":r2_tr,"r2_te":r2_te,"rmse_tr":rmse_tr,"rmse_te":rmse_te, "yhat_te":yhat_te}

    def eval_mlp(model, scaler, Xtr, ytr, Xte, yte):
        Xtr_s = mlp_scaler.transform(Xtr); Xte_s = mlp_scaler.transform(Xte)
        import torch
        with torch.no_grad():
            yhat_tr = model(torch.tensor(Xtr_s, dtype=torch.float32)).numpy().reshape(-1)
            yhat_te = model(torch.tensor(Xte_s, dtype=torch.float32)).numpy().reshape(-1)
        from sklearn.metrics import r2_score, mean_squared_error
        r2_tr = r2_score(ytr, yhat_tr); r2_te = r2_score(yte, yhat_te)
        rmse_tr = math.sqrt(mean_squared_error(ytr, yhat_tr))
        rmse_te = math.sqrt(mean_squared_error(yte, yhat_te))
        return {"name":"MLP","r2_tr":r2_tr,"r2_te":r2_te,"rmse_tr":rmse_tr,"rmse_te":rmse_te, "yhat_te":yhat_te}

    results = []
    results.append(eval_model(dt, Xf_tr, yf_tr, Xf_te, yf_te, "DecisionTree"))
    results.append(eval_model(rf, Xf_tr, yf_tr, Xf_te, yf_te, "RandomForest"))
    if SVR_ENABLE: results.append(eval_model(svr, Xf_tr, yf_tr, Xf_te, yf_te, "SVR"))
    # lgbm has .predict
    results.append(eval_model(lgbm, Xf_tr, yf_tr, Xf_te, yf_te, "LightGBM"))
    results.append(eval_mlp(mlp_model, mlp_scaler, Xf_tr, yf_tr, Xf_te, yf_te))

    # Rank
    rank_df = pd.DataFrame(results).sort_values(by=["rmse_te","rmse_tr"], ascending=[True,True]).reset_index(drop=True)
    best_name = rank_df.iloc[0]["name"]

    # --------- SAVE ARTIFACTS FIRST (so plotting errors won't break saving) ---------
    import joblib, torch
    joblib.dump(dt,   out_dir/"DecisionTree.joblib")
    joblib.dump(rf,   out_dir/"RandomForest.joblib")
    if SVR_ENABLE: joblib.dump(svr, out_dir/"SVR.joblib")
    joblib.dump(lgbm, out_dir/"LightGBM.joblib")
    torch.save(mlp_model.state_dict(), out_dir/"MLP.pt")
    joblib.dump(mlp_scaler, out_dir/"MLP_Scaler.joblib")

    # Risk classifier
    # Label by thresholds on true final CGPA
    def label_risk(cgpa):
        if cgpa <= RISK_HIGH_MAX: return "High"
        elif cgpa <= RISK_MED_MAX: return "Medium"
        return "Low"
    risk_y = np.array([label_risk(v) for v in y_final])
    unique_risk = np.unique(risk_y)
    ypred_risk = None
    yc_te = None
    from sklearn.ensemble import RandomForestClassifier
    if len(unique_risk) >= 2:
        from sklearn.model_selection import train_test_split as _tts2
        Xc_tr, Xc_te, yc_tr, yc_te = _tts2(X_final, risk_y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=risk_y)
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=RANDOM_SEED)
        Xc_tr_res, yc_tr_res = sm.fit_resample(Xc_tr, yc_tr)
        risk_clf = RandomForestClassifier(n_estimators=250, random_state=RANDOM_SEED, n_jobs=-1)
        risk_clf.fit(Xc_tr_res, yc_tr_res)
        ypred_risk = risk_clf.predict(Xc_te)
        risk_accuracy = accuracy_score(yc_te, ypred_risk)
    else:
        from sklearn.dummy import DummyClassifier
        # Fall back to constant classifier if we have only one risk label
        constant_label = unique_risk[0] if len(unique_risk) else "Low"
        risk_clf = DummyClassifier(strategy="constant", constant=constant_label)
        risk_clf.fit(X_final, risk_y)
        risk_accuracy = 1.0

    joblib.dump(risk_clf, out_dir/"RiskClassifier.joblib")

    # Metadata
    meta = {
        "created_at": datetime.datetime.utcnow().isoformat()+"Z",
        "feature_order": feat_names,
        "grade_points": GRADE_POINTS,
        "max_gpa": max_gpa,
        "best_model": str(best_name),
        "models": {r["name"]: {"r2_tr": r["r2_tr"], "r2_te": r["r2_te"], "rmse_tr": r["rmse_tr"], "rmse_te": r["rmse_te"]} for r in results},
        "risk_accuracy": float(risk_accuracy)
    }
    with open(out_dir/"metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Determine storage root so we can emit static URLs
    storage_root = None
    for idx, part in enumerate(out_dir.parts):
        if part == "storage":
            storage_root = Path(*out_dir.parts[:idx + 1])
            break

    def to_static_path(path_obj: Path) -> str:
        if storage_root is not None:
            try:
                rel = path_obj.relative_to(storage_root)
                return "/static/" + rel.as_posix()
            except Exception:
                pass
        return str(path_obj)

    # --------- Plotting (best-effort, won't crash training) ---------
    saved_plots = {}
    if plt is not None:
        try:
            # Residuals for best model
            yhat_map = {r["name"]: r["yhat_te"] for r in results}
            resid = yf_te - yhat_map[best_name]
            fig, ax = plt.subplots(1,2, figsize=(10,4))
            ax[0].scatter(yhat_map[best_name], resid, alpha=0.5); ax[0].axhline(0, color='k', lw=1)
            ax[0].set_title(f"{best_name}: Residuals vs Prediction"); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Residual")
            if sns is not None:
                sns.histplot(resid, kde=True, ax=ax[1]); ax[1].set_title(f"{best_name}: Residual Distribution")
            else:
                ax[1].hist(resid, bins=30); ax[1].set_title(f"{best_name}: Residual Distribution")
            plt.tight_layout(); p = plots_dir/"best_model_residuals.png"; plt.savefig(p); plt.close(fig)
            saved_plots["best_model_residuals"] = to_static_path(p)

            # Feature importances for RF / LGBM
            try:
                fig = plt.figure(figsize=(6,4))
                (sns.barplot(x=pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False).values,
                             y=pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False).index)
                 if sns is not None else plt.barh(feat_names, rf.feature_importances_))
                plt.title("RF Feature Importance"); plt.tight_layout()
                p = plots_dir/"rf_importance.png"
                plt.savefig(p); plt.close()
                saved_plots["rf_feature_importance"] = to_static_path(p)
            except Exception as e:
                print(f"[WARN] RF plot failed: {e}")

            try:
                fig = plt.figure(figsize=(6,4))
                (sns.barplot(x=pd.Series(lgbm.feature_importances_, index=feat_names).sort_values(ascending=False).values,
                             y=pd.Series(lgbm.feature_importances_, index=feat_names).sort_values(ascending=False).index)
                 if sns is not None else plt.barh(feat_names, lgbm.feature_importances_))
                plt.title("LGBM Feature Importance"); plt.tight_layout()
                p = plots_dir/"lgbm_importance.png"
                plt.savefig(p); plt.close()
                saved_plots["lgbm_feature_importance"] = to_static_path(p)
            except Exception as e:
                print(f"[WARN] LGBM plot failed: {e}")

            # Risk confusion matrix
            try:
                from sklearn.metrics import ConfusionMatrixDisplay
                if yc_te is not None and ypred_risk is not None:
                    ConfusionMatrixDisplay.from_predictions(yc_te, ypred_risk)
                    plt.title("Risk Classifier Confusion Matrix"); plt.tight_layout()
                    p = plots_dir/"risk_confusion_matrix.png"
                    plt.savefig(p); plt.close()
                    saved_plots["risk_confusion_matrix"] = to_static_path(p)
            except Exception as e:
                print(f"[WARN] Risk CM plot failed: {e}")

            # LightGBM learning curve
            try:
                train_curve = evals_result.get("train", {}).get("rmse", [])
                valid_curve = evals_result.get("valid", {}).get("rmse", [])
                if train_curve or valid_curve:
                    fig = plt.figure(figsize=(6,4))
                    if train_curve:
                        plt.plot(train_curve, label="Train RMSE")
                    if valid_curve:
                        plt.plot(valid_curve, label="Validation RMSE")
                    plt.xlabel("Iteration"); plt.ylabel("RMSE"); plt.title("LightGBM Learning Curve")
                    plt.legend(); plt.tight_layout()
                    p = plots_dir/"lightgbm_learning_curve.png"
                    plt.savefig(p); plt.close()
                    saved_plots["lightgbm_learning_curve"] = to_static_path(p)
            except Exception as e:
                print(f"[WARN] Learning curve plot failed: {e}")

        except Exception as e:
            print(f"[WARN] plotting failed: {e}")

    # --------- Final result ---------
    best_metrics = rank_df.iloc[0]
    metrics_summary = {
        "rmse": float(best_metrics["rmse_te"]),
        "r2": float(best_metrics["r2_te"]),
        "accuracy": float(risk_accuracy)
    }

    result = {
        "status": "ok",
        "bestModel": meta["best_model"],
        "artifactsDir": str(out_dir),
        "plots": saved_plots,
        "gradePoints": GRADE_POINTS,
        "metrics": metrics_summary,
        "models": meta["models"]
    }
    print("__RESULT__" + json.dumps(result))
    return 0

if __name__ == "__main__":
    try:
        import time; sys.exit(main())
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        print("__RESULT__" + json.dumps({"status":"error","error":str(e)}))
        sys.exit(1)
