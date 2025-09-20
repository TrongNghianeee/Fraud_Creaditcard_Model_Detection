# !pip install xgboost==2.0.3 pandas numpy==1.26.4 scipy==1.13.1 joblib matplotlib scikit-learn==1.2.2 --no-deps

import os
import sys
import contextlib
import json
import time
import math
import argparse
import warnings
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve  # Sửa: Import từ sklearn.calibration
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, cv as xgb_cv, DMatrix
import joblib
from scipy.stats import ks_2samp

# Optional resampling (imbalanced-learn)
try:
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import EditedNearestNeighbours
    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False
    print("[WARN] imbalanced-learn not installed. Install with pip install imbalanced-learn to enable resampling.")

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Sử dụng GPU nếu có
warnings.filterwarnings("ignore", category=UserWarning)

# Utilities & Metrics
@contextlib.contextmanager
def _suppress_stderr():
    saved_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, "w")
        yield
    finally:
        try:
            sys.stderr.close()
        except Exception:
            pass
        sys.stderr = saved_stderr

def ensure_outputs_dir(path: str = "outputs"):
    os.makedirs(path, exist_ok=True)
    return path

def recall_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: float = 0.01) -> float:
    n = len(y_true)
    k_n = max(1, int(math.ceil(k * n)))
    idx = np.argsort(-y_proba)[:k_n]
    return float(y_true[idx].sum()) / max(1, int(y_true.sum()))

def precision_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: float = 0.01) -> float:
    n = len(y_true)
    k_n = max(1, int(math.ceil(k * n)))
    idx = np.argsort(-y_proba)[:k_n]
    return float(y_true[idx].sum()) / max(1, k_n)

def recall_at_precision(y_true: np.ndarray, y_proba: np.ndarray, target_precision: float = 0.9) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    idx = np.where(prec >= target_precision)[0]
    if len(idx) == 0:
        return 0.0
    return np.max(rec[idx])

def compute_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    pos_scores = y_proba[y_true == 1]
    neg_scores = y_proba[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0
    return ks_2samp(pos_scores, neg_scores).statistic

def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
    return np.sum(np.abs(prob_true - prob_pred) * np.histogram(y_proba, bins=n_bins, range=(0,1))[0] / len(y_true))

def compute_cost_metric(y_true: np.ndarray, y_pred: np.ndarray, cost_fp: float = 1.0, cost_fn: float = 100.0) -> float:
    cm = confusion_matrix(y_true, y_pred)
    fp = cm[0, 1]
    fn = cm[1, 0]
    return (fp * cost_fp) + (fn * cost_fn)

def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5, k_list: List[float] = [0.001, 0.005, 0.01], p_for_recall: float = 0.9, cost_fp: float = 1.0, cost_fn: float = 100.0) -> Dict:
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),  # Lưu dưới dạng list để dễ JSON
        "ks_statistic": compute_ks_statistic(y_true, y_proba),
        "ece": compute_ece(y_true, y_proba),
        "cost_metric": compute_cost_metric(y_true, y_pred, cost_fp, cost_fn),
    }
    for k in k_list:
        metrics[f"recall@{k}"] = recall_at_k(y_true, y_proba, k)
        metrics[f"precision@{k}"] = precision_at_k(y_true, y_proba, k)
    metrics["recall_at_p"] = recall_at_precision(y_true, y_proba, p_for_recall)
    return metrics

# Plotting helpers
def plot_xgb_evals(evals_result: Dict, out_path: str):
    plt.figure(figsize=(10, 4))
    for data_name, metrics in evals_result.items():
        if "aucpr" in metrics:
            plt.plot(metrics["aucpr"], label=f"{data_name} aucpr")
    plt.title("XGBoost eval: AUC-PR over boosting rounds")
    plt.xlabel("Boosting round")
    plt.ylabel("AUC-PR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "evals_aucpr.png"), dpi=150)
    plt.close()

def plot_roc_pr(y_true_val, y_proba_val, y_true_test, y_proba_test, out_path: str):
    fpr_v, tpr_v, _ = roc_curve(y_true_val, y_proba_val)
    fpr_t, tpr_t, _ = roc_curve(y_true_test, y_proba_test)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_v, tpr_v, label=f"Val ROC AUC={roc_auc_score(y_true_val, y_proba_val):.4f}")
    plt.plot(fpr_t, tpr_t, label=f"Test ROC AUC={roc_auc_score(y_true_test, y_proba_test):.4f}")
    plt.plot([0,1],[0,1],"--",label="random")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "roc_curves.png"), dpi=150)
    plt.close()

    prec_v, rec_v, _ = precision_recall_curve(y_true_val, y_proba_val)
    prec_t, rec_t, _ = precision_recall_curve(y_true_test, y_proba_test)
    plt.figure(figsize=(8, 6))
    plt.plot(rec_v, prec_v, label=f"Val PR-AUC={average_precision_score(y_true_val, y_proba_val):.4f}")
    plt.plot(rec_t, prec_t, label=f"Test PR-AUC={average_precision_score(y_true_test, y_proba_test):.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curves"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "pr_curves.png"), dpi=150)
    plt.close()

def plot_calibration(y_true_val, y_proba_val, y_true_test, y_proba_test, out_path: str, n_bins: int = 10):
    prob_true_v, prob_pred_v = calibration_curve(y_true_val, y_proba_val, n_bins=n_bins, strategy='uniform')
    prob_true_t, prob_pred_t = calibration_curve(y_true_test, y_proba_test, n_bins=n_bins, strategy='uniform')
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred_v, prob_true_v, marker='o', label="Validation")
    plt.plot(prob_pred_t, prob_true_t, marker='o', label="Test")
    plt.plot([0,1], [0,1], linestyle="--", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed probability")
    plt.title("Calibration Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "calibration_curves.png"), dpi=150)
    plt.close()

def plot_feature_importance(clf: XGBClassifier, feature_names: List[str], out_path: str, topk: int = 30):
    try:
        imp = clf.get_booster().get_score(importance_type="gain")
        items = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:topk]
        names = [k for k, v in items]
        vals = [v for k, v in items]
        plt.figure(figsize=(8, max(4, len(names)*0.25)))
        y_pos = np.arange(len(names))
        plt.barh(y_pos, vals)
        plt.yticks(y_pos, names)
        plt.gca().invert_yaxis()
        plt.title("Feature importance (gain)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_path, "feature_importance.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[WARN] plot_feature_importance failed: {e}")

# Data Preprocessing
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Time-based features
    df['hour_of_day'] = (df['Time'] % 86400) / 3600
    df['day_of_week'] = ((df['Time'] / 86400) % 7).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['time_since_midnight'] = df['Time'] % 86400
    
    # Amount-based features
    if 'Amount' in df.columns:
        df['log_amount'] = np.log1p(df['Amount'])
        df['amount_bin'] = pd.cut(df['Amount'], 
                                  bins=[-0.1, 0, 10, 50, 200, 1000, np.inf], 
                                  labels=['zero', 'micro', 'small', 'medium', 'large', 'huge'])
        df['amount_bin'] = df['amount_bin'].cat.codes
    
    # V-features statistical summaries
    v_cols = [col for col in df.columns if col.startswith('V') and col != 'V']
    if len(v_cols) > 0:
        df['v_sum'] = df[v_cols].sum(axis=1)
        df['v_mean'] = df[v_cols].mean(axis=1) 
        df['v_std'] = df[v_cols].std(axis=1).fillna(0)
        df['v_min'] = df[v_cols].min(axis=1)
        df['v_max'] = df[v_cols].max(axis=1)
        df['v_range'] = df['v_max'] - df['v_min']
        v_array = df[v_cols].values
        df['v_n_zeros'] = (v_array == 0).sum(axis=1)
        df['v_n_negative'] = (v_array < 0).sum(axis=1)
        df['v_n_extreme'] = (np.abs(v_array) > 2).sum(axis=1)
    
    # Interaction features
    if 'Amount' in df.columns:
        df['time_amount_interaction'] = df['hour_of_day'] * np.log1p(df['Amount'])
        df['weekend_amount'] = df['is_weekend'] * np.log1p(df['Amount'])
    
    # Standardization
    numerical_features = [col for col in df.columns if col not in ['Class'] and df[col].dtype in ['float64', 'int64']]
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df

# Data Loading & Splitting
DEFAULT_FEATURES = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]
TARGET = "Class"

def load_dataset(csv_path: str, features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = features + [TARGET]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    df = preprocess_data(df)
    derived_features = ["hour_of_day"]  # Ví dụ, thêm các features derived
    all_features = features + derived_features
    X = df[all_features].copy()
    y = df[TARGET].copy()
    return X, y

def chronological_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    time_col: str = "Time",
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    embargo: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6
    order = np.argsort(X[time_col].values)
    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = y.iloc[order].reset_index(drop=True)
    n = len(X_sorted)
    n_train = int(n * train_size)
    n_val = int(n * (train_size + val_size))
    e = embargo
    train_end = max(0, n_train - e)
    val_start = min(n, n_train + e)
    val_end = max(val_start, n_val - e)
    test_start = min(n, n_val + e)
    X_train, y_train = X_sorted.iloc[:train_end], y_sorted.iloc[:train_end]
    X_val, y_val = X_sorted.iloc[val_start:val_end], y_sorted.iloc[val_start:val_end]
    X_test, y_test = X_sorted.iloc[test_start:], y_sorted.iloc[test_start:]
    return X_train, X_val, X_test, y_train, y_val, y_test

def compute_scale_pos_weight(y: pd.Series) -> float:
    pos = int(y.sum())
    neg = len(y) - pos
    return max(1.0, neg / max(1, pos))

# Resampling helper (borrowed idea from model_v2)
def _apply_resampling(
    X: pd.DataFrame,
    y: pd.Series,
    method: int = 0,
    ratio: float = 0.1,
    k_neighbors: int = 5,
    target_fraud_share: float = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    method:
      0 = none
      1 = BorderlineSMOTE oversampling
      2 = BorderlineSMOTE + EditedNearestNeighbours (cleaning)
    ratio: target minority/majority sampling_strategy for oversampling
    """
    if method == 0 or not HAS_IMBLEARN:
        if method != 0 and not HAS_IMBLEARN:
            print("[WARN] imbalanced-learn missing; skipping resampling.")
        return X, y

    # Constrain ratio and neighbors to conservative ranges to avoid overfitting
    if target_fraud_share is not None:
        # Convert target share p to sampling_strategy r = p/(1-p)
        ratio = float(target_fraud_share / max(1e-9, (1.0 - target_fraud_share)))
    ratio = float(max(0.05, min(0.12, ratio)))
    k_neighbors = int(max(5, min(15, k_neighbors)))

    try:
        n0 = len(X)
        pos0 = int(y.sum())
        rate0 = (pos0 / n0) if n0 > 0 else 0.0
        print(f"[INFO] Before resampling: n={n0}, fraud={pos0} ({rate0*100:.4f}%)")
        if target_fraud_share is not None:
            print(f"[INFO] Requested target fraud share: {target_fraud_share*100:.2f}% -> sampling_strategy={ratio:.4f}")
        sm = BorderlineSMOTE(sampling_strategy=ratio, k_neighbors=k_neighbors, random_state=42)
        with _suppress_stderr():
            X_res, y_res = sm.fit_resample(X, y)
            if method == 2:
                enn = EditedNearestNeighbours(n_neighbors=3, kind_sel="all", n_jobs=1)
                X_res, y_res = enn.fit_resample(X_res, y_res)
        n1 = len(X_res)
        pos1 = int(y_res.sum())
        rate1 = (pos1 / n1) if n1 > 0 else 0.0
        print(f"[INFO] Resampling applied: method={method}, ratio={ratio}, k={k_neighbors} | {n0} -> {n1}")
        print(f"[INFO] After resampling: n={n1}, fraud={pos1} ({rate1*100:.4f}%)")
        if target_fraud_share is not None:
            status = "OK" if abs(rate1 - target_fraud_share) <= 0.01 else "OFF-TARGET"
            print(f"[INFO] Target check: achieved ~{rate1*100:.2f}% vs target {target_fraud_share*100:.2f}% -> {status}")
        return X_res, y_res
    except Exception as e:
        print(f"[WARN] Resampling failed ({e}); continuing without resampling.")
        return X, y

# XGBoost Training (Pure)
def train_pure_xgb(
    X_tr,
    y_tr,
    X_va,
    y_va,
    X_te,
    y_te,
    seed=42,
    out_dir: str = "outputs",
    resample_method: int = 0,
    resample_ratio: float = 0.1,
    resample_k: int = 5,
    resample_target_fraud: float = None,
):
    ensure_outputs_dir(out_dir)
    start = time.time()

    print(
        f"[INFO] Resampling config -> method={resample_method}, "
        f"ratio={resample_ratio}, k={resample_k}, target_fraud={resample_target_fraud}"
    )
    base_rate = float(y_tr.mean()) if len(y_tr) > 0 else 0.0
    print(f"[INFO] Train baseline fraud rate: {base_rate*100:.4f}% ({int(y_tr.sum())}/{len(y_tr)})")

    # Optional resampling on training set only (to avoid leakage)
    X_tr_use, y_tr_use = _apply_resampling(
        X_tr, y_tr,
        resample_method,
        resample_ratio,
        resample_k,
        target_fraud_share=resample_target_fraud,
    )

    # Tính scale_pos_weight theo dữ liệu train (sau resampling nếu có)
    spw = compute_scale_pos_weight(y_tr_use)
    
    # Tham số XGBoost cơ bản (pure, không tối ưu hóa)
    base_params = {
        "objective": "binary:logistic",
        "learning_rate": 0.02,
        "max_depth": 4,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_lambda": 5.0,
        "reg_alpha": 2.0,
        "min_child_weight": 5.0,
        "gamma": 0.2,
        "scale_pos_weight": spw,
        "tree_method": "gpu_hist",
        "n_gpus": 2,
        "eval_metric": ["aucpr", "logloss"],
    }
    
    # Cross-validation để tìm n_estimators
    try:
        if resample_method == 0:
            dtrain = DMatrix(X_tr_use, label=y_tr_use)
            tscv = TimeSeriesSplit(n_splits=5, test_size=len(X_tr)//5)
            res = xgb_cv(
                params=base_params,
                dtrain=dtrain,
                num_boost_round=800,
                folds=tscv.split(X_tr, y_tr),
                metrics=("aucpr", "logloss"),
                early_stopping_rounds=50,
                seed=seed,
                as_pandas=True,
                verbose_eval=False,
            )
            best_nrounds = len(res)
            print(f"[INFO] XGB CV chose {best_nrounds} rounds.")
        else:
            print("[INFO] Skipping CV due to resampling; using default n_estimators 400")
            best_nrounds = 400
    except Exception as e:
        print(f"[WARN] xgb cv failed: {e}; using default n_estimators 400")
        best_nrounds = 400
    
    # Train model
    clf = XGBClassifier(
        n_estimators=best_nrounds,
        random_state=seed,
        **base_params
    )
    evals_result = {}
    model_checkpoint_path = os.path.join(out_dir, "pure_xgb_model.pkl")
    
    try:
        clf.fit(
            X_tr_use, y_tr_use,
            eval_set=[(X_va, y_va)],
            verbose=False,
            early_stopping_rounds=50
        )
        evals_result = clf.evals_result()
        joblib.dump(clf, model_checkpoint_path)
        print(f"[INFO] Trained and saved pure XGBoost model to {model_checkpoint_path}")
    except Exception as e:
        print(f"[WARN] GPU training failed: {e}. Falling back to CPU.")
        clf.tree_method = "hist"
        clf.fit(X_tr_use, y_tr_use, eval_set=[(X_va, y_va)], verbose=False, early_stopping_rounds=50)
        evals_result = clf.evals_result()
        joblib.dump(clf, model_checkpoint_path)
    
    # Predictions and metrics
    proba_val = clf.predict_proba(X_va)[:, 1]
    proba_test = clf.predict_proba(X_te)[:, 1]
    thr = 0.5  # Default threshold
    metrics = compute_metrics(y_te.values, proba_test, threshold=thr)
    metrics.update({
        "chosen_threshold": thr,
        "selected_features": list(X_tr.columns),
        "n_selected": len(X_tr.columns),
    })
    
    # Plots
    try:
        if evals_result:
            plot_xgb_evals(evals_result, out_dir)
        plot_roc_pr(y_va.values, proba_val, y_te.values, proba_test, out_dir)
        plot_calibration(y_va.values, proba_val, y_te.values, proba_test, out_dir)
        plot_feature_importance(clf, list(X_tr.columns), out_dir)
        print("[INFO] Plots generated successfully.")
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")
    
    return {
        "model": clf,
        "metrics": metrics,
        "val_pr_auc": float(average_precision_score(y_va, proba_val)),
        "evals_result": evals_result,
        "resampling": {
            "method": resample_method,
            "ratio": resample_ratio,
            "k_neighbors": resample_k,
            "target_fraud_share": resample_target_fraud,
            "fraud_rate_before": float(y_tr.mean()),
            "fraud_rate_after": float(y_tr_use.mean()),
            "train_size_before": int(len(X_tr)),
            "train_size_after": int(len(X_tr_use)),
        },
    }

# Main Runner
def run_experiments(
    data_path: str,
    features: List[str] = None,
    seed: int = 42,
    embargo: int = 0,
    out_dir: str = "outputs",
    resample_method: int = 0,
    resample_ratio: float = 0.1,
    resample_k: int = 5,
    resample_target_fraud: float = None,
):
    features = features or DEFAULT_FEATURES
    X, y = load_dataset(data_path, features)
    
    X_tr, X_va, X_te, y_tr, y_va, y_te = chronological_train_val_test_split(
        X, y, time_col="Time", embargo=embargo
    )
    
    results = {}
    start = time.time()
    out = train_pure_xgb(
        X_tr.copy(), y_tr.copy(),
        X_va.copy(), y_va.copy(),
        X_te.copy(), y_te.copy(),
        seed,
        out_dir=out_dir,
        resample_method=resample_method,
        resample_ratio=resample_ratio,
        resample_k=resample_k,
        resample_target_fraud=resample_target_fraud,
    )
    duration = round(time.time() - start, 2)
    res = {
        "metrics": out["metrics"],
        "val_pr_auc": out.get("val_pr_auc", None),
        "time_sec": duration,
        "resampling": out.get("resampling", {}),
    }
    results["pure_xgb"] = res
    print(f"Pure XGBoost done in {duration}s | Test PR-AUC: {res['metrics']['pr_auc']:.5f} | ROC-AUC: {res['metrics']['roc_auc']:.5f}")
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"pure_xgb_results_{ts}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_file}")

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Pure XGBoost for Fraud Detection")
    p.add_argument("--data", type=str, default="/kaggle/input/creditcardfraud/creditcard.csv", help="Path to dataset CSV")
    p.add_argument("--embargo", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs")
    p.add_argument("--resample-method", type=int, default=2, choices=[0,1,2], help="0: none, 1: BorderlineSMOTE, 2: BorderlineSMOTE+ENN")
    p.add_argument("--resample-ratio", type=float, default=0.1, help="Target minority/majority ratio for oversampling (0.05-0.12 clamped)")
    p.add_argument("--resample-k", type=int, default=7, help="k_neighbors for BorderlineSMOTE (5-15 clamped)")
    p.add_argument("--resample-target-fraud", type=float, default=0.10, help="Desired fraud share after resampling (e.g., 0.10 = 10%). Overrides --resample-ratio.")
    args, _ = p.parse_known_args(argv)
    return args

if __name__ == "__main__":
    args = parse_args()
    try:
        run_experiments(
            data_path=args.data,
            seed=args.seed,
            embargo=args.embargo,
            out_dir=args.out,
            resample_method=args.resample_method,
            resample_ratio=args.resample_ratio,
            resample_k=args.resample_k,
            resample_target_fraud=args.resample_target_fraud,
        )
        print("\nHOÀN THÀNH! Kiểm tra thư mục outputs để xem kết quả.")
    except Exception as e:
        print(f"[ERROR] {e}")