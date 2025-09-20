# !pip install scikit-learn==1.2.2 --force-reinstall
# !pip install imbalanced-learn==0.10.1 xgboost==2.0.3 pandas numpy==1.26.4 scipy==1.13.1 joblib matplotlib category-encoders cesium --no-deps
import os
import json
import time
import math
import argparse
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

try:
    from imblearn.over_sampling import BorderlineSMOTE
    from imblearn.under_sampling import EditedNearestNeighbours
    from imblearn.pipeline import make_pipeline
    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False
    print("[WARN] imbalanced-learn not installed. Install with !pip install imbalanced-learn to enable resampling.")

from xgboost import XGBClassifier, cv as xgb_cv, DMatrix
import joblib

# Ép dùng cả hai GPU T4
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

warnings.filterwarnings("ignore", category=UserWarning)

# Custom Isotonic Regression (tương thích với scikit-learn 1.2.2)
class CustomIsotonicRegression:
    def __init__(self, out_of_bounds="clip"):
        self.out_of_bounds = out_of_bounds
        self.X_ = None
        self.y_ = None
        self.X_thresholds_ = None
        self.y_thresholds_ = None

    def fit(self, X, y):
        X = np.asarray(X).ravel()
        y = np.asarray(y).ravel()
        order = np.argsort(X)
        self.X_ = X[order]
        self.y_ = y[order]
        self.X_thresholds_, self.y_thresholds_ = self._make_unique(self.X_, self.y_)
        self._inplace_contiguous_isotonic_regression(self.y_thresholds_)
        return self

    def transform(self, X):
        X = np.asarray(X).ravel()
        if self.out_of_bounds == "clip":
            X = np.clip(X, self.X_thresholds_[0], self.X_thresholds_[-1])
        result = np.interp(X, self.X_thresholds_, self.y_thresholds_)
        return result

    @staticmethod
    def _make_unique(X, y):
        unique_indices = np.unique(X, return_index=True)[1]
        return X[unique_indices], y[unique_indices]

    @staticmethod
    def _inplace_contiguous_isotonic_regression(y):
        n = len(y)
        for k in range(1, n):
            while k > 0 and y[k - 1] > y[k]:
                y[k - 1] = y[k]
                k -= 1
        return y

# Utilities & Metrics
def ensure_outputs_dir(path: str = "outputs"):
    os.makedirs(path, exist_ok=True)
    return path

def choose_threshold_by_cost(y_true: np.ndarray, y_proba: np.ndarray, c_fn: float = 10.0, c_fp: float = 1.0) -> Tuple[float, Dict]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    cand_thresholds = np.r_[0.0, thresholds, 1.0]
    best_thr, best_cost, best_stats = 0.5, float("inf"), {}
    y_true = y_true.astype(int)
    P = y_true.sum()
    N = len(y_true) - P
    for thr in cand_thresholds:
        y_pred = (y_proba >= thr).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = P - tp
        tn = N - fp
        cost = c_fn * fn + c_fp * fp
        if cost < best_cost:
            best_cost = cost
            best_thr = thr
            best_stats = dict(tp=tp, fp=fp, fn=fn, tn=tn)
    return best_thr, {"cost": best_cost, **best_stats}

def recall_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: float = 0.01) -> float:
    n = len(y_true)
    k_n = max(1, int(math.ceil(k * n)))
    idx = np.argsort(-y_proba)[:k_n]
    return float(y_true[idx].sum()) / max(1, int(y_true.sum()))

def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5, k_list: List[float] = [0.001, 0.005, 0.01]) -> Dict:
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
    }
    for k in k_list:
        metrics[f"recall@{k}"] = recall_at_k(y_true, y_proba, k)
    return metrics

def calibrate_probabilities(y_val: np.ndarray, proba_val: np.ndarray, proba_test: np.ndarray, method: str = "isotonic") -> np.ndarray:
    proba_val = np.asarray(proba_val).reshape(-1)
    proba_test = np.asarray(proba_test).reshape(-1)
    if method == "isotonic":
        iso = CustomIsotonicRegression(out_of_bounds="clip")
        iso.fit(proba_val, y_val)
        return iso.transform(proba_test)
    else:
        return proba_test  # Bỏ 'platt' để tránh lỗi

# -----------------------------
# Plotting helpers
# -----------------------------
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

    plt.figure(figsize=(10, 4))
    for data_name, metrics in evals_result.items():
        if "logloss" in metrics:
            plt.plot(metrics["logloss"], label=f"{data_name} logloss")
    plt.title("XGBoost eval: logloss over boosting rounds")
    plt.xlabel("Boosting round")
    plt.ylabel("logloss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "evals_logloss.png"), dpi=150)
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

def plot_calibration(y_true_val, y_proba_val, out_path: str, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_proba_val, bins) - 1
    prob_true = []
    prob_pred = []
    for i in range(n_bins):
        idx = binids == i
        if idx.sum() == 0:
            prob_true.append(np.nan)
            prob_pred.append((bins[i] + bins[i+1]) / 2.0)
        else:
            prob_true.append(y_true_val[idx].mean())
            prob_pred.append(y_proba_val[idx].mean())
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, "o-")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives"); plt.title("Calibration plot")
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(out_path, "calibration.png"), dpi=150)
    plt.close()

# -----------------------------
# Data Preprocessing & Feature Engineering - Cải thiện để chống overfitting
# -----------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Time-based features để capture temporal patterns
    df['hour_of_day'] = (df['Time'] % 86400) / 3600
    df['day_of_week'] = ((df['Time'] / 86400) % 7).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['time_since_midnight'] = df['Time'] % 86400
    
    # Amount-based features để capture spending patterns
    if 'Amount' in df.columns:
        # Log transformation để giảm skewness
        df['log_amount'] = np.log1p(df['Amount'])
        
        # Binning amount để tạo categorical features
        df['amount_bin'] = pd.cut(df['Amount'], 
                                  bins=[-0.1, 0, 10, 50, 200, 1000, np.inf], 
                                  labels=['zero', 'micro', 'small', 'medium', 'large', 'huge'])
        df['amount_bin'] = df['amount_bin'].cat.codes
        
        # Rolling statistics để capture spending trends (chỉ cho training)
        if len(df) > 100:  # Tránh memory issue
            window_size = min(50, len(df) // 10)
            df['amount_rolling_mean'] = df['Amount'].rolling(window=window_size, min_periods=1).mean()
            df['amount_rolling_std'] = df['Amount'].rolling(window=window_size, min_periods=1).std().fillna(0)
        else:
            df['amount_rolling_mean'] = df['Amount']
            df['amount_rolling_std'] = 0
    
    # V-features statistical summaries để reduce dimensionality
    v_cols = [col for col in df.columns if col.startswith('V') and col != 'V']
    if len(v_cols) > 0:
        df['v_sum'] = df[v_cols].sum(axis=1)
        df['v_mean'] = df[v_cols].mean(axis=1) 
        df['v_std'] = df[v_cols].std(axis=1).fillna(0)
        df['v_min'] = df[v_cols].min(axis=1)
        df['v_max'] = df[v_cols].max(axis=1)
        df['v_range'] = df['v_max'] - df['v_min']
        
        # Count of extreme values để detect outliers
        v_array = df[v_cols].values
        df['v_n_zeros'] = (v_array == 0).sum(axis=1)
        df['v_n_negative'] = (v_array < 0).sum(axis=1)
        df['v_n_extreme'] = (np.abs(v_array) > 2).sum(axis=1)  # >2 std từ mean
    
    # Interaction features giữa Time và Amount (quan trọng cho fraud detection)
    if 'Amount' in df.columns:
        df['time_amount_interaction'] = df['hour_of_day'] * np.log1p(df['Amount'])
        df['weekend_amount'] = df['is_weekend'] * np.log1p(df['Amount'])
    
    # Standardization - quan trọng để tránh overfitting
    numerical_features = [col for col in df.columns if col not in ['Class'] and df[col].dtype in ['float64', 'int64']]
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df

# -----------------------------
# Data Loading & Splitting
# -----------------------------
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
    derived_features = ["hour_of_day"]
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
    random_state: int = 42,
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

# -----------------------------
# Firefly Algorithm
# -----------------------------
@dataclass
class FAConfig:
    n_fireflies: int = 50  # Tăng để tìm kiếm tốt hơn
    n_epochs: int = 25     # Tăng để convergence tốt hơn
    alpha: float = 0.25    # Tăng exploration
    beta0: float = 2.0     # Tăng attraction
    gamma: float = 0.20    # Tăng decay rate
    lambda_feat: float = 0.01  # Tăng feature regularization mạnh hơn
    diversity_threshold: float = 0.1  # Threshold cho diversity preservation
    patience: int = 8      # Early stopping patience cho FA
    random_state: int = 42

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def initialize_population(n_fireflies: int, dim: int, rng: np.random.RandomState) -> np.ndarray:
    return rng.uniform(low=-1.0, high=1.0, size=(n_fireflies, dim))

def _apply_resampling(X: pd.DataFrame, y: pd.Series, method: int, ratio: float, k_neighbors: int) -> Tuple[pd.DataFrame, pd.Series]:
    # Kích hoạt resampling để xử lý imbalanced data
    if method == 0 or not HAS_IMBLEARN:
        return X, y
    
    # Cải thiện tham số
    ratio = float(max(0.3, min(1.0, ratio)))  # Tăng min ratio từ 0.1 lên 0.3
    k_neighbors = int(max(5, min(15, k_neighbors)))  # Giảm max k_neighbors
    
    try:
        # Đảm bảo có đủ samples cho SMOTE
        minority_samples = int(y.sum())
        if minority_samples < k_neighbors + 1:
            k_neighbors = max(1, minority_samples - 1)
            
        # Adaptive SMOTE với tham số tối ưu
        sm = BorderlineSMOTE(
            sampling_strategy=ratio, 
            k_neighbors=k_neighbors,
            m_neighbors=min(10, k_neighbors), 
            random_state=42,
            kind='borderline-1'  # Tập trung vào borderline cases
        )
        
        if method == 1:  # Chỉ SMOTE
            Xr, yr = sm.fit_resample(X, y)
        else:  # SMOTE + ENN
            # ENN với tham số conservative hơn
            enn = EditedNearestNeighbours(
                n_neighbors=3,  # Giảm từ default để ít aggressive hơn
                kind_sel='mode'
            )
            pipe = make_pipeline(sm, enn)
            Xr, yr = pipe.fit_resample(X, y)
            
        print(f"[INFO] Resampling: {len(X)} -> {len(Xr)} samples, fraud ratio: {yr.sum()/len(yr):.4f}")
        return pd.DataFrame(Xr, columns=X.columns), pd.Series(yr)
        
    except Exception as e:
        print(f"[WARN] Resampling failed: {e}. Using original data.")
        return X, y

def evaluate_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feat_mask: np.ndarray,
    hyperparams: Dict,
    resample_cfg: Dict,
) -> float:
    if feat_mask.sum() == 0:
        return 0.0
    cols = X_train.columns[feat_mask.astype(bool)]
    X_tr_use = X_train[cols]
    y_tr_use = y_train
    X_tr_use, y_tr_use = _apply_resampling(
        X_tr_use,
        y_tr_use,
        method=int(resample_cfg.get("method", 0)),
        ratio=float(resample_cfg.get("ratio", 0.5)),  # Giảm ratio mặc định
        k_neighbors=int(resample_cfg.get("k_neighbors", 5)),
    )
    if resample_cfg.get("method", 0) != 0 and HAS_IMBLEARN:
        hp_spw = 1.0
    else:
        pos = int(pd.Series(y_tr_use).sum()); neg = len(y_tr_use) - pos
        hp_spw = max(1.0, neg / max(1, pos))
    
    # XGBoost với regularization mạnh hơn chống overfitting    
    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="gpu_hist",
        n_gpus=2,  
        eval_metric="aucpr",
        n_estimators=200,  # Giảm từ 400 để tránh overfitting
        learning_rate=hyperparams.get("eta", 0.03),  # Giảm learning rate mặc định
        max_depth=min(int(hyperparams.get("max_depth", 4)), 6),  # Giảm max_depth
        subsample=hyperparams.get("subsample", 0.7),  # Giảm subsample
        colsample_bytree=hyperparams.get("colsample_bytree", 0.7),  # Giảm colsample
        colsample_bylevel=0.7,  # Thêm column sampling theo level
        colsample_bynode=0.8,   # Thêm column sampling theo node
        reg_alpha=hyperparams.get("reg_alpha", 1.0),  # Thêm L1 regularization
        reg_lambda=hyperparams.get("reg_lambda", 3.0),  # Tăng L2 regularization
        min_child_weight=hyperparams.get("min_child_weight", 3.0),  # Tăng min_child_weight
        gamma=hyperparams.get("gamma", 0.1),  # Thêm gamma để pruning
        max_delta_step=1,  # Thêm constraint cho imbalanced data
        scale_pos_weight=hp_spw,
        n_jobs=1,
        random_state=42,
        enable_categorical=True,
    )
    try:
        start = time.time()
        model.fit(
            X_tr_use, y_tr_use, 
            eval_set=[(X_val[cols], y_val)], 
            verbose=False, 
            early_stopping_rounds=50  # Tăng early stopping
        )
        proba = model.predict_proba(X_val[cols])[:, 1]
        ap = float(average_precision_score(y_val, proba))
        
        # Penalize overfitting bằng cách kiểm tra train/val gap
        train_proba = model.predict_proba(X_tr_use)[:, 1]
        train_ap = float(average_precision_score(y_tr_use, train_proba))
        overfitting_penalty = max(0, (train_ap - ap - 0.05)) * 0.5  # Penalty nếu gap > 5%
        
        final_score = ap - overfitting_penalty
        print(f"[DEBUG] Candidate eval: {time.time() - start:.2f}s, rounds={model.best_iteration or 200}, AP={ap:.5f}, train_AP={train_ap:.5f}, final={final_score:.5f}")
        return final_score
    except Exception as e:
        print(f"[WARN] Candidate eval failed: {e}")
        return 0.0

def decode_position(pos: np.ndarray, n_features: int) -> Tuple[np.ndarray, Dict, Dict]:
    feat_mask = (sigmoid(pos[:n_features]) >= 0.5).astype(int)
    hp_vals = pos[n_features:]
    def scale(v, lo, hi):
        return lo + (v + 1) / 2 * (hi - lo)
    hp = {}
    hp_pad = np.concatenate([hp_vals, np.zeros(max(0, 13 - len(hp_vals)))])  # Tăng từ 11 lên 13
    
    # Tối ưu hyperparameters để chống overfitting
    hp["eta"] = float(scale(hp_pad[0], 0.01, 0.08))  # Giảm learning rate range
    hp["max_depth"] = int(round(scale(hp_pad[1], 3, 6)))  # Giảm max depth
    hp["subsample"] = float(scale(hp_pad[2], 0.6, 0.85))  # Giảm subsample range
    hp["colsample_bytree"] = float(scale(hp_pad[3], 0.6, 0.85))  # Giảm colsample range
    hp["reg_lambda"] = float(scale(hp_pad[4], 2.0, 10.0))  # Tăng L2 regularization
    hp["reg_alpha"] = float(scale(hp_pad[5], 0.5, 5.0))   # Thêm L1 regularization  
    hp["min_child_weight"] = float(scale(hp_pad[6], 2.0, 8.0))  # Tăng min_child_weight
    hp["gamma"] = float(scale(hp_pad[7], 0.0, 1.0))  # Thêm gamma pruning
    hp["max_delta_step"] = int(round(scale(hp_pad[8], 0, 5)))  # Giảm max_delta_step
    
    # Resampling parameters  
    raw_method = hp_pad[9]
    method = int(np.clip(np.floor((raw_method + 1) / 2 * 3), 0, 2))
    ratio = float(scale(hp_pad[10], 0.3, 0.8))  # Giảm ratio range để tránh over-sampling
    k_neighbors = int(round(scale(hp_pad[11], 5, 12)))  # Giảm k_neighbors range
    
    resample_cfg = {"method": method, "ratio": ratio, "k_neighbors": k_neighbors}
    return feat_mask, hp, resample_cfg

def firefly_optimize(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: FAConfig,
) -> Tuple[np.ndarray, Dict, Dict, float]:
    rng = np.random.RandomState(config.random_state)
    n_features = X_train.shape[1]
    dim = n_features + 13  # Tăng từ 11 lên 13 cho thêm hyperparameters

    pop = initialize_population(config.n_fireflies, dim, rng)

    def fitness(pos):
        feat_mask, hp, rcfg = decode_position(pos, n_features)
        ap = evaluate_candidate(X_train, y_train, X_val, y_val, feat_mask, hp, rcfg)
        penalty = config.lambda_feat * float(feat_mask.sum())
        return ap - penalty

    fits = np.array([fitness(p) for p in pop])
    best_fit_history = [max(fits)]
    best_pop = pop.copy()
    
    # Cải thiện convergence tracking
    stagnation_count = 0
    alpha0, beta0, gamma0 = config.alpha, config.beta0, config.gamma

    for epoch in range(config.n_epochs):
        print(f"[INFO] Starting FA epoch {epoch + 1}/{config.n_epochs}")
        start_epoch = time.time()
        
        # Adaptive parameters để tránh premature convergence
        alpha = alpha0 * (0.9 ** epoch)  # Slower decay
        gamma = gamma0 * (0.95 ** epoch)
        
        # Preserve diversity bằng cách thêm mutation cho worst performers
        sorted_indices = np.argsort(fits)
        worst_quarter = int(config.n_fireflies * 0.25)
        
        for i in range(config.n_fireflies):
            for j in range(config.n_fireflies):
                if fits[j] > fits[i]:
                    rij = np.linalg.norm(pop[i] - pop[j])
                    beta = beta0 * np.exp(-gamma * (rij ** 2))
                    eps = rng.uniform(-0.5, 0.5, size=dim)
                    pop[i] = pop[i] + beta * (pop[j] - pop[i]) + alpha * eps
                    pop[i] = np.clip(pop[i], -1, 1)
            
            # Diversity preservation: mutate worst performers
            if i in sorted_indices[:worst_quarter]:
                mutation_rate = 0.3 * (1.0 - epoch / config.n_epochs)  # Decay mutation
                if rng.random() < mutation_rate:
                    mutation = rng.normal(0, 0.2, dim)
                    pop[i] = np.clip(pop[i] + mutation, -1, 1)
        
        fits = np.array([fitness(p) for p in pop])
        current_best = max(fits)
        
        # Improved convergence tracking
        if current_best > best_fit_history[-1] + 1e-6:  # Minimum improvement threshold
            best_fit_history.append(current_best)
            best_pop = pop.copy()
            stagnation_count = 0
        else:
            best_fit_history.append(best_fit_history[-1])
            stagnation_count += 1
            
        print(f"[INFO] Epoch {epoch + 1} took {time.time() - start_epoch:.2f}s, best fitness: {current_best:.5f}, stagnation: {stagnation_count}")
        
        # Early stopping với improved patience
        if stagnation_count >= config.patience:
            print(f"Early stopping FA at epoch {epoch + 1} due to stagnation, best fitness: {best_fit_history[-1]:.5f}")
            break

    best_idx = int(np.argmax(fits))
    best_pos = best_pop[best_idx]
    best_fit = float(best_fit_history[-1])
    best_mask, best_hp, best_rcfg = decode_position(best_pos, n_features)
    return best_mask, best_hp, best_rcfg, best_fit

# -----------------------------
# Overfitting prevention helpers - Cải thiện validation
# -----------------------------
def xgb_find_best_nrounds(X: pd.DataFrame, y: pd.Series, params: Dict, num_boost_round: int = 1500, early_stopping_rounds: int = 75, folds: int = 5):
    dtrain = DMatrix(X, label=y)
    xgb_params = params.copy()
    if "objective" not in xgb_params:
        xgb_params["objective"] = "binary:logistic"
    
    # Time-series aware cross-validation để tránh data leakage
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=folds, test_size=len(X)//folds)
    
    res = xgb_cv(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        folds=tscv.split(X, y),  # Sử dụng time-series splits
        metrics=("aucpr", "logloss"),
        early_stopping_rounds=early_stopping_rounds,
        seed=42,
        as_pandas=True,
        verbose_eval=False,
    )
    best_n = int(len(res))
    best_score = float(res["test-aucpr-mean"].max()) if "test-aucpr-mean" in res.columns else None
    return best_n, res

def nested_cross_validation(X: pd.DataFrame, y: pd.Series, params: Dict, n_outer_folds: int = 3) -> float:
    """Nested CV để đánh giá không bias model performance"""
    from sklearn.model_selection import TimeSeriesSplit
    
    outer_cv = TimeSeriesSplit(n_splits=n_outer_folds, test_size=len(X)//n_outer_folds)
    scores = []
    
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
        
        # Inner CV để tìm best parameters (đã được FA tối ưu)
        model = XGBClassifier(**params, random_state=42)
        model.fit(X_train_outer, y_train_outer)
        
        proba = model.predict_proba(X_test_outer)[:, 1]
        score = average_precision_score(y_test_outer, proba)
        scores.append(score)
    
    return np.mean(scores)

# -----------------------------
# Ensemble Methods để giảm overfitting
# -----------------------------
class EnsembleModel:
    def __init__(self, base_models: List, meta_model=None, use_stacking: bool = True):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(C=0.1, random_state=42)  # Regularized meta-model
        self.use_stacking = use_stacking
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        # Train base models
        for i, model in enumerate(self.base_models):
            print(f"[INFO] Training base model {i+1}/{len(self.base_models)}")
            if hasattr(model, 'fit'):
                if 'early_stopping_rounds' in model.get_params() and X_val is not None:
                    model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X, y)
        
        if self.use_stacking and X_val is not None:
            # Create meta-features using validation predictions
            meta_features = np.column_stack([
                model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba')
                else model.predict(X_val) for model in self.base_models
            ])
            self.meta_model.fit(meta_features, y_val)
            
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if self.use_stacking:
            meta_features = np.column_stack([
                model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba')
                else model.predict(X) for model in self.base_models
            ])
            return self.meta_model.predict_proba(meta_features)
        else:
            # Simple averaging
            predictions = np.column_stack([
                model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba')
                else model.predict(X) for model in self.base_models
            ])
            avg_pred = predictions.mean(axis=1)
            return np.column_stack([1 - avg_pred, avg_pred])

def create_diverse_models(best_hp: Dict, spw: float, n_estimators: int = 200) -> List:
    """Tạo diverse base models để ensemble"""
    models = []
    
    # Model 1: Conservative XGBoost (chống overfitting mạnh)
    conservative_params = best_hp.copy()
    conservative_params.update({
        'n_estimators': n_estimators,
        'learning_rate': conservative_params.get('eta', 0.03) * 0.8,  # Slower learning
        'max_depth': min(4, conservative_params.get('max_depth', 4)),  # Shallower trees
        'reg_alpha': conservative_params.get('reg_alpha', 1.0) * 2.0,  # More L1 reg
        'reg_lambda': conservative_params.get('reg_lambda', 3.0) * 1.5,  # More L2 reg
        'subsample': 0.7,
        'colsample_bytree': 0.7,
    })
    models.append(XGBClassifier(
        objective="binary:logistic", tree_method="gpu_hist", n_gpus=2,
        scale_pos_weight=spw, random_state=42, **conservative_params
    ))
    
    # Model 2: Balanced XGBoost  
    balanced_params = best_hp.copy()
    balanced_params.update({
        'n_estimators': n_estimators,
        'learning_rate': conservative_params.get('eta', 0.03),
        'max_depth': conservative_params.get('max_depth', 5),
        'reg_alpha': conservative_params.get('reg_alpha', 1.0),
        'reg_lambda': conservative_params.get('reg_lambda', 3.0),
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    })
    models.append(XGBClassifier(
        objective="binary:logistic", tree_method="gpu_hist", n_gpus=2,
        scale_pos_weight=spw, random_state=43, **balanced_params  # Different seed
    ))
    
    # Model 3: Regularized Logistic Regression cho diversity
    models.append(LogisticRegression(
        C=0.01,  # Strong regularization
        penalty='elasticnet',
        l1_ratio=0.5,
        solver='saga',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ))
    
    return models
def train_fa_xgb(X_tr, y_tr, X_va, y_va, X_te, y_te, seed=42, fa_cfg: Optional[FAConfig] = None, out_dir: str = "outputs"):
    ensure_outputs_dir(out_dir)
    cfg = fa_cfg or FAConfig(random_state=seed)
    start = time.time()
    print("[INFO] Checking GPU availability...")
    # !nvidia-smi
    best_mask, best_hp, best_rcfg, best_fit = firefly_optimize(X_tr, y_tr, X_va, y_va, cfg)
    
    fa_config = {
        "best_mask": best_mask.tolist(),
        "best_hyperparams": best_hp,
        "best_resampling_config": best_rcfg,
        "best_fitness": best_fit,
        "fa_time_sec": round(time.time() - start, 2),
        "selected_feature_indices": np.where(best_mask)[0].tolist()
    }
    fa_config_path = os.path.join(out_dir, "fa_optimized_config.json")
    with open(fa_config_path, "w", encoding="utf-8") as f:
        json.dump(fa_config, f, indent=2)
    print(f"[INFO] Saved FA-optimized config to {fa_config_path}")
    
    cols = X_tr.columns[best_mask.astype(bool)]
    if len(cols) == 0:
        raise RuntimeError("FA selected zero features.")
    X_tr_use = X_tr[cols].copy()
    y_tr_use = y_tr.copy()
    X_tr_use, y_tr_use = _apply_resampling(
        X_tr_use,
        y_tr_use,
        method=int(best_rcfg.get("method", 0)),
        ratio=float(best_rcfg.get("ratio", 1.0)),
        k_neighbors=int(best_rcfg.get("k_neighbors", 5)),
    )
    if best_rcfg.get("method", 0) != 0 and HAS_IMBLEARN:
        spw_final = 1.0
    else:
        pos = int(pd.Series(y_tr_use).sum()); neg = len(y_tr_use) - pos
        spw_final = max(1.0, neg / max(1, pos))
    base_params = {
        "objective": "binary:logistic",
        "learning_rate": best_hp.get("eta", 0.03),
        "max_depth": int(best_hp.get("max_depth", 6)),
        "subsample": best_hp.get("subsample", 0.8),
        "colsample_bytree": best_hp.get("colsample_bytree", 0.8),
        "reg_lambda": best_hp.get("reg_lambda", 2.0),
        "min_child_weight": best_hp.get("min_child_weight", 1.0),
        "scale_pos_weight": spw_final,
        "tree_method": "gpu_hist",
        "n_gpus": 2,  # Ép dùng 2 GPU T4
        "eval_metric": ["aucpr", "logloss"],
    }
    try:
        best_nrounds, cvres = xgb_find_best_nrounds(X_tr_use, y_tr_use, base_params, num_boost_round=1000, early_stopping_rounds=75, folds=5)
        print(f"[INFO] XGB CV chose {best_nrounds} rounds (approx).")
        
        # Nested CV để đánh giá model performance không bias
        nested_score = nested_cross_validation(X_tr_use, y_tr_use, base_params, n_outer_folds=3)
        print(f"[INFO] Nested CV score: {nested_score:.5f}")
        
        try:
            plt.figure(figsize=(12,4))
            if "test-aucpr-mean" in cvres.columns:
                plt.subplot(1,2,1)
                plt.plot(cvres["test-aucpr-mean"], label="test-aucpr-mean", alpha=0.8)
                plt.plot(cvres["train-aucpr-mean"], label="train-aucpr-mean", alpha=0.8)
                # Highlight overfitting region
                gap = cvres["train-aucpr-mean"] - cvres["test-aucpr-mean"]
                plt.fill_between(range(len(gap)), 
                                cvres["test-aucpr-mean"], 
                                cvres["train-aucpr-mean"],
                                alpha=0.3, color='red', label='overfitting gap')
                plt.xlabel("boosting round"); plt.ylabel("aucpr"); plt.legend(); plt.grid(True)
                plt.title("Training vs Validation Performance")
                
                plt.subplot(1,2,2)
                plt.plot(gap, label="train-val gap", color='red')
                plt.axhline(y=0.05, color='orange', linestyle='--', label='overfitting threshold')
                plt.xlabel("boosting round"); plt.ylabel("performance gap"); plt.legend(); plt.grid(True)
                plt.title("Overfitting Detection")
                
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "xgb_cv_analysis.png"), dpi=150)
                plt.close()
        except Exception as e:
            print(f"[WARN] failed to plot xgb cv: {e}")
    except Exception as e:
        print(f"[WARN] xgb cv failed: {e}; falling back to default n_estimators 600")
        best_nrounds = 600

    # Train both single model và ensemble
    print("[INFO] Training single XGBoost model...")
    clf = XGBClassifier(
        objective="binary:logistic",
        tree_method="gpu_hist",
        n_gpus=2,  
        n_estimators=max(50, best_nrounds),
        learning_rate=base_params["learning_rate"],
        max_depth=base_params["max_depth"],
        subsample=base_params["subsample"],
        colsample_bytree=base_params["colsample_bytree"],
        reg_alpha=base_params.get("reg_alpha", 1.0),
        reg_lambda=base_params["reg_lambda"],
        min_child_weight=base_params["min_child_weight"],
        gamma=base_params.get("gamma", 0.1),
        scale_pos_weight=base_params["scale_pos_weight"],
        eval_metric=["aucpr", "logloss"],
        n_jobs=-1,
        random_state=seed,
        enable_categorical=True,
    )
    evals_result = {}
    model_checkpoint_path = os.path.join(out_dir, "model_checkpoint.pkl")
    gpu_success = False
    
    try:
        print("[INFO] Training final XGBoost model with GPU...")
        clf.fit(
            X_tr_use, y_tr_use,
            eval_set=[(X_va[cols], y_va)],
            verbose=False,
            early_stopping_rounds=100,
        )
        evals_result = clf.evals_result()
        gpu_success = True
        joblib.dump(clf, model_checkpoint_path)
        print(f"[INFO] Trained and saved model checkpoint (GPU) to {model_checkpoint_path}")
    except Exception as e:
        print(f"[WARN] GPU training failed: {e}. Falling back to CPU.")
        clf.tree_method = "hist"
        clf.fit(X_tr_use, y_tr_use, eval_set=[(X_va[cols], y_va)], verbose=False, early_stopping_rounds=100)
        evals_result = clf.evals_result()
        joblib.dump(clf, model_checkpoint_path)
        print(f"[INFO] Trained and saved model checkpoint (CPU fallback) to {model_checkpoint_path}")

    # Train ensemble model cho better generalization
    print("[INFO] Training ensemble model...")
    try:
        base_models = create_diverse_models(best_hp, spw_final, n_estimators=max(50, best_nrounds//2))
        ensemble = EnsembleModel(base_models, use_stacking=True)
        ensemble.fit(X_tr_use, y_tr_use, X_va[cols], y_va)
        
        # So sánh single vs ensemble performance
        proba_val_single = clf.predict_proba(X_va[cols])[:, 1]
        proba_val_ensemble = ensemble.predict_proba(X_va[cols])[:, 1]
        
        ap_single = float(average_precision_score(y_va.values, proba_val_single))
        ap_ensemble = float(average_precision_score(y_va.values, proba_val_ensemble))
        
        print(f"[INFO] Single model val AP: {ap_single:.5f}")
        print(f"[INFO] Ensemble model val AP: {ap_ensemble:.5f}")
        
        # Choose better model
        if ap_ensemble > ap_single:
            print("[INFO] Using ensemble model for final predictions")
            best_model = ensemble
            proba_val = proba_val_ensemble
        else:
            print("[INFO] Using single model for final predictions")
            best_model = clf
            proba_val = proba_val_single
            
        # Save ensemble if it's better
        if ap_ensemble > ap_single:
            ensemble_path = os.path.join(out_dir, "ensemble_model.pkl") 
            joblib.dump(ensemble, ensemble_path)
            print(f"[INFO] Saved ensemble model to {ensemble_path}")
            
    except Exception as e:
        print(f"[WARN] Ensemble training failed: {e}. Using single model.")
        best_model = clf
        proba_val = clf.predict_proba(X_va[cols])[:, 1]
    # Final test predictions với model tốt nhất
    proba_test = best_model.predict_proba(X_te[cols])[:, 1]
    proba_test_cal = calibrate_probabilities(y_va.values, proba_val, proba_test, method="isotonic")
    thr, cost_stats = choose_threshold_by_cost(y_va.values, proba_val)
    metrics = compute_metrics(y_te.values, proba_test_cal, threshold=thr)
    metrics.update({
        "chosen_threshold": thr,
        "cost_stats": cost_stats,
        "selected_features": list(cols),
        "n_selected": int(len(cols)),
        "fa_resampling": best_rcfg,
        "fa_best_fitness": best_fit,
        "fa_time_sec": round(time.time() - start, 2),
    })

    try:
        if evals_result:
            plot_xgb_evals(evals_result, out_dir)
    except Exception as e:
        print(f"[WARN] plot evals failed: {e}")
    try:
        plot_roc_pr(y_va.values, proba_val, y_te.values, proba_test_cal, out_dir)
        plot_feature_importance(clf, list(cols), out_dir)  # Use original clf for feature importance
        plot_calibration(y_va.values, proba_val, out_dir)
        print("[INFO] All plots generated successfully.")
    except Exception as e:
        print(f"[WARN] additional plotting failed: {e}")

    return {"model": best_model, "single_model": clf, "metrics": metrics, "val_pr_auc": float(average_precision_score(y_va, proba_val)), "evals_result": evals_result, "checkpoint": model_checkpoint_path}

def convert_to_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    if isinstance(obj, tuple):
        return tuple(convert_to_serializable(i) for i in obj)
    return obj

# -----------------------------
# Main Runner
# -----------------------------
def run_experiments(
    data_path: str,
    features: List[str] = None,
    seed: int = 42,
    embargo: int = 0,
    out_dir: str = "outputs",
):
    features = features or DEFAULT_FEATURES
    X, y = load_dataset(data_path, features)

    scaler = StandardScaler()
    X_tr, X_va, X_te, y_tr, y_va, y_te = chronological_train_val_test_split(
        X, y, time_col="Time", embargo=embargo
    )

    results = {}
    start = time.time()
    out = train_fa_xgb(X_tr.copy(), y_tr.copy(), X_va.copy(), y_va.copy(), X_te.copy(), y_te.copy(), seed, fa_cfg=None, out_dir=out_dir)
    duration = round(time.time() - start, 2)
    res = {"metrics": out["metrics"], "val_pr_auc": out.get("val_pr_auc", None), "time_sec": duration}
    results["fa_xgb"] = res
    print(f"Scenario fa_xgb done in {duration}s | Test PR-AUC: {res['metrics']['pr_auc']:.5f} | ROC-AUC: {res['metrics']['roc_auc']:.5f}")

    ensure_outputs_dir(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"fraud_v3_results_{ts}.json")
    data_to_dump = convert_to_serializable({
        "data_path": data_path,
        "features": X.columns.tolist(),
        "embargo": embargo,
        "seed": seed,
        "results": results,
    })
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data_to_dump, f, indent=2)
    print(f"Saved results to {out_file}")

    model_data = {
        "model": out["model"],
        "scaler": scaler
    }
    joblib.dump(model_data, os.path.join(out_dir, 'fraud_model.pkl'))
    print("Saved model and scaler to fraud_model.pkl")

    try:
        config_data = {
            "metrics": convert_to_serializable(out["metrics"]),
            "selected_features": out["metrics"]["selected_features"],
            "chosen_threshold": out["metrics"]["chosen_threshold"]
        }
        with open(os.path.join(out_dir, 'fraud_config.json'), 'w', encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        print("Saved config to fraud_config.json")
    except Exception as e:
        print(f"Error saving fraud_config.json: {e}")

def parse_args():
    p = argparse.ArgumentParser(description="Fraud detection experiments: FA + XGBoost + plots + anti-overfitting")
    p.add_argument("--data", type=str, default="/kaggle/input/creditcardfraud/creditcard.csv", help="Path to fraud dataset CSV")
    p.add_argument("--embargo", type=int, default=0, help="Embargo samples between splits to avoid leakage")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs", help="Output directory for plots and artifacts")
    args = p.parse_args([])
    return args

if __name__ == "__main__":
    args = parse_args()
    try:
        run_experiments(
            data_path=args.data,
            seed=args.seed,
            embargo=args.embargo,
            out_dir=args.out,
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Ensure dataset path is correct and required packages are installed: xgboost scikit-learn==1.2.2 imbalanced-learn matplotlib joblib scipy==1.13.1")