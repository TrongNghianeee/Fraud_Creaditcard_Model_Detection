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
    precision_score,
    recall_score,
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
# Firefly Algorithm - Enhanced for better feature selection
# -----------------------------
@dataclass
class FAConfig:
    n_fireflies: int = 40     # Tăng từ 50 lên 40 để balance performance/speed
    n_epochs: int = 25        # Tăng từ 25 để convergence tốt hơn 
    alpha: float = 0.25       # Giữ nguyên exploration
    beta0: float = 2.0        # Giữ nguyên attraction
    gamma: float = 0.20       # Giữ nguyên decay rate
    lambda_feat: float = 0.02 # Tăng feature regularization để select ít feature hơn
    diversity_threshold: float = 0.1  # Threshold cho diversity preservation
    patience: int = 6         # Giảm patience từ 10 -> 6 để early stop sớm hơn, tránh overfitting
    validation_strictness: float = 0.8  # Strict validation - require high consistency across folds
    overfitting_threshold: float = 0.02  # Lower threshold để detect overfitting sớm hơn
    random_state: int = 42
    
    # Feature selection strategy
    feature_selection_mode: str = "flexible"  # "flexible" hoặc "fixed_count"
    target_feature_count: int = 14           # Số features mục tiêu khi dùng "fixed_count"
    min_feature_count: int = 8               # Tối thiểu features khi dùng "flexible"

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def initialize_population(n_fireflies: int, dim: int, rng: np.random.RandomState, config: FAConfig = None) -> np.ndarray:
    """
    Initialize population với 2 strategies:
    - flexible: tự do chọn features
    - fixed_count: cố định số lượng features
    """
    pop = rng.uniform(low=-1.0, high=1.0, size=(n_fireflies, dim))
    
    if config and config.feature_selection_mode == "fixed_count":
        n_features = dim - 13  # trừ đi hyperparameters
        target_count = config.target_feature_count
        
        # Đảm bảo mỗi firefly có đúng target_count features được chọn
        for i in range(n_fireflies):
            # Reset feature selection part
            feature_part = pop[i, :n_features]
            
            # Chọn ngẫu nhiên target_count features để activate
            selected_indices = rng.choice(n_features, size=min(target_count, n_features), replace=False)
            
            # Set all features to negative (not selected)
            feature_part[:] = rng.uniform(-2.0, -0.5, n_features)
            
            # Set selected features to positive (selected)
            feature_part[selected_indices] = rng.uniform(0.5, 2.0, len(selected_indices))
            
            pop[i, :n_features] = feature_part
            
        print(f"[INFO] Initialized population with fixed {target_count} features per firefly")
    else:
        print(f"[INFO] Initialized population with flexible feature selection")
    
    return pop

def _apply_resampling(X: pd.DataFrame, y: pd.Series, method: int, ratio: float, k_neighbors: int) -> Tuple[pd.DataFrame, pd.Series]:
    # Kích hoạt resampling với ratio thấp hơn để tránh over-sampling
    if method == 0 or not HAS_IMBLEARN:
        return X, y
    
    # Cải thiện tham số - giảm ratio từ 0.42 xuống 0.1-0.2
    ratio = float(max(0.05, min(0.12, ratio)))  # Giảm ratio range xuống 5-12% để tránh overfitting
    k_neighbors = int(max(5, min(15, k_neighbors)))
    
    try:
        # Đảm bảo có đủ samples cho SMOTE
        minority_samples = int(y.sum())
        if minority_samples < k_neighbors + 1:
            k_neighbors = max(1, minority_samples - 1)
            
        # Conservative SMOTE với tham số tối ưu
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
                n_neighbors=3,  # Conservative cleaning
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
        ratio=float(resample_cfg.get("ratio", 0.15)),  # Giảm ratio mặc định từ 0.5 -> 0.15
        k_neighbors=int(resample_cfg.get("k_neighbors", 5)),
    )
    if resample_cfg.get("method", 0) != 0 and HAS_IMBLEARN:
        hp_spw = 1.0
    else:
        pos = int(pd.Series(y_tr_use).sum()); neg = len(y_tr_use) - pos
        hp_spw = max(1.0, neg / max(1, pos))
    
    # XGBoost với regularization mạnh hơn và early stopping tốt hơn
    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="gpu_hist",
        n_gpus=2,  
        eval_metric="aucpr",
        n_estimators=300,  # Giảm từ 400 để tránh overfitting
        learning_rate=hyperparams.get("eta", 0.02),  # Giảm learning rate hơn nữa
        max_depth=min(int(hyperparams.get("max_depth", 4)), 5),  # Giảm max_depth tối đa
        subsample=hyperparams.get("subsample", 0.65),  # Giảm subsample hơn nữa
        colsample_bytree=hyperparams.get("colsample_bytree", 0.65),  # Giảm colsample
        colsample_bylevel=0.65,  # Giảm column sampling theo level
        colsample_bynode=0.75,   # Giảm column sampling theo node
        reg_alpha=hyperparams.get("reg_alpha", 2.0),  # Tăng L1 regularization
        reg_lambda=hyperparams.get("reg_lambda", 5.0),  # Tăng L2 regularization mạnh hơn
        min_child_weight=hyperparams.get("min_child_weight", 5.0),  # Tăng min_child_weight
        gamma=hyperparams.get("gamma", 0.2),  # Tăng gamma để pruning mạnh hơn
        max_delta_step=1,  # Giữ constraint cho imbalanced data
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
            early_stopping_rounds=50  # Tăng early stopping patience ~50 rounds
        )
        proba = model.predict_proba(X_val[cols])[:, 1]
        ap = float(average_precision_score(y_val, proba))
        
        # Enhanced overfitting penalty với multiple checks
        train_proba = model.predict_proba(X_tr_use)[:, 1]
        train_ap = float(average_precision_score(y_tr_use, train_proba))
        
        # Multi-level overfitting detection
        train_val_gap = train_ap - ap
        overfitting_penalty = 0.0
        
        # Level 1: Gap-based penalty (stricter)
        if train_val_gap > 0.05:
            overfitting_penalty += train_val_gap * 5.0  # Heavy penalty for large gaps
        elif train_val_gap > 0.03:
            overfitting_penalty += train_val_gap * 3.0
        elif train_val_gap > 0.02:
            overfitting_penalty += train_val_gap * 1.5
        
        # Level 2: Absolute performance penalty (realistic for fraud)
        if train_ap > 0.85:  # Train performance quá cao
            overfitting_penalty += 0.1
        if ap > 0.75:        # Validation performance quá cao cho fraud data
            overfitting_penalty += 0.05
            
        # Level 3: F1-based reality check
        val_proba_binary = (proba >= 0.5).astype(int)
        if len(np.unique(val_proba_binary)) == 2:  # Avoid f1_score error
            f1 = float(f1_score(y_val, val_proba_binary, zero_division=0))
            if f1 > 0.6:  # F1 > 0.6 nghi ngờ cho imbalanced fraud data
                overfitting_penalty += (f1 - 0.6) * 0.3
        
        
        final_score = ap - overfitting_penalty
        print(f"[DEBUG] Candidate eval: {time.time() - start:.2f}s, rounds={model.best_iteration or 300}, AP={ap:.5f}, train_AP={train_ap:.5f}, gap={train_val_gap:.5f}, penalty={overfitting_penalty:.5f}, final={final_score:.5f}")
        return final_score
    except Exception as e:
        print(f"[WARN] Candidate eval failed: {e}")
        return 0.0

def decode_position(pos: np.ndarray, n_features: int, config: FAConfig = None) -> Tuple[np.ndarray, Dict, Dict]:
    feat_vals = pos[:n_features]
    
    if config and config.feature_selection_mode == "fixed_count":
        # Cố định số lượng features
        target_count = config.target_feature_count
        
        # Lấy top target_count features có giá trị cao nhất
        top_indices = np.argsort(feat_vals)[-target_count:]
        feat_mask = np.zeros(n_features, dtype=int)
        feat_mask[top_indices] = 1
        
        print(f"[DEBUG] Fixed selection: {target_count} features selected")
    else:
        # Flexible selection với minimum constraint
        feat_mask = (sigmoid(feat_vals) >= 0.5).astype(int)
        
        # Đảm bảo ít nhất min_feature_count features
        if config and feat_mask.sum() < config.min_feature_count:
            min_count = min(config.min_feature_count, n_features)
            top_indices = np.argsort(feat_vals)[-min_count:]
            feat_mask = np.zeros(n_features, dtype=int)
            feat_mask[top_indices] = 1
            print(f"[DEBUG] Enforced minimum: {min_count} features selected")
        else:
            print(f"[DEBUG] Flexible selection: {feat_mask.sum()} features selected")
    
    hp_vals = pos[n_features:]
    def scale(v, lo, hi):
        return lo + (v + 1) / 2 * (hi - lo)
    hp = {}
    hp_pad = np.concatenate([hp_vals, np.zeros(max(0, 13 - len(hp_vals)))])
    
    # Tối ưu hyperparameters để chống overfitting mạnh hơn
    hp["eta"] = float(scale(hp_pad[0], 0.002, 0.02))  # Giảm learning rate xuống 0.002-0.02 để tránh overfitting
    hp["max_depth"] = int(round(scale(hp_pad[1], 3, 5)))  # Giảm max depth tối đa
    hp["subsample"] = float(scale(hp_pad[2], 0.5, 0.8))   # Giảm subsample range
    hp["colsample_bytree"] = float(scale(hp_pad[3], 0.5, 0.8))  # Giảm colsample range
    hp["reg_lambda"] = float(scale(hp_pad[4], 5.0, 25.0))  # Tăng L2 regularization mạnh hơn: 5-25
    hp["reg_alpha"] = float(scale(hp_pad[5], 2.0, 12.0))   # Tăng L1 regularization mạnh hơn: 2-12  
    hp["min_child_weight"] = float(scale(hp_pad[6], 3.0, 10.0))  # Tăng min_child_weight range
    hp["gamma"] = float(scale(hp_pad[7], 0.1, 2.0))  # Tăng gamma pruning range
    hp["max_delta_step"] = int(round(scale(hp_pad[8], 0, 3)))  # Giảm max_delta_step range
    
    # Resampling parameters với ratio thấp hơn
    raw_method = hp_pad[9]
    method = int(np.clip(np.floor((raw_method + 1) / 2 * 3), 0, 2))
    ratio = float(scale(hp_pad[10], 0.05, 0.12))  # Giảm ratio range xuống 5-12% để tránh synthetic leakage
    k_neighbors = int(round(scale(hp_pad[11], 5, 12)))
    
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
    dim = n_features + 13

    pop = initialize_population(config.n_fireflies, dim, rng, config)

    def fitness(pos):
        feat_mask, hp, rcfg = decode_position(pos, n_features, config)
        ap = evaluate_candidate(X_train, y_train, X_val, y_val, feat_mask, hp, rcfg)
        
        # Điều chỉnh penalty dựa trên mode
        if config.feature_selection_mode == "fixed_count":
            # Không penalty cho số lượng features vì đã cố định
            penalty = 0
        else:
            # Flexible mode: penalty cho quá nhiều features
            penalty = config.lambda_feat * float(feat_mask.sum())
        
        return ap - penalty

    fits = np.array([fitness(p) for p in pop])
    best_fit_history = [max(fits)]
    best_pop = pop.copy()
    
    # Enhanced convergence tracking
    stagnation_count = 0
    alpha0, beta0, gamma0 = config.alpha, config.beta0, config.gamma

    for epoch in range(config.n_epochs):
        print(f"[INFO] Starting FA epoch {epoch + 1}/{config.n_epochs}")
        start_epoch = time.time()
        
        # Adaptive parameters với decay chậm hơn
        alpha = alpha0 * (0.92 ** epoch)  # Slower decay để maintain exploration
        gamma = gamma0 * (0.97 ** epoch)
        
        # Enhanced diversity preservation
        sorted_indices = np.argsort(fits)
        worst_third = int(config.n_fireflies * 0.33)  # Tăng từ 25% lên 33%
        
        for i in range(config.n_fireflies):
            for j in range(config.n_fireflies):
                if fits[j] > fits[i]:
                    rij = np.linalg.norm(pop[i] - pop[j])
                    beta = beta0 * np.exp(-gamma * (rij ** 2))
                    eps = rng.uniform(-0.5, 0.5, size=dim)
                    pop[i] = pop[i] + beta * (pop[j] - pop[i]) + alpha * eps
                    pop[i] = np.clip(pop[i], -1, 1)
                    
                    # Enforce feature count constraint sau khi update
                    if config.feature_selection_mode == "fixed_count":
                        n_feat = dim - 13
                        target_count = config.target_feature_count
                        feat_part = pop[i, :n_feat]
                        
                        # Chọn top target_count features
                        top_indices = np.argsort(feat_part)[-target_count:]
                        feat_part[:] = rng.uniform(-2.0, -0.5, n_feat)
                        feat_part[top_indices] = rng.uniform(0.5, 2.0, len(top_indices))
                        pop[i, :n_feat] = feat_part
            
            # Enhanced diversity preservation
            if i in sorted_indices[:worst_third]:
                mutation_rate = 0.35 * (1.0 - epoch / config.n_epochs)  # Tăng mutation rate
                if rng.random() < mutation_rate:
                    mutation = rng.normal(0, 0.25, dim)  # Tăng mutation strength
                    pop[i] = np.clip(pop[i] + mutation, -1, 1)
                    
                    # Enforce feature count constraint sau mutation
                    if config.feature_selection_mode == "fixed_count":
                        n_feat = dim - 13
                        target_count = config.target_feature_count
                        feat_part = pop[i, :n_feat]
                        
                        # Chọn top target_count features
                        top_indices = np.argsort(feat_part)[-target_count:]
                        feat_part[:] = rng.uniform(-2.0, -0.5, n_feat)
                        feat_part[top_indices] = rng.uniform(0.5, 2.0, len(top_indices))
                        pop[i, :n_feat] = feat_part
        
        fits = np.array([fitness(p) for p in pop])
        current_best = max(fits)
        
        # Improved convergence tracking với threshold nhỏ hơn
        if current_best > best_fit_history[-1] + 1e-7:  # Giảm minimum improvement threshold
            best_fit_history.append(current_best)
            best_pop = pop.copy()
            stagnation_count = 0
        else:
            best_fit_history.append(best_fit_history[-1])
            stagnation_count += 1
            
        print(f"[INFO] Epoch {epoch + 1} took {time.time() - start_epoch:.2f}s, best fitness: {current_best:.5f}, stagnation: {stagnation_count}")
        
        # Early stopping với enhanced patience
        if stagnation_count >= config.patience:
            print(f"Early stopping FA at epoch {epoch + 1} due to stagnation, best fitness: {best_fit_history[-1]:.5f}")
            break

    best_idx = int(np.argmax(fits))
    best_pos = best_pop[best_idx]
    best_fit = float(best_fit_history[-1])
    best_mask, best_hp, best_rcfg = decode_position(best_pos, n_features, config)
    
    print(f"[INFO] FA optimization completed:")
    print(f"  - Mode: {config.feature_selection_mode}")
    print(f"  - Selected features: {best_mask.sum()}/{n_features}")
    print(f"  - Best fitness: {best_fit:.5f}")
    
    return best_mask, best_hp, best_rcfg, best_fit

# -----------------------------
# Overfitting prevention helpers - Enhanced validation
# -----------------------------
def robust_cv_with_resampling(X: pd.DataFrame, y: pd.Series, params: Dict, num_boost_round: int = 800, early_stopping_rounds: int = 50, folds: int = 5) -> Tuple[int, Dict, Dict]:
    """
    Robust CV với resampling riêng biệt cho từng fold để tránh data leakage
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import precision_score, recall_score
    
    tscv = TimeSeriesSplit(n_splits=folds, test_size=len(X)//folds)
    scores = []
    detailed_results = {'fold_scores': [], 'fold_details': []}
    
    print(f"[INFO] Starting robust CV with {folds} folds...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"[INFO] Processing fold {fold_idx + 1}/{folds}")
        
        X_train_fold = X.iloc[train_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()
        X_val_fold = X.iloc[val_idx].copy()
        y_val_fold = y.iloc[val_idx].copy()
        
        # Original fraud rates
        train_fraud_rate = y_train_fold.mean()
        val_fraud_rate = y_val_fold.mean()
        
        # Resampling CHỈ trên train fold để tránh leakage
        X_train_resampled, y_train_resampled = _apply_resampling(
            X_train_fold, y_train_fold, 
            method=1,  # SMOTE only
            ratio=0.15, # Conservative ratio
            k_neighbors=5
        )
        
        resampled_fraud_rate = y_train_resampled.mean()
        
        # Train model trên resampled data
        model = XGBClassifier(
            **params,
            n_estimators=200,  # Conservative for CV
            random_state=42 + fold_idx,  # Different seed per fold
            eval_metric="aucpr",
        )
        
        try:
            model.fit(
                X_train_resampled, y_train_resampled,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False,
                early_stopping_rounds=early_stopping_rounds
            )
            
            # Predict trên ORIGINAL validation data (không resampled)
            proba_val = model.predict_proba(X_val_fold)[:, 1]
            proba_train_original = model.predict_proba(X_train_fold)[:, 1]  # Original train data
            
            # Metrics trên original data distributions
            val_ap = float(average_precision_score(y_val_fold, proba_val))
            train_ap_original = float(average_precision_score(y_train_fold, proba_train_original))
            
            # Reality check metrics
            top_1_percent = int(np.ceil(0.01 * len(y_val_fold)))
            top_indices = np.argsort(-proba_val)[:top_1_percent]
            precision_at_1_percent = float(y_val_fold.iloc[top_indices].mean()) if top_1_percent > 0 else 0.0
            
            scores.append(val_ap)
            
            fold_detail = {
                'fold': fold_idx + 1,
                'train_fraud_rate_original': float(train_fraud_rate),
                'train_fraud_rate_resampled': float(resampled_fraud_rate),
                'val_fraud_rate': float(val_fraud_rate),
                'val_ap': val_ap,
                'train_ap_original': train_ap_original,
                'overfitting_gap': max(0, train_ap_original - val_ap),
                'precision_at_1_percent': precision_at_1_percent,
                'n_estimators_used': model.best_iteration or 200
            }
            detailed_results['fold_details'].append(fold_detail)
            
            print(f"  Fold {fold_idx + 1}: Val AP={val_ap:.5f}, Train AP={train_ap_original:.5f}, Gap={fold_detail['overfitting_gap']:.5f}")
            
        except Exception as e:
            print(f"[WARN] Fold {fold_idx + 1} failed: {e}")
            scores.append(0.0)
            detailed_results['fold_details'].append({'fold': fold_idx + 1, 'error': str(e)})
    
    # Aggregate results
    detailed_results['fold_scores'] = scores
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    
    # Calculate recommended n_estimators dựa trên average của các folds
    valid_estimators = [d.get('n_estimators_used', 200) for d in detailed_results['fold_details'] if 'error' not in d]
    recommended_n_estimators = int(np.mean(valid_estimators)) if valid_estimators else 200
    
    summary = {
        'mean_cv_score': mean_score,
        'std_cv_score': std_score,
        'recommended_n_estimators': recommended_n_estimators,
        'mean_overfitting_gap': float(np.mean([d.get('overfitting_gap', 0) for d in detailed_results['fold_details'] if 'error' not in d])),
        'mean_precision_at_1_percent': float(np.mean([d.get('precision_at_1_percent', 0) for d in detailed_results['fold_details'] if 'error' not in d])),
    }
    
    print(f"[INFO] Robust CV completed:")
    print(f"  Mean CV score: {mean_score:.5f} ± {std_score:.5f}")
    print(f"  Recommended n_estimators: {recommended_n_estimators}")
    print(f"  Mean overfitting gap: {summary['mean_overfitting_gap']:.5f}")
    print(f"  Mean precision@1%: {summary['mean_precision_at_1_percent']:.5f}")
    
    return recommended_n_estimators, detailed_results, summary

def xgb_find_best_nrounds(X: pd.DataFrame, y: pd.Series, params: Dict, num_boost_round: int = 1000, early_stopping_rounds: int = 50, folds: int = 5):
    """Legacy wrapper - giữ để backward compatibility"""
    try:
        # Sử dụng robust CV method
        best_n, detailed_results, summary = robust_cv_with_resampling(
            X, y, params, num_boost_round, early_stopping_rounds, folds
        )
        
        # Create compatible return format
        mock_cvres = pd.DataFrame({
            'test-aucpr-mean': [summary['mean_cv_score']] * best_n,
            'train-aucpr-mean': [summary['mean_cv_score'] + summary['mean_overfitting_gap']] * best_n
        })
        
        return best_n, mock_cvres
        
    except Exception as e:
        print(f"[WARN] Robust CV failed: {e}. Falling back to simple method.")
        # Fallback to original method
        dtrain = DMatrix(X, label=y)
        xgb_params = params.copy()
        if "objective" not in xgb_params:
            xgb_params["objective"] = "binary:logistic"
        
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=folds, test_size=len(X)//folds)
        
        res = xgb_cv(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            folds=tscv.split(X, y),
            metrics=("aucpr", "logloss"),
            early_stopping_rounds=early_stopping_rounds,
            seed=42,
            as_pandas=True,
            verbose_eval=False,
        )
        best_n = int(len(res))
        return best_n, res

def reality_check_validation(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "Model") -> Dict:
    """
    Reality check validation trên pure original data để detect performance "ảo"
    """
    print(f"[INFO] Running reality check for {model_name}...")
    
    proba = model.predict_proba(X_test)[:, 1]
    
    # Original data statistics
    original_fraud_rate = float(y_test.mean())
    n_fraud = int(y_test.sum())
    n_total = len(y_test)
    
    print(f"  Original fraud rate: {original_fraud_rate:.4f} ({n_fraud}/{n_total})")
    
    # Overall metrics
    pr_auc = float(average_precision_score(y_test, proba))
    roc_auc = float(roc_auc_score(y_test, proba))
    
    # Precision at different percentiles (realistic business scenarios)
    reality_metrics = {
        'original_fraud_rate': original_fraud_rate,
        'n_fraud_cases': n_fraud,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'percentile_analysis': {}
    }
    
    for percentile in [99.9, 99.5, 99, 98, 95]:
        try:
            threshold = np.percentile(proba, percentile)
            predicted_positive = (proba >= threshold)
            n_predicted = int(predicted_positive.sum())
            
            if n_predicted > 0:
                precision = float(precision_score(y_test, predicted_positive, zero_division=0))
                recall = float(recall_score(y_test, predicted_positive, zero_division=0))
                n_true_positive = int(((y_test == 1) & predicted_positive).sum())
            else:
                precision = recall = n_true_positive = 0
            
            percentile_info = {
                'threshold': float(threshold),
                'n_predicted': n_predicted,
                'n_true_positive': n_true_positive,
                'precision': precision,
                'recall': recall,
                'review_rate': float(n_predicted / n_total)  # Percentage need to review
            }
            
            reality_metrics['percentile_analysis'][f'top_{100-percentile}%'] = percentile_info
            
            print(f"  Top {100-percentile:>4}%: {n_predicted:>5} pred, {n_true_positive:>3} TP, P={precision:.3f}, R={recall:.3f}, Review={percentile_info['review_rate']:.3f}")
            
        except Exception as e:
            print(f"  Error at percentile {percentile}: {e}")
    
    # Business-realistic thresholds
    business_thresholds = [0.1, 0.2, 0.3, 0.5]  # More realistic than percentile-based
    reality_metrics['business_thresholds'] = {}
    
    print(f"  Business threshold analysis:")
    for thresh in business_thresholds:
        predicted_positive = (proba >= thresh)
        n_predicted = int(predicted_positive.sum())
        
        if n_predicted > 0:
            precision = float(precision_score(y_test, predicted_positive, zero_division=0))
            recall = float(recall_score(y_test, predicted_positive, zero_division=0))
            n_true_positive = int(((y_test == 1) & predicted_positive).sum())
        else:
            precision = recall = n_true_positive = 0
        
        business_info = {
            'n_predicted': n_predicted,
            'n_true_positive': n_true_positive,
            'precision': precision,
            'recall': recall,
            'review_rate': float(n_predicted / n_total)
        }
        
        reality_metrics['business_thresholds'][f'thresh_{thresh}'] = business_info
        print(f"    Thresh {thresh}: {n_predicted:>5} pred, {n_true_positive:>3} TP, P={precision:.3f}, R={recall:.3f}")
    
    # Distribution analysis
    fraud_scores = proba[y_test == 1]
    normal_scores = proba[y_test == 0] 
    
    reality_metrics['score_distribution'] = {
        'fraud_mean': float(fraud_scores.mean()) if len(fraud_scores) > 0 else 0,
        'fraud_std': float(fraud_scores.std()) if len(fraud_scores) > 0 else 0,
        'fraud_median': float(np.median(fraud_scores)) if len(fraud_scores) > 0 else 0,
        'normal_mean': float(normal_scores.mean()),
        'normal_std': float(normal_scores.std()),
        'normal_median': float(np.median(normal_scores)),
    }
    
    separation = reality_metrics['score_distribution']['fraud_mean'] - reality_metrics['score_distribution']['normal_mean']
    print(f"  Score separation (fraud_mean - normal_mean): {separation:.4f}")
    
    # Warning flags for "ảo" performance
    warnings = []
    if pr_auc > 0.9:
        warnings.append("Unusually high PR-AUC (>0.9) - check for data leakage")
    if reality_metrics['percentile_analysis'].get('top_1%', {}).get('precision', 0) > 0.5:
        warnings.append("Very high precision at top 1% - may be too optimistic") 
    if separation < 0.1:
        warnings.append("Low score separation between fraud/normal - weak discrimination")
    
    reality_metrics['warnings'] = warnings
    
    if warnings:
        print(f"  ⚠️  WARNINGS:")
        for warning in warnings:
            print(f"    - {warning}")
    else:
        print(f"  ✅ No obvious signs of performance inflation")
    
    return reality_metrics

# -----------------------------
# Ensemble Methods để giảm overfitting
# -----------------------------
class EnsembleModel:
    def __init__(self, base_models: List, meta_model=None, use_stacking: bool = True):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(C=0.05, random_state=42)  # Tăng regularization
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

def create_diverse_models(best_hp: Dict, spw: float, n_estimators: int = 150) -> List:
    """Tạo diverse base models để ensemble với regularization mạnh hơn"""
    models = []
    
    # Model 1: Ultra Conservative XGBoost (chống overfitting cực mạnh)
    conservative_params = best_hp.copy()
    conservative_params.update({
        'n_estimators': n_estimators,
        'learning_rate': conservative_params.get('eta', 0.02) * 0.7,  # Rất chậm learning
        'max_depth': min(3, conservative_params.get('max_depth', 4)),  # Rất shallow trees
        'reg_alpha': conservative_params.get('reg_alpha', 2.0) * 2.5,  # Rất mạnh L1 reg
        'reg_lambda': conservative_params.get('reg_lambda', 5.0) * 2.0,  # Rất mạnh L2 reg
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'min_child_weight': 8,
        'gamma': 0.3,
    })
    models.append(XGBClassifier(
        objective="binary:logistic", tree_method="gpu_hist", n_gpus=2,
        scale_pos_weight=spw, random_state=42, **conservative_params
    ))
    
    # Model 2: Balanced XGBoost với strong regularization
    balanced_params = best_hp.copy()
    balanced_params.update({
        'n_estimators': n_estimators,
        'learning_rate': conservative_params.get('eta', 0.02),
        'max_depth': conservative_params.get('max_depth', 4),
        'reg_alpha': conservative_params.get('reg_alpha', 2.0) * 1.5,
        'reg_lambda': conservative_params.get('reg_lambda', 5.0) * 1.3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 6,
        'gamma': 0.2,
    })
    models.append(XGBClassifier(
        objective="binary:logistic", tree_method="gpu_hist", n_gpus=2,
        scale_pos_weight=spw, random_state=43, **balanced_params
    ))
    
    # Model 3: Heavily Regularized Logistic Regression
    models.append(LogisticRegression(
        C=0.005,  # Rất mạnh regularization
        penalty='elasticnet',
        l1_ratio=0.7,  # Nhiều L1 hơn để feature selection
        solver='saga',
        max_iter=2000,
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
    fa_config_path = os.path.join(out_dir, "fa_optimized_config_v2.json")
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
        ratio=float(best_rcfg.get("ratio", 0.15)),  # Giảm ratio mặc định
        k_neighbors=int(best_rcfg.get("k_neighbors", 5)),
    )
    if best_rcfg.get("method", 0) != 0 and HAS_IMBLEARN:
        spw_final = 1.0
    else:
        pos = int(pd.Series(y_tr_use).sum()); neg = len(y_tr_use) - pos
        spw_final = max(1.0, neg / max(1, pos))
    base_params = {
        "objective": "binary:logistic",
        "learning_rate": best_hp.get("eta", 0.02),
        "max_depth": int(best_hp.get("max_depth", 4)),
        "subsample": best_hp.get("subsample", 0.7),
        "colsample_bytree": best_hp.get("colsample_bytree", 0.7),
        "reg_lambda": best_hp.get("reg_lambda", 5.0),
        "reg_alpha": best_hp.get("reg_alpha", 2.0),
        "min_child_weight": best_hp.get("min_child_weight", 5.0),
        "gamma": best_hp.get("gamma", 0.2),
        "scale_pos_weight": spw_final,
        "tree_method": "gpu_hist",
        "n_gpus": 2,
        "eval_metric": ["aucpr", "logloss"],
    }
    try:
        # Sử dụng early stopping patience ~50 rounds
        best_nrounds, cvres = xgb_find_best_nrounds(X_tr_use, y_tr_use, base_params, num_boost_round=800, early_stopping_rounds=50, folds=5)
        print(f"[INFO] XGB CV chose {best_nrounds} rounds (approx).")
        
        # Nested CV để đánh giá model performance không bias
        
        # Robust CV validation thay cho nested CV 
        print("[INFO] Running robust cross-validation...")
        robust_cv_results = robust_cv_with_resampling(
            X_tr_use, y_tr_use, base_params, 
            n_folds=3, resampling_ratio=float(best_rcfg.get("ratio", 0.15)),
            k_neighbors=int(best_rcfg.get("k_neighbors", 5))
        )
        print(f"  Robust CV PR-AUC: {robust_cv_results['mean_pr_auc']:.4f} ± {robust_cv_results['std_pr_auc']:.4f}")
        print(f"  Original data PR-AUC: {robust_cv_results['mean_original_pr_auc']:.4f}")
        print(f"  Recommended n_estimators: {robust_cv_results['recommended_n_estimators']}")
        
        # Update params với recommended n_estimators
        if robust_cv_results['recommended_n_estimators'] > 0:
            base_params['n_estimators'] = robust_cv_results['recommended_n_estimators']
            print(f"[INFO] Updated n_estimators to {base_params['n_estimators']} based on robust CV")
        
        nested_score = robust_cv_results['mean_pr_auc']
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
                plt.axhline(y=0.03, color='orange', linestyle='--', label='overfitting threshold (3%)')
                plt.xlabel("boosting round"); plt.ylabel("performance gap"); plt.legend(); plt.grid(True)
                plt.title("Overfitting Detection")
                
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "xgb_cv_analysis_v2.png"), dpi=150)
                plt.close()
        except Exception as e:
            print(f"[WARN] failed to plot xgb cv: {e}")
    except Exception as e:
        print(f"[WARN] xgb cv failed: {e}; falling back to default n_estimators 400")
        best_nrounds = 400

    # Train both single model và ensemble với enhanced regularization
    print("[INFO] Training single XGBoost model with enhanced regularization...")
    clf = XGBClassifier(
        objective="binary:logistic",
        tree_method="gpu_hist",
        n_gpus=2,  
        n_estimators=max(50, best_nrounds),
        learning_rate=base_params["learning_rate"],
        max_depth=base_params["max_depth"],
        subsample=base_params["subsample"],
        colsample_bytree=base_params["colsample_bytree"],
        reg_alpha=base_params["reg_alpha"],
        reg_lambda=base_params["reg_lambda"],
        min_child_weight=base_params["min_child_weight"],
        gamma=base_params["gamma"],
        scale_pos_weight=base_params["scale_pos_weight"],
        eval_metric=["aucpr", "logloss"],
        n_jobs=-1,
        random_state=seed,
        enable_categorical=True,
    )
    evals_result = {}
    model_checkpoint_path = os.path.join(out_dir, "model_checkpoint_v2.pkl")
    gpu_success = False
    
    try:
        print("[INFO] Training final XGBoost model with GPU and enhanced early stopping...")
        clf.fit(
            X_tr_use, y_tr_use,
            eval_set=[(X_va[cols], y_va)],
            verbose=False,
            early_stopping_rounds=50,  # Patience ~50 rounds monitoring PR-AUC
        )
        evals_result = clf.evals_result()
        gpu_success = True
        joblib.dump(clf, model_checkpoint_path)
        print(f"[INFO] Trained and saved model checkpoint (GPU) to {model_checkpoint_path}")
    except Exception as e:
        print(f"[WARN] GPU training failed: {e}. Falling back to CPU.")
        clf.tree_method = "hist"
        clf.fit(X_tr_use, y_tr_use, eval_set=[(X_va[cols], y_va)], verbose=False, early_stopping_rounds=50)
        evals_result = clf.evals_result()
        joblib.dump(clf, model_checkpoint_path)
        print(f"[INFO] Trained and saved model checkpoint (CPU fallback) to {model_checkpoint_path}")

    # Train ensemble model với enhanced regularization
    print("[INFO] Training ensemble model with enhanced regularization...")
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
            ensemble_path = os.path.join(out_dir, "ensemble_model_v2.pkl") 
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
        plot_feature_importance(clf, list(cols), out_dir)
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
    feature_mode: str = "fixed_count",  # "flexible" hoặc "fixed_count"
    target_features: int = 14,          # Số features mục tiêu
):
    """
    Run experiments với 2 modes:
    - flexible: FA tự do chọn features (có thể ít hơn target_features)  
    - fixed_count: FA phải chọn đúng target_features features
    """
    features = features or DEFAULT_FEATURES
    X, y = load_dataset(data_path, features)

    # Tạo FA config dựa trên mode
    fa_cfg = FAConfig(
        random_state=seed,
        feature_selection_mode=feature_mode,
        target_feature_count=target_features,
        min_feature_count=max(8, target_features // 2),  # Tối thiểu = nửa target
    )
    
    print(f"[INFO] Feature selection mode: {feature_mode}")
    if feature_mode == "fixed_count":
        print(f"[INFO] Target features: {target_features} (cố định)")
    else:
        print(f"[INFO] Target features: {target_features} (linh hoạt, tối thiểu {fa_cfg.min_feature_count})")

    scaler = StandardScaler()
    X_tr, X_va, X_te, y_tr, y_va, y_te = chronological_train_val_test_split(
        X, y, time_col="Time", embargo=embargo
    )

    results = {}
    start = time.time()
    out = train_fa_xgb(X_tr.copy(), y_tr.copy(), X_va.copy(), y_va.copy(), X_te.copy(), y_te.copy(), seed, fa_cfg=fa_cfg, out_dir=out_dir)
    duration = round(time.time() - start, 2)
    res = {"metrics": out["metrics"], "val_pr_auc": out.get("val_pr_auc", None), "time_sec": duration}
    results[f"fa_xgb_v2_{feature_mode}"] = res
    print(f"Scenario fa_xgb_v2_{feature_mode} done in {duration}s | Test PR-AUC: {res['metrics']['pr_auc']:.5f} | ROC-AUC: {res['metrics']['roc_auc']:.5f}")

    ensure_outputs_dir(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(out_dir, f"fraud_v2_results_{feature_mode}_{ts}.json")
    data_to_dump = convert_to_serializable({
        "data_path": data_path,
        "features": X.columns.tolist(),
        "embargo": embargo,
        "seed": seed,
        "feature_mode": feature_mode,
        "target_features": target_features,
        "results": results,
    })
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data_to_dump, f, indent=2)
    print(f"Saved results to {out_file}")

    model_data = {
        "model": out["model"],
        "scaler": scaler
    }
    joblib.dump(model_data, os.path.join(out_dir, f'fraud_model_v2_{feature_mode}.pkl'))
    print(f"Saved model and scaler to fraud_model_v2_{feature_mode}.pkl")

    try:
        config_data = {
            "metrics": convert_to_serializable(out["metrics"]),
            "selected_features": out["metrics"]["selected_features"],
            "chosen_threshold": out["metrics"]["chosen_threshold"],
            "feature_mode": feature_mode,
            "n_selected_features": len(out["metrics"]["selected_features"])
        }
        with open(os.path.join(out_dir, f'fraud_config_v2_{feature_mode}.json'), 'w', encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        print(f"Saved config to fraud_config_v2_{feature_mode}.json")
    except Exception as e:
        print(f"Error saving fraud_config_v2_{feature_mode}.json: {e}")

def parse_args():
    p = argparse.ArgumentParser(description="Fraud detection experiments v2: Enhanced FA + XGBoost with stronger anti-overfitting")
    p.add_argument("--data", type=str, default="/kaggle/input/creditcardfraud/creditcard.csv", help="Path to Kaggle Credit Card Fraud dataset CSV")
    p.add_argument("--embargo", type=int, default=0, help="Embargo samples between splits to avoid leakage")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="outputs", help="Output directory for plots and artifacts")
    p.add_argument("--feature_mode", type=str, default="fixed_count", choices=["flexible", "fixed_count"], 
                   help="Feature selection mode: 'flexible' (tự do chọn) hoặc 'fixed_count' (cố định 14)")
    p.add_argument("--target_features", type=int, default=14, help="Số features mục tiêu")
    args = p.parse_args([])
    return args

if __name__ == "__main__":
    args = parse_args()
    try:
        # Chạy cả 2 modes để so sánh
        print("="*60)
        print("CHẠY MODE 1: CỐ ĐỊNH 14 FEATURES")
        print("="*60)
        run_experiments(
            data_path=args.data,
            seed=args.seed,
            embargo=args.embargo,
            out_dir=args.out,
            feature_mode="fixed_count",
            target_features=14,
        )
        
        # print("\n" + "="*60)
        # print("CHẠY MODE 2: FLEXIBLE FEATURE SELECTION")  
        # print("="*60)
        # run_experiments(
        #     data_path=args.data,
        #     seed=args.seed,
        #     embargo=args.embargo,
        #     out_dir=args.out,
        #     feature_mode="flexible",
        #     target_features=14,
        # )
        
        print("\n" + "="*60)
        print("HOÀN THÀNH! Kiểm tra thư mục outputs để so sánh kết quả.")
        print("="*60)
        
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Ensure dataset path is correct and required packages are installed: xgboost scikit-learn==1.2.2 imbalanced-learn matplotlib joblib scipy==1.13.1")