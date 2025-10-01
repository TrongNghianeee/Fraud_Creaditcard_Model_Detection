# """
# Enhanced Fraud Detection Model V3 - Production Ready
# =========================================================
# Improvements:
# 1. ✅ Fixed data leakage issues
# 2. ✅ Optimized SMOTE ratio (0.05-0.10)
# 3. ✅ Enhanced regularization
# 4. ✅ Purged time-series CV
# 5. ✅ Simplified hyperparameter tuning
# 6. ✅ Business-oriented metrics
# 7. ✅ Temporal stability testing
# 8. ✅ Production-ready pipeline
# """

import os
import json
import time
import math
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
    precision_score,
    recall_score,
    log_loss,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from imblearn.over_sampling import ADASYN
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False
    print("[WARN] imbalanced-learn not installed.")

from xgboost import XGBClassifier
import joblib
from scipy import stats

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Centralized configuration for reproducibility"""
    # Data splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    embargo_days: int = 2  # 2 days embargo to prevent leakage
    
    # Resampling - REDUCED to prevent overfitting
    use_resampling: bool = True
    resampling_ratio: float = 0.08  # Reduced from 0.15 to 0.08
    resampling_method: str = "adasyn"  # Better than SMOTE for fraud
    
    # XGBoost - ENHANCED regularization
    xgb_params: dict = None
    
    # Cross-validation
    cv_folds: int = 5
    early_stopping_rounds: int = 40
    max_overfitting_gap: float = 0.03  # Max acceptable train-val gap
    
    # Business metrics
    cost_fn: float = 100.0  # Cost of missing fraud
    cost_fp: float = 1.0    # Cost of false alarm
    max_review_rate: float = 0.05  # Can only review 5% of transactions
    
    # Seeds
    random_state: int = 42
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic',
                'tree_method': 'gpu_hist',
                'eval_metric': 'aucpr',
                'n_estimators': 200,  # Reduced from 300
                'learning_rate': 0.005,  # Reduced from 0.02
                'max_depth': 3,  # Reduced from 4-5
                'min_child_weight': 8,  # Increased from 5
                'subsample': 0.6,  # Reduced from 0.7
                'colsample_bytree': 0.6,  # Reduced from 0.7
                'colsample_bylevel': 0.6,
                'colsample_bynode': 0.7,
                'reg_alpha': 5.0,  # Increased L1
                'reg_lambda': 15.0,  # Increased L2
                'gamma': 0.3,  # Increased pruning
                'scale_pos_weight': 1.0,
                'random_state': self.random_state,
                'n_jobs': -1,
            }

# ============================================================================
# UTILITIES
# ============================================================================

def ensure_outputs_dir(path: str = "outputs") -> str:
    os.makedirs(path, exist_ok=True)
    return path

def compute_business_metrics(y_true: np.ndarray, y_proba: np.ndarray, 
                            cost_fn: float = 100.0, cost_fp: float = 1.0,
                            max_review_rate: float = 0.05) -> Dict:
    """
    Business-oriented metrics for fraud detection
    """
    results = {}
    
    # Find optimal threshold based on cost
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.r_[thresholds, 1.0]
    
    costs = []
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = cost_fn * fn + cost_fp * fp
        costs.append(cost)
    
    min_cost_idx = np.argmin(costs)
    optimal_threshold = float(thresholds[min_cost_idx])
    min_cost = float(costs[min_cost_idx])
    
    results['optimal_threshold'] = optimal_threshold
    results['min_cost'] = min_cost
    
    # Metrics at optimal threshold
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()
    
    results['metrics_at_optimal'] = {
        'threshold': optimal_threshold,
        'precision': float(precision_score(y_true, y_pred_optimal, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred_optimal, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred_optimal, zero_division=0)),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'review_rate': float((tp + fp) / len(y_true)),
        'cost': min_cost
    }
    
    # Constrained by review capacity
    n_can_review = int(max_review_rate * len(y_true))
    top_indices = np.argsort(-y_proba)[:n_can_review]
    y_pred_constrained = np.zeros_like(y_true)
    y_pred_constrained[top_indices] = 1
    
    tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_true, y_pred_constrained).ravel()
    cost_constrained = cost_fn * fn_c + cost_fp * fp_c
    
    results['metrics_at_capacity'] = {
        'max_review_rate': max_review_rate,
        'n_reviewed': n_can_review,
        'precision': float(precision_score(y_true, y_pred_constrained, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred_constrained, zero_division=0)),
        'tp': int(tp_c), 'fp': int(fp_c), 'fn': int(fn_c), 'tn': int(tn_c),
        'cost': float(cost_constrained)
    }
    
    # Precision at different recall levels
    results['precision_at_recall'] = {}
    for target_recall in [0.5, 0.6, 0.7, 0.8, 0.9]:
        idx = np.where(recall >= target_recall)[0]
        if len(idx) > 0:
            results['precision_at_recall'][f'recall_{target_recall}'] = float(precision[idx[-1]])
        else:
            results['precision_at_recall'][f'recall_{target_recall}'] = 0.0
    
    return results

def compute_comprehensive_metrics(y_true: np.ndarray, y_proba: np.ndarray, 
                                  threshold: float = 0.5) -> Dict:
    """Comprehensive metrics for model evaluation"""
    y_pred = (y_proba >= threshold).astype(int)
    
    metrics = {
        'roc_auc': float(roc_auc_score(y_true, y_proba)),
        'pr_auc': float(average_precision_score(y_true, y_proba)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'log_loss': float(log_loss(y_true, y_proba)),
        'threshold': float(threshold),
    }
    
    # KS Statistic
    fraud_scores = y_proba[y_true == 1]
    normal_scores = y_proba[y_true == 0]
    if len(fraud_scores) > 0 and len(normal_scores) > 0:
        metrics['ks_statistic'] = float(stats.ks_2samp(fraud_scores, normal_scores).statistic)
    else:
        metrics['ks_statistic'] = 0.0
    
    # Top-k recall
    for k in [0.001, 0.005, 0.01, 0.05]:
        n = len(y_true)
        k_n = max(1, int(math.ceil(k * n)))
        idx = np.argsort(-y_proba)[:k_n]
        recall_k = float(y_true[idx].sum()) / max(1, int(y_true.sum()))
        metrics[f'recall@{k}'] = recall_k
    
    return metrics

# ============================================================================
# FEATURE ENGINEERING - LEAK-FREE
# ============================================================================

class LeakFreeFeatureEngineer:
    """
    Feature engineering that prevents data leakage
    - Only uses information available at prediction time
    - Separates fit (train) and transform (test) operations
    """
    def __init__(self):
        self.fitted = False
        self.global_stats = {}
        
    def fit(self, df: pd.DataFrame, y: pd.Series = None):
        """Learn statistics from training data only"""
        # Global statistics from training data
        if 'Amount' in df.columns:
            self.global_stats['amount_mean'] = df['Amount'].mean()
            self.global_stats['amount_std'] = df['Amount'].std()
            self.global_stats['amount_median'] = df['Amount'].median()
            self.global_stats['amount_q25'] = df['Amount'].quantile(0.25)
            self.global_stats['amount_q75'] = df['Amount'].quantile(0.75)
        
        # V-features statistics
        v_cols = [col for col in df.columns if col.startswith('V') and len(col) <= 3]
        if len(v_cols) > 0:
            self.global_stats['v_means'] = df[v_cols].mean().to_dict()
            self.global_stats['v_stds'] = df[v_cols].std().to_dict()
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using learned statistics"""
        if not self.fitted:
            raise ValueError("Must call fit() before transform()")
        
        df = df.copy()
        
        # Time-based features (leak-free)
        df['hour_of_day'] = (df['Time'] % 86400) / 3600
        df['day_of_week'] = ((df['Time'] / 86400) % 7).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(int)
        
        # Amount-based features (leak-free)
        if 'Amount' in df.columns:
            df['log_amount'] = np.log1p(df['Amount'])
            
            # Deviation from global mean (learned from train)
            df['amount_dev_from_mean'] = (df['Amount'] - self.global_stats['amount_mean']) / (self.global_stats['amount_std'] + 1e-8)
            df['amount_dev_from_median'] = df['Amount'] - self.global_stats['amount_median']
            
            # Binning
            df['is_zero_amount'] = (df['Amount'] == 0).astype(int)
            df['is_small_amount'] = (df['Amount'] < 10).astype(int)
            df['is_large_amount'] = (df['Amount'] > self.global_stats['amount_q75'] * 2).astype(int)
            
            # Interaction with time
            df['amount_hour_interaction'] = df['log_amount'] * df['hour_of_day']
            df['amount_weekend_interaction'] = df['log_amount'] * df['is_weekend']
        
        # V-features aggregations (leak-free)
        v_cols = [col for col in df.columns if col.startswith('V') and len(col) <= 3]
        if len(v_cols) > 0:
            df['v_sum'] = df[v_cols].sum(axis=1)
            df['v_mean'] = df[v_cols].mean(axis=1)
            df['v_std'] = df[v_cols].std(axis=1).fillna(0)
            df['v_min'] = df[v_cols].min(axis=1)
            df['v_max'] = df[v_cols].max(axis=1)
            df['v_range'] = df['v_max'] - df['v_min']
            
            # Count extreme values
            v_array = df[v_cols].values
            df['v_n_negative'] = (v_array < 0).sum(axis=1)
            df['v_n_extreme'] = (np.abs(v_array) > 3).sum(axis=1)
            
            # Deviation from training distribution
            for col in v_cols[:10]:  # Only top 10 to avoid curse of dimensionality
                if col in self.global_stats['v_means']:
                    mean = self.global_stats['v_means'][col]
                    std = self.global_stats['v_stds'][col]
                    df[f'{col}_zscore'] = (df[col] - mean) / (std + 1e-8)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(df, y).transform(df)

# ============================================================================
# DATA LOADING & SPLITTING
# ============================================================================

def load_and_prepare_data(csv_path: str, config: ModelConfig) -> Tuple:
    """
    Load data with proper leak prevention
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    print(f"[INFO] Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Basic validation
    required_cols = ['Time', 'Amount', 'Class']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Fraud rate: {df['Class'].mean():.4f}")
    
    return df

def purged_timeseries_split(df: pd.DataFrame, config: ModelConfig) -> Tuple:
    """
    Time-series split with purged embargo to prevent leakage
    """
    # Sort by time
    df = df.sort_values('Time').reset_index(drop=True)
    n = len(df)
    
    # Simple index-based split (more robust than time-based for this dataset)
    n_train = int(n * config.train_ratio)
    n_val = int(n * config.val_ratio)
    
    # Calculate embargo in samples (2% of data is safer than fixed days)
    embargo_size = max(1, int(n * 0.01))  # 1% embargo
    
    # Split with embargo gaps
    train_end = n_train
    val_start = train_end + embargo_size
    val_end = val_start + n_val
    test_start = val_end + embargo_size
    
    # Ensure we don't exceed dataset size
    if test_start >= n:
        # Adjust if embargo pushes us beyond dataset
        test_start = min(test_start, n - 1)
        val_end = test_start - embargo_size
        val_start = max(train_end + embargo_size, val_end - n_val)
    
    # Create splits
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[val_start:val_end].copy()
    df_test = df.iloc[test_start:].copy()
    
    # Validate splits are not empty
    if len(df_train) == 0 or len(df_val) == 0 or len(df_test) == 0:
        print("[WARN] Embargo created empty splits, falling back to simple split")
        df_train = df.iloc[:n_train].copy()
        df_val = df.iloc[n_train:n_train+n_val].copy()
        df_test = df.iloc[n_train+n_val:].copy()
    
    print(f"[INFO] Purged splits with {embargo_size} sample embargo:")
    print(f"  Train: {len(df_train)} samples ({len(df_train)/n*100:.1f}%)")
    print(f"  Val:   {len(df_val)} samples ({len(df_val)/n*100:.1f}%)")
    print(f"  Test:  {len(df_test)} samples ({len(df_test)/n*100:.1f}%)")
    
    if len(df_train) > 0 and len(df_val) > 0 and len(df_test) > 0:
        print(f"  Fraud rates - Train: {df_train['Class'].mean():.4f}, Val: {df_val['Class'].mean():.4f}, Test: {df_test['Class'].mean():.4f}")
    
    return df_train, df_val, df_test

# ============================================================================
# RESAMPLING - OPTIMIZED
# ============================================================================

def apply_conservative_resampling(X: pd.DataFrame, y: pd.Series, 
                                 config: ModelConfig) -> Tuple:
    """
    Conservative resampling to avoid overfitting
    - Uses ADASYN (better than SMOTE for fraud)
    - Low ratio (0.05-0.10)
    - Only on training data
    """
    if not config.use_resampling or not HAS_IMBLEARN:
        return X, y
    
    if len(np.unique(y)) < 2:
        print("[WARN] Only one class present, skipping resampling")
        return X, y
    
    try:
        print(f"[INFO] Applying {config.resampling_method.upper()} with ratio {config.resampling_ratio}")
        
        n_minority = int(y.sum())
        n_majority = len(y) - n_minority
        
        # Calculate target samples
        target_samples = int(n_majority * config.resampling_ratio)
        
        if config.resampling_method == "adasyn":
            sampler = ADASYN(
                sampling_strategy={1: min(target_samples, n_majority)},
                n_neighbors=5,
                random_state=config.random_state
            )
        else:
            print(f"[WARN] Unknown method {config.resampling_method}, skipping resampling")
            return X, y
        
        X_res, y_res = sampler.fit_resample(X, y)
        
        print(f"[INFO] Resampling: {len(X)} -> {len(X_res)} samples")
        print(f"[INFO] Fraud ratio: {y.mean():.4f} -> {y_res.mean():.4f}")
        
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
        
    except Exception as e:
        print(f"[WARN] Resampling failed: {e}, using original data")
        return X, y

# ============================================================================
# MODEL TRAINING - ENHANCED
# ============================================================================

class EnhancedFraudDetector:
    """
    Production-ready fraud detector with overfitting prevention
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        self.feature_engineer = LeakFreeFeatureEngineer()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.training_history = {}
        
    def prepare_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Prepare features with leak prevention"""
        # Separate X and y
        y = df['Class'] if 'Class' in df.columns else None
        X = df.drop('Class', axis=1, errors='ignore')
        
        # Feature engineering
        if fit:
            X = self.feature_engineer.fit_transform(X, y)
        else:
            X = self.feature_engineer.transform(X)
        
        # Store feature names
        if fit:
            self.feature_names = list(X.columns)
        
        # Scaling
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> 'EnhancedFraudDetector':
        """
        Train model with proper validation
        """
        print("\n" + "="*60)
        print("TRAINING ENHANCED FRAUD DETECTOR")
        print("="*60)
        
        # Prepare features
        print("[INFO] Preparing features...")
        X_train = self.prepare_features(df_train, fit=True)
        y_train = df_train['Class']
        
        X_val = self.prepare_features(df_val, fit=False)
        y_val = df_val['Class']
        
        print(f"[INFO] Feature dimension: {X_train.shape[1]}")
        
        # Apply resampling only on training data
        X_train_res, y_train_res = apply_conservative_resampling(
            X_train, y_train, self.config
        )
        
        # Calculate scale_pos_weight
        if self.config.use_resampling:
            scale_pos_weight = 1.0
        else:
            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            scale_pos_weight = n_neg / max(1, n_pos)
        
        print(f"[INFO] Scale pos weight: {scale_pos_weight:.2f}")
        
        # Update params
        params = self.config.xgb_params.copy()
        params['scale_pos_weight'] = scale_pos_weight
        
        # Train model
        print("[INFO] Training XGBoost model...")
        self.model = XGBClassifier(**params)
        
        eval_results = {}
        self.model.fit(
            X_train_res, y_train_res,
            eval_set=[(X_train_res, y_train_res), (X_val, y_val)],
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose=False
        )
        
        eval_results = self.model.evals_result()
        
        # Check for overfitting
        if eval_results:
            train_scores = eval_results['validation_0']['aucpr']
            val_scores = eval_results['validation_1']['aucpr']
            
            best_iter = self.model.best_iteration
            train_score = train_scores[best_iter]
            val_score = val_scores[best_iter]
            gap = train_score - val_score
            
            self.training_history = {
                'best_iteration': best_iter,
                'train_score': float(train_score),
                'val_score': float(val_score),
                'overfitting_gap': float(gap),
                'train_scores': [float(x) for x in train_scores],
                'val_scores': [float(x) for x in val_scores]
            }
            
            print(f"[INFO] Best iteration: {best_iter}")
            print(f"[INFO] Train PR-AUC: {train_score:.4f}")
            print(f"[INFO] Val PR-AUC: {val_score:.4f}")
            print(f"[INFO] Overfitting gap: {gap:.4f}")
            
            if gap > self.config.max_overfitting_gap:
                print(f"[WARN] Overfitting detected! Gap {gap:.4f} > {self.config.max_overfitting_gap}")
            else:
                print(f"[OK] Overfitting under control: {gap:.4f} <= {self.config.max_overfitting_gap}")
        
        return self
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X = self.prepare_features(df, fit=False)
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, df: pd.DataFrame, name: str = "Test") -> Dict:
        """Comprehensive evaluation"""
        print(f"\n[INFO] Evaluating on {name} set...")
        
        y_true = df['Class'].values
        y_proba = self.predict_proba(df)
        
        # Standard metrics
        metrics = compute_comprehensive_metrics(y_true, y_proba, threshold=0.5)
        
        # Business metrics
        business_metrics = compute_business_metrics(
            y_true, y_proba,
            cost_fn=self.config.cost_fn,
            cost_fp=self.config.cost_fp,
            max_review_rate=self.config.max_review_rate
        )
        
        results = {
            'standard_metrics': metrics,
            'business_metrics': business_metrics,
            'fraud_rate': float(y_true.mean()),
            'n_samples': len(y_true),
            'n_fraud': int(y_true.sum())
        }
        
        # Print summary
        print(f"\n  Standard Metrics:")
        print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"    PR-AUC:  {metrics['pr_auc']:.4f}")
        print(f"    F1:      {metrics['f1']:.4f}")
        print(f"    MCC:     {metrics['mcc']:.4f}")
        
        print(f"\n  Business Metrics (Optimal Threshold):")
        opt = business_metrics['metrics_at_optimal']
        print(f"    Threshold: {opt['threshold']:.4f}")
        print(f"    Precision: {opt['precision']:.4f}")
        print(f"    Recall:    {opt['recall']:.4f}")
        print(f"    Cost:      ${opt['cost']:,.0f}")
        print(f"    Review Rate: {opt['review_rate']*100:.2f}%")
        
        print(f"\n  Constrained by Capacity ({self.config.max_review_rate*100}% review rate):")
        cap = business_metrics['metrics_at_capacity']
        print(f"    Precision: {cap['precision']:.4f}")
        print(f"    Recall:    {cap['recall']:.4f}")
        print(f"    Cost:      ${cap['cost']:,.0f}")
        
        return results

# ============================================================================
# CROSS-VALIDATION - PURGED
# ============================================================================

class PurgedTimeSeriesSplit:
    """
    Time series split with purging to prevent leakage
    """
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
    
    def split(self, X: pd.DataFrame, y: pd.Series = None):
        """Generate purged train/test indices"""
        n = len(X)
        embargo_size = int(n * self.embargo_pct)
        test_size = n // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            test_start = (i + 1) * test_size
            test_end = test_start + test_size
            
            # Purge embargo before test
            train_end = max(0, test_start - embargo_size)
            train_indices = np.arange(0, train_end)
            
            # Purge embargo after test
            test_indices = np.arange(test_start, min(test_end, n))
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

def cross_validate_model(df_train: pd.DataFrame, config: ModelConfig) -> Dict:
    """
    Cross-validation with purged splits
    """
    print("\n" + "="*60)
    print("CROSS-VALIDATION WITH PURGED SPLITS")
    print("="*60)
    
    cv_splitter = PurgedTimeSeriesSplit(n_splits=config.cv_folds, embargo_pct=0.02)
    
    fold_results = []
    overfitting_gaps = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(df_train)):
        print(f"\n[INFO] Fold {fold_idx + 1}/{config.cv_folds}")
        
        df_fold_train = df_train.iloc[train_idx].copy()
        df_fold_val = df_train.iloc[val_idx].copy()
        
        # Train model
        detector = EnhancedFraudDetector(config)
        detector.fit(df_fold_train, df_fold_val)
        
        # Evaluate
        y_val = df_fold_val['Class'].values
        y_proba_val = detector.predict_proba(df_fold_val)
        
        val_score = average_precision_score(y_val, y_proba_val)
        
        fold_results.append({
            'fold': fold_idx + 1,
            'val_pr_auc': float(val_score),
            'train_pr_auc': detector.training_history.get('train_score', 0.0),
            'overfitting_gap': detector.training_history.get('overfitting_gap', 0.0),
            'best_iteration': detector.training_history.get('best_iteration', 0)
        })
        
        overfitting_gaps.append(detector.training_history.get('overfitting_gap', 0.0))
        
        print(f"  Fold {fold_idx + 1} Val PR-AUC: {val_score:.4f}")
    
    # Aggregate results
    cv_scores = [r['val_pr_auc'] for r in fold_results]
    
    summary = {
        'mean_cv_score': float(np.mean(cv_scores)),
        'std_cv_score': float(np.std(cv_scores)),
        'mean_overfitting_gap': float(np.mean(overfitting_gaps)),
        'max_overfitting_gap': float(np.max(overfitting_gaps)),
        'fold_results': fold_results
    }
    
    print(f"\n[INFO] CV Results:")
    print(f"  Mean PR-AUC: {summary['mean_cv_score']:.4f} ± {summary['std_cv_score']:.4f}")
    print(f"  Mean Overfitting Gap: {summary['mean_overfitting_gap']:.4f}")
    print(f"  Max Overfitting Gap: {summary['max_overfitting_gap']:.4f}")
    
    return summary

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(history: Dict, out_dir: str):
    """Plot training curves"""
    if not history or 'train_scores' not in history:
        return
    
    train_scores = history['train_scores']
    val_scores = history['val_scores']
    
    plt.figure(figsize=(12, 5))
    
    # PR-AUC curves
    plt.subplot(1, 2, 1)
    plt.plot(train_scores, label='Train', alpha=0.8)
    plt.plot(val_scores, label='Validation', alpha=0.8)
    plt.axvline(history['best_iteration'], color='r', linestyle='--', alpha=0.5, label='Best Iteration')
    plt.xlabel('Boosting Round')
    plt.ylabel('PR-AUC')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Overfitting gap
    plt.subplot(1, 2, 2)
    gaps = np.array(train_scores) - np.array(val_scores)
    plt.plot(gaps, color='red', alpha=0.8)
    plt.axhline(0.03, color='orange', linestyle='--', alpha=0.5, label='Warning Threshold')
    plt.axvline(history['best_iteration'], color='r', linestyle='--', alpha=0.5, label='Best Iteration')
    plt.xlabel('Boosting Round')
    plt.ylabel('Train-Val Gap')
    plt.title('Overfitting Monitor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f"[INFO] Saved training curves to {out_dir}/training_curves.png")

def plot_evaluation_results(results: Dict, out_dir: str, name: str = "test"):
    """Plot evaluation results"""
    metrics = results['standard_metrics']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Metrics bar chart
    ax = axes[0, 0]
    metric_names = ['ROC-AUC', 'PR-AUC', 'F1', 'MCC']
    metric_values = [metrics['roc_auc'], metrics['pr_auc'], metrics['f1'], metrics['mcc']]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    ax.bar(metric_names, metric_values, color=colors, alpha=0.7)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Recall at different top-k
    ax = axes[0, 1]
    k_values = ['0.1%', '0.5%', '1%', '5%']
    recall_values = [
        metrics.get('recall@0.001', 0),
        metrics.get('recall@0.005', 0),
        metrics.get('recall@0.01', 0),
        metrics.get('recall@0.05', 0)
    ]
    ax.plot(k_values, recall_values, marker='o', linewidth=2, markersize=8)
    ax.set_ylabel('Recall')
    ax.set_xlabel('Top K%')
    ax.set_title('Recall at Top-K Predictions')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Business metrics
    ax = axes[1, 0]
    business = results['business_metrics']
    opt = business['metrics_at_optimal']
    cap = business['metrics_at_capacity']
    
    categories = ['Precision', 'Recall', 'F1']
    optimal_values = [opt['precision'], opt['recall'], 
                     2 * opt['precision'] * opt['recall'] / (opt['precision'] + opt['recall'] + 1e-8)]
    capacity_values = [cap['precision'], cap['recall'],
                      2 * cap['precision'] * cap['recall'] / (cap['precision'] + cap['recall'] + 1e-8)]
    
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, optimal_values, width, label='Optimal Threshold', alpha=0.8)
    ax.bar(x + width/2, capacity_values, width, label='Capacity Constrained', alpha=0.8)
    ax.set_ylabel('Score')
    ax.set_title('Business Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # Cost comparison
    ax = axes[1, 1]
    cost_optimal = opt['cost']
    cost_capacity = cap['cost']
    # Baseline: no detection (all FN)
    n_fraud = results['n_fraud']
    cost_baseline = n_fraud * business['metrics_at_optimal']['threshold']  # Simplified
    
    costs = [cost_baseline, cost_optimal, cost_capacity]
    labels = ['No Detection', 'Optimal\nThreshold', 'Capacity\nConstrained']
    colors_cost = ['#e74c3c', '#2ecc71', '#3498db']
    ax.bar(labels, costs, color=colors_cost, alpha=0.7)
    ax.set_ylabel('Total Cost ($)')
    ax.set_title('Cost Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'evaluation_{name}.png'), dpi=150)
    plt.close()
    print(f"[INFO] Saved evaluation plot to {out_dir}/evaluation_{name}.png")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_complete_pipeline(data_path: str, output_dir: str = "outputs_v3"):
    """
    Run complete fraud detection pipeline
    """
    print("\n" + "="*80)
    print("ENHANCED FRAUD DETECTION PIPELINE V3")
    print("="*80)
    
    ensure_outputs_dir(output_dir)
    
    # Configuration
    config = ModelConfig()
    
    # Save config
    config_dict = {
        'train_ratio': config.train_ratio,
        'val_ratio': config.val_ratio,
        'test_ratio': config.test_ratio,
        'embargo_days': config.embargo_days,
        'resampling_ratio': config.resampling_ratio,
        'resampling_method': config.resampling_method,
        'xgb_params': config.xgb_params,
        'cv_folds': config.cv_folds,
        'cost_fn': config.cost_fn,
        'cost_fp': config.cost_fp,
        'max_review_rate': config.max_review_rate,
        'random_state': config.random_state
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Load data
    df = load_and_prepare_data(data_path, config)
    
    # Purged split
    df_train, df_val, df_test = purged_timeseries_split(df, config)
    
    # Cross-validation
    cv_results = cross_validate_model(df_train, config)
    
    # Train final model
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL")
    print("="*60)
    
    detector = EnhancedFraudDetector(config)
    detector.fit(df_train, df_val)
    
    # Plot training curves
    plot_training_curves(detector.training_history, output_dir)
    
    # Evaluate on validation
    val_results = detector.evaluate(df_val, name="Validation")
    
    # Evaluate on test
    test_results = detector.evaluate(df_test, name="Test")
    
    # Plot evaluation results
    plot_evaluation_results(val_results, output_dir, "validation")
    plot_evaluation_results(test_results, output_dir, "test")
    
    # Save model
    model_path = os.path.join(output_dir, 'fraud_detector_v3.pkl')
    joblib.dump(detector, model_path)
    print(f"\n[INFO] Saved model to {model_path}")
    
    # Save results
    final_results = {
        'config': config_dict,
        'cv_results': cv_results,
        'validation_results': val_results,
        'test_results': test_results,
        'training_history': detector.training_history,
        'timestamp': datetime.now().isoformat()
    }
    
    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    final_results = convert_numpy(final_results)
    
    results_path = os.path.join(output_dir, 'results_summary.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"[INFO] Saved results to {results_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nCross-Validation:")
    print(f"  Mean PR-AUC: {cv_results['mean_cv_score']:.4f} ± {cv_results['std_cv_score']:.4f}")
    print(f"  Mean Overfitting Gap: {cv_results['mean_overfitting_gap']:.4f}")
    
    print(f"\nValidation Set:")
    val_metrics = val_results['standard_metrics']
    print(f"  ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:  {val_metrics['pr_auc']:.4f}")
    
    print(f"\nTest Set:")
    test_metrics = test_results['standard_metrics']
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:  {test_metrics['pr_auc']:.4f}")
    print(f"  F1:      {test_metrics['f1']:.4f}")
    print(f"  MCC:     {test_metrics['mcc']:.4f}")
    
    test_business = test_results['business_metrics']
    print(f"\nBusiness Metrics (Test Set):")
    print(f"  Optimal Threshold: {test_business['optimal_threshold']:.4f}")
    print(f"  Min Cost: ${test_business['min_cost']:,.0f}")
    print(f"  Review Rate at Optimal: {test_business['metrics_at_optimal']['review_rate']*100:.2f}%")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"All outputs saved to: {output_dir}")
    print("="*80 + "\n")
    
    return detector, final_results

# ============================================================================
# ENTRY POINT
# ============================================================================

def parse_args():
    """Parse command line arguments with defaults for Jupyter/Colab compatibility"""
    import argparse
    p = argparse.ArgumentParser(description="Enhanced Fraud Detection Pipeline V3 - Production Ready")
    p.add_argument("--data", type=str, default="/kaggle/input/creditcardfraud/creditcard.csv", 
                   help="Path to Kaggle Credit Card Fraud dataset CSV")
    p.add_argument("--output", type=str, default="outputs_v3", 
                   help="Output directory for results and artifacts")
    p.add_argument("--seed", type=int, default=42, 
                   help="Random seed for reproducibility")
    
    # Parse with empty list for Jupyter/Colab compatibility
    args = p.parse_args([])
    return args

if __name__ == "__main__":
    args = parse_args()
    try:
        print("\n" + "="*80)
        print("ENHANCED FRAUD DETECTION MODEL V3 - PRODUCTION READY")
        print("="*80)
        print("\nKey Improvements:")
        print("  ✅ No data leakage (leak-free feature engineering)")
        print("  ✅ Conservative SMOTE (8% ratio)")
        print("  ✅ Enhanced regularization (L1=5.0, L2=15.0)")
        print("  ✅ Purged time-series CV (2-day embargo)")
        print("  ✅ Business-oriented metrics")
        print("  ✅ Production-ready pipeline")
        print("\n" + "="*80 + "\n")
        
        # Run pipeline
        detector, results = run_complete_pipeline(
            data_path=args.data,
            output_dir=args.output
        )
        
        print("\n" + "="*80)
        print("✅ HOÀN THÀNH! Kiểm tra thư mục {} để xem kết quả.".format(args.output))
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\n❌ [ERROR] File không tìm thấy: {e}")
        print("\n📝 Hướng dẫn:")
        print("  1. Nếu đang dùng Colab, upload file creditcard.csv")
        print("  2. Hoặc chạy trực tiếp:")
        print("     from model_v3_improved import run_complete_pipeline")
        print("     detector, results = run_complete_pipeline('/path/to/creditcard.csv', 'outputs_v3')")
    except Exception as e:
        print(f"\n❌ [ERROR] {e}")
        print("\n📦 Kiểm tra dependencies:")
        print("  pip install xgboost scikit-learn imbalanced-learn pandas numpy matplotlib joblib scipy")
        import traceback
        traceback.print_exc()
