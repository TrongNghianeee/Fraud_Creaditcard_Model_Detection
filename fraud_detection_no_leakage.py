# =========================
# FRAUD DETECTION - NO DATA LEAKAGE VERSION
# =========================

import pandas as pd
import numpy as np
import json
import os
import time
import warnings
from typing import Tuple, Dict
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, f1_score, precision_score, recall_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION - FEATURE SELECTION (FA)
# ============================================================================

@dataclass
class FAConfig:
    """Configuration cho Feature Selection (simulating Firefly Algorithm)"""
    
    # Feature selection parameters
    selection_ratio: float = 0.7           # Tỷ lệ features được chọn (70%)
    min_feature_ratio: float = 0.6         # Tối thiểu 60% features
    max_feature_ratio: float = 0.8         # Tối đa 80% features
    min_feature_count: int = 8             # Tối thiểu 8 features
    
    # Random seed
    random_state: int = 42
    
    # Selection mode
    feature_selection_mode: str = "random"  # "random", "importance", "correlation"
    
    # Advanced options (for future FA implementation)
    n_fireflies: int = 30                  # Số fireflies (nếu implement full FA)
    n_epochs: int = 20                     # Số epochs (nếu implement full FA)
    alpha: float = 0.25                    # Exploration rate
    beta0: float = 2.0                     # Attraction coefficient
    gamma: float = 0.20                    # Light absorption coefficient
    lambda_feat: float = 0.01              # Feature penalty
    diversity_threshold: float = 0.1       # Diversity preservation threshold
    patience: int = 6                      # Early stopping patience
    validation_strictness: float = 0.8     # Validation strictness
    overfitting_threshold: float = 0.03    # Overfitting threshold


# ============================================================================
# PHẦN 1: LOAD DATA - KHÔNG PREPROCESSING
# ============================================================================

def load_raw_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load dữ liệu RAW - KHÔNG tiền xử lý để tránh data leakage
    """
    print("[INFO] Loading RAW data from:", filepath)
    df = pd.read_csv(filepath)
    
    print("Shape ban đầu:", df.shape)
    print("\nTarget distribution:\n", df['is_fraud'].value_counts(normalize=True))
    
    # Chỉ drop các cột không cần thiết
    cols_to_drop = ['index', 'Unnamed: 0', 'trans_num']
    df.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')
    
    # Tách target
    y = df['is_fraud']
    X = df.drop('is_fraud', axis=1)
    
    print("\nFeatures:", X.columns.tolist())
    print("Shape:", X.shape)
    
    return X, y


# ============================================================================
# PHẦN 2: FEATURE ENGINEERING - TRONG PIPELINE
# ============================================================================

from sklearn.base import BaseEstimator, TransformerMixin

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features từ datetime columns"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Convert datetime
        X['trans_date_trans_time'] = pd.to_datetime(X['trans_date_trans_time'])
        X['dob'] = pd.to_datetime(X['dob'])
        
        # Extract features
        X['transaction_hour'] = X['trans_date_trans_time'].dt.hour
        X['transaction_day'] = X['trans_date_trans_time'].dt.dayofweek
        X['transaction_month'] = X['trans_date_trans_time'].dt.month
        X['age'] = (X['trans_date_trans_time'] - X['dob']).dt.days // 365
        
        # Drop original datetime columns
        X.drop(['trans_date_trans_time', 'dob', 'unix_time'], axis=1, inplace=True, errors='ignore')
        
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features"""
    
    def __init__(self):
        self.label_encoders = {}
    
    def fit(self, X, y=None):
        X = X.copy()
        
        # Identify categorical columns
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Fit label encoders
        for col in cat_cols:
            le = LabelEncoder()
            # Handle missing values
            X[col] = X[col].fillna('unknown')
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Transform using fitted encoders
        for col, le in self.label_encoders.items():
            if col in X.columns:
                X[col] = X[col].fillna('unknown')
                # Handle unseen categories
                X[col] = X[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else 'unknown'
                )
                X[col] = le.transform(X[col])
        
        return X


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Handle missing values"""
    
    def __init__(self):
        self.fill_values = {}
    
    def fit(self, X, y=None):
        X = X.copy()
        
        # For numeric columns, use median
        num_cols = X.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            self.fill_values[col] = X[col].median()
        
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Fill numeric missing values
        for col, fill_val in self.fill_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill_val)
        
        return X


# ============================================================================
# PHẦN 3: CREATE PIPELINE - NO LEAKAGE
# ============================================================================

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Feature Selection using random selection (simulating FA)"""
    
    def __init__(self, config: FAConfig = None):
        """
        Initialize FeatureSelector with FAConfig
        
        Args:
            config: FAConfig instance cho các tham số feature selection
        """
        if config is None:
            config = FAConfig()
        
        self.config = config
        self.selection_ratio = config.selection_ratio
        self.random_state = config.random_state
        self.min_feature_ratio = config.min_feature_ratio
        self.max_feature_ratio = config.max_feature_ratio
        self.min_feature_count = config.min_feature_count
        self.feature_selection_mode = config.feature_selection_mode
        
        self.selected_features_ = None
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        np.random.seed(self.random_state)
        
        # Get feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Calculate number of features to select
        n_features = len(self.feature_names_)
        min_features = max(self.min_feature_count, int(n_features * self.min_feature_ratio))
        max_features = int(n_features * self.max_feature_ratio)
        n_selected = int(n_features * self.selection_ratio)
        n_selected = max(min_features, min(n_selected, max_features))
        
        # Random selection (simulating FA)
        self.selected_features_ = np.random.choice(
            self.feature_names_, 
            size=n_selected, 
            replace=False
        ).tolist()
        
        print(f"[INFO] Feature Selection (FA Config):")
        print(f"  - Selection ratio: {self.selection_ratio:.2f}")
        print(f"  - Selected: {n_selected}/{n_features} features")
        print(f"  - Mode: {self.feature_selection_mode}")
        
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            # Convert to DataFrame for selection
            df = pd.DataFrame(X, columns=self.feature_names_)
            return df[self.selected_features_].values


def create_training_pipeline(random_state=42, use_feature_selection=False, fa_config=None):
    """
    Tạo pipeline training KHÔNG có data leakage
    
    Args:
        random_state: Random seed
        use_feature_selection: Nếu True, sử dụng Feature Selection (FA mode)
        fa_config: FAConfig instance cho feature selection (optional)
    """
    
    steps = [
        # Step 1: Extract date features
        ('date_features', DateFeatureExtractor()),
        
        # Step 2: Handle missing values
        ('missing_handler', MissingValueHandler()),
        
        # Step 3: Encode categorical
        ('categorical_encoder', CategoricalEncoder()),
        
        # Step 4: Scale numerical features
        ('scaler', StandardScaler()),
    ]
    
    # Step 5: Feature Selection (chỉ khi use_feature_selection=True)
    if use_feature_selection:
        if fa_config is None:
            fa_config = FAConfig()  # Sử dụng default config
        
        steps.append(('feature_selector', FeatureSelector(config=fa_config)))
    
    # Step 6: Resampling
    steps.extend([
        ('smote', BorderlineSMOTE(
            random_state=random_state,
            k_neighbors=5,
            sampling_strategy=0.1  # Chỉ tăng minority class lên 10% của majority
        )),
        
        # Step 7: ENN cleaning
        ('enn', EditedNearestNeighbours(
            n_neighbors=3,
            sampling_strategy='auto'
        )),
        
        # Step 8: XGBoost classifier
        ('classifier', XGBClassifier(
            objective='binary:logistic',
            tree_method='hist',
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2.0,
            reg_lambda=5.0,
            min_child_weight=5.0,
            eval_metric='aucpr',
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    
    pipeline = ImbPipeline(steps)
    
    return pipeline


# ============================================================================
# PHẦN 4: TRAINING & EVALUATION
# ============================================================================

def evaluate_model(pipeline, X, y, dataset_name="Dataset"):
    """Evaluate model trên dataset"""
    
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = pipeline.predict(X)
    
    roc_auc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred)
    
    print(f"\n{dataset_name} Metrics:")
    print(f"  ROC-AUC:    {roc_auc:.4f}")
    print(f"  PR-AUC:     {pr_auc:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    
    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'y_proba': y_proba
    }


def plot_curves(results_dict, out_dir='outputs'):
    """Plot ROC và PR curves"""
    
    os.makedirs(out_dir, exist_ok=True)
    
    # ROC Curve
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, res in results_dict.items():
        if 'y_true' in res and 'y_proba' in res:
            fpr, tpr, _ = roc_curve(res['y_true'], res['y_proba'])
            plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    
    # PR Curve
    plt.subplot(1, 2, 2)
    for name, res in results_dict.items():
        if 'y_true' in res and 'y_proba' in res:
            precision, recall, _ = precision_recall_curve(res['y_true'], res['y_proba'])
            plt.plot(recall, precision, label=f"{name} (AUC={res['pr_auc']:.4f})")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'roc_pr_curves_no_leakage.png'), dpi=150)
    plt.close()
    
    print(f"\n[INFO] Saved curves to {out_dir}/roc_pr_curves_no_leakage.png")


# ============================================================================
# PHẦN 5: MAIN TRAINING
# ============================================================================

def main(mode: str = 'xgboost_smoteenn'):
    """
    Main training function - NO DATA LEAKAGE
    
    Args:
        mode: 'xgboost_smoteenn' hoặc 'xgboost_fa_smoteenn'
    """
    
    print("\n" + "="*80)
    print(" FRAUD DETECTION - NO DATA LEAKAGE VERSION ")
    print("="*80)
    print(f"Mode: {mode.upper()}")
    print("="*80 + "\n")
    
    # Create output directory
    out_dir = 'outputs_no_leakage'
    os.makedirs(out_dir, exist_ok=True)
    
    # Load RAW data
    data_path = '/kaggle/input/fraud-detection/fraudTest.csv'  # Sử dụng fraudTest.csv
    X, y = load_raw_data(data_path)
    
    # FA Config (chỉ dùng khi mode = xgboost_fa_smoteenn)
    fa_config = FAConfig(
        selection_ratio=0.7,
        min_feature_ratio=0.6,
        max_feature_ratio=0.8,
        min_feature_count=8,
        random_state=42
    )
    
    # ========================================
    # CRITICAL: Split FIRST, then preprocess
    # ========================================
    
    print("\n[STEP 1] Splitting data (Train 60% / Val 20% / Test 20%)")
    print("="*60)
    
    # First split: Train vs (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.4,  # 40% cho val+test
        random_state=42, 
        stratify=y
    )
    
    # Second split: Val vs Test (chia đều 20% mỗi phần)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,  # 50% của 40% = 20%
        random_state=42,
        stratify=y_temp
    )
    
    print(f"Train: {X_train.shape}, fraud rate: {y_train.mean():.4f}")
    print(f"Val:   {X_val.shape}, fraud rate: {y_val.mean():.4f}")
    print(f"Test:  {X_test.shape}, fraud rate: {y_test.mean():.4f}")
    
    # ========================================
    # MODE: XGBoost + SMOTEENN
    # ========================================
    
    if mode == 'xgboost_smoteenn':
        print("\n" + "="*80)
        print(" XGBOOST + SMOTEENN (KHÔNG Feature Selection - No Leakage) ")
        print("="*80)
        
        start_time = time.time()
        
        # Create pipeline WITHOUT feature selection
        pipeline = create_training_pipeline(
            random_state=42, 
            use_feature_selection=False
        )
        
        # Calculate scale_pos_weight
        pos = int(y_train.sum())
        neg = len(y_train) - pos
        scale_pos_weight = neg / pos
        
        # Set scale_pos_weight
        pipeline.named_steps['classifier'].set_params(scale_pos_weight=scale_pos_weight)
        
        print(f"\n[INFO] Training with scale_pos_weight={scale_pos_weight:.2f}")
        
        # Fit pipeline
        pipeline.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"\n[INFO] Training completed in {training_time:.2f} seconds")
        
        # Evaluate
        print("\n" + "="*60)
        print(" EVALUATION RESULTS ")
        print("="*60)
        
        train_results = evaluate_model(pipeline, X_train, y_train, "TRAIN")
        val_results = evaluate_model(pipeline, X_val, y_val, "VALIDATION")
        test_results = evaluate_model(pipeline, X_test, y_test, "TEST")
        
        # Add y_true for plotting
        train_results['y_true'] = y_train
        val_results['y_true'] = y_val
        test_results['y_true'] = y_test
        
        # Plot curves
        plot_curves({
            'Train': train_results,
            'Validation': val_results,
            'Test': test_results
        }, out_dir)
        
        # Save model
        model_path = os.path.join(out_dir, 'fraud_detection_smoteenn.pkl')
        joblib.dump(pipeline, model_path)
        print(f"\n[INFO] Model saved to {model_path}")
        
        # Overfitting check
        print("\n" + "="*60)
        print(" OVERFITTING CHECK ")
        print("="*60)
        
        val_test_gap_roc = val_results['roc_auc'] - test_results['roc_auc']
        val_test_gap_pr = val_results['pr_auc'] - test_results['pr_auc']
        
        print(f"Val-Test ROC-AUC gap: {val_test_gap_roc:+.4f}")
        print(f"Val-Test PR-AUC gap:  {val_test_gap_pr:+.4f}")
        
        if abs(val_test_gap_roc) < 0.05 and abs(val_test_gap_pr) < 0.1:
            print("\n✅ Good generalization! Val and Test performance are similar.")
        else:
            print("\n⚠️  Warning: Possible overfitting or data distribution mismatch.")
        
        # Save results to JSON
        results_json = {
            'mode': mode,
            'train': {k: v for k, v in train_results.items() if k not in ['y_proba', 'y_true']},
            'val': {k: v for k, v in val_results.items() if k not in ['y_proba', 'y_true']},
            'test': {k: v for k, v in test_results.items() if k not in ['y_proba', 'y_true']},
            'training_time_sec': training_time,
            'val_test_gap': {
                'roc_auc': float(val_test_gap_roc),
                'pr_auc': float(val_test_gap_pr)
            }
        }
        
        with open(os.path.join(out_dir, 'results_no_leakage.json'), 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n[INFO] Results saved to {out_dir}/results_no_leakage.json")
    
    # ========================================
    # MODE: XGBoost + FA + SMOTEENN
    # ========================================
    
    elif mode == 'xgboost_fa_smoteenn':
        print("\n" + "="*80)
        print(" XGBOOST + FA + SMOTEENN (CÓ Feature Selection - No Leakage) ")
        print("="*80)
        
        print("\nFA Configuration:")
        print(f"  - Selection ratio: {fa_config.selection_ratio}")
        print(f"  - Min features: {fa_config.min_feature_count}")
        print(f"  - Random state: {fa_config.random_state}")
        
        start_time = time.time()
        
        # Create pipeline WITH feature selection
        pipeline = create_training_pipeline(
            random_state=42, 
            use_feature_selection=True,
            fa_config=fa_config  # Truyền FA config vào pipeline
        )
        
        # Calculate scale_pos_weight
        pos = int(y_train.sum())
        neg = len(y_train) - pos
        scale_pos_weight = neg / pos
        
        # Set scale_pos_weight
        pipeline.named_steps['classifier'].set_params(scale_pos_weight=scale_pos_weight)
        
        print(f"\n[INFO] Training with scale_pos_weight={scale_pos_weight:.2f}")
        
        # Fit pipeline
        pipeline.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"\n[INFO] Training completed in {training_time:.2f} seconds")
        
        # Get selected features
        selected_features = pipeline.named_steps['feature_selector'].selected_features_
        total_features = len(pipeline.named_steps['feature_selector'].feature_names_)
        
        print(f"\n[INFO] Selected {len(selected_features)}/{total_features} features")
        
        # Evaluate
        print("\n" + "="*60)
        print(" EVALUATION RESULTS ")
        print("="*60)
        
        train_results = evaluate_model(pipeline, X_train, y_train, "TRAIN")
        val_results = evaluate_model(pipeline, X_val, y_val, "VALIDATION")
        test_results = evaluate_model(pipeline, X_test, y_test, "TEST")
        
        # Add y_true for plotting
        train_results['y_true'] = y_train
        val_results['y_true'] = y_val
        test_results['y_true'] = y_test
        
        # Plot curves
        plot_curves({
            'Train': train_results,
            'Validation': val_results,
            'Test': test_results
        }, out_dir)
        
        # Save model
        model_path = os.path.join(out_dir, 'fraud_detection_fa_smoteenn.pkl')
        joblib.dump(pipeline, model_path)
        print(f"\n[INFO] Model saved to {model_path}")
        
        # Overfitting check
        print("\n" + "="*60)
        print(" OVERFITTING CHECK ")
        print("="*60)
        
        val_test_gap_roc = val_results['roc_auc'] - test_results['roc_auc']
        val_test_gap_pr = val_results['pr_auc'] - test_results['pr_auc']
        
        print(f"Val-Test ROC-AUC gap: {val_test_gap_roc:+.4f}")
        print(f"Val-Test PR-AUC gap:  {val_test_gap_pr:+.4f}")
        
        if abs(val_test_gap_roc) < 0.05 and abs(val_test_gap_pr) < 0.1:
            print("\n✅ Good generalization! Val and Test performance are similar.")
        else:
            print("\n⚠️  Warning: Possible overfitting or data distribution mismatch.")
        
        # Save results to JSON (bao gồm selected features)
        results_json = {
            'mode': mode,
            'train': {k: v for k, v in train_results.items() if k not in ['y_proba', 'y_true']},
            'val': {k: v for k, v in val_results.items() if k not in ['y_proba', 'y_true']},
            'test': {k: v for k, v in test_results.items() if k not in ['y_proba', 'y_true']},
            'training_time_sec': training_time,
            'val_test_gap': {
                'roc_auc': float(val_test_gap_roc),
                'pr_auc': float(val_test_gap_pr)
            },
            'feature_selection': {
                'fa_config': {
                    'selection_ratio': fa_config.selection_ratio,
                    'min_feature_ratio': fa_config.min_feature_ratio,
                    'max_feature_ratio': fa_config.max_feature_ratio,
                    'min_feature_count': fa_config.min_feature_count,
                    'random_state': fa_config.random_state,
                    'mode': fa_config.feature_selection_mode
                },
                'total_features': total_features,
                'selected_count': len(selected_features),
                'selected_features': selected_features
            }
        }
        
        with open(os.path.join(out_dir, 'results_no_leakage.json'), 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n[INFO] Results saved to {out_dir}/results_no_leakage.json")
    
    else:
        print(f"\n❌ ERROR: Invalid mode '{mode}'")
        print("Valid modes: 'xgboost_smoteenn' or 'xgboost_fa_smoteenn'")
        return None
    
    print("\n" + "="*80)
    print(" HOÀN THÀNH! ")
    print("="*80)
    
    return pipeline


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    """
    Chạy training với pipeline KHÔNG có data leakage
    
    Key improvements:
    1. Load RAW data (không preprocessing trước)
    2. Split FIRST (train/val/test = 60/20/20)
    3. Preprocessing trong Pipeline
    4. SMOTE/ENN chỉ áp dụng trên train set (sampling_strategy=0.1)
    5. Val và Test KHÔNG được resample
    
    Modes:
        - 'xgboost_smoteenn': XGBoost + SMOTEENN (KHÔNG Feature Selection)
        - 'xgboost_fa_smoteenn': XGBoost + FA + SMOTEENN (CÓ Feature Selection)
    
    Để tùy chỉnh FA config, sửa trực tiếp trong main() function.
    """
    
    # ========== VÍ DỤ SỬ DỤNG ==========
    
    # Ví dụ 1: Chạy XGBoost + SMOTEENN (KHÔNG Feature Selection)
    pipeline = main(mode='xgboost_smoteenn')
    
    # Ví dụ 2: Chạy XGBoost + FA + SMOTEENN (CÓ Feature Selection)
    # pipeline = main(mode='xgboost_fa_smoteenn')
    
    # Để thay đổi FA config, sửa trực tiếp trong hàm main() ở dòng fa_config = FAConfig(...)
