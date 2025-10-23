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

def create_training_pipeline(random_state=42):
    """
    Tạo pipeline training KHÔNG có data leakage
    """
    
    # Tính scale_pos_weight sẽ được tính trong hàm train
    
    pipeline = ImbPipeline([
        # Step 1: Extract date features
        ('date_features', DateFeatureExtractor()),
        
        # Step 2: Handle missing values
        ('missing_handler', MissingValueHandler()),
        
        # Step 3: Encode categorical
        ('categorical_encoder', CategoricalEncoder()),
        
        # Step 4: Scale numerical features
        ('scaler', StandardScaler()),
        
        # Step 5: Resampling (chỉ áp dụng trên training data)
        # Chỉ resample 10% để tránh overfitting
        ('smote', BorderlineSMOTE(
            random_state=random_state,
            k_neighbors=5,
            sampling_strategy=0.1  # Chỉ tăng minority class lên 10% của majority
        )),
        
        # Step 6: ENN cleaning
        ('enn', EditedNearestNeighbours(
            n_neighbors=3,
            sampling_strategy='auto'
        )),
        
        # Step 7: XGBoost classifier
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

def main(data_path: str = None, mode: str = 'both'):
    """
    Main training function - NO DATA LEAKAGE
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
    if data_path is None:
        data_path = '/kaggle/input/fraud-detection/fraudTest.csv'  # Sử dụng fraudTest.csv
    
    X, y = load_raw_data(data_path)
    
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
    # MODE 1: XGBoost + SMOTEENN
    # ========================================
    
    if mode in ['xgboost_smoteenn', 'both']:
        print("\n" + "="*80)
        print(" MODE 1: XGBOOST + SMOTEENN (với Pipeline - No Leakage) ")
        print("="*80)
        
        start_time = time.time()
        
        # Create pipeline
        pipeline = create_training_pipeline(random_state=42)
        
        # Calculate scale_pos_weight
        pos = int(y_train.sum())
        neg = len(y_train) - pos
        scale_pos_weight = neg / pos
        
        # Set scale_pos_weight
        pipeline.named_steps['classifier'].set_params(scale_pos_weight=scale_pos_weight)
        
        print(f"\n[INFO] Training with scale_pos_weight={scale_pos_weight:.2f}")
        
        # Fit pipeline (preprocessing + resampling + training)
        # SMOTE và ENN CHỈ được áp dụng trên train set
        pipeline.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        print(f"\n[INFO] Training completed in {training_time:.2f} seconds")
        
        # Evaluate on all sets
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
        
        # Check for overfitting
        print("\n" + "="*60)
        print(" OVERFITTING CHECK ")
        print("="*60)
        
        train_test_gap_roc = train_results['roc_auc'] - test_results['roc_auc']
        train_test_gap_pr = train_results['pr_auc'] - test_results['pr_auc']
        val_test_gap_roc = val_results['roc_auc'] - test_results['roc_auc']
        val_test_gap_pr = val_results['pr_auc'] - test_results['pr_auc']
        
        print(f"Train-Test ROC-AUC gap: {train_test_gap_roc:+.4f}")
        print(f"Train-Test PR-AUC gap:  {train_test_gap_pr:+.4f}")
        print(f"Val-Test ROC-AUC gap:   {val_test_gap_roc:+.4f}")
        print(f"Val-Test PR-AUC gap:    {val_test_gap_pr:+.4f}")
        
        if abs(val_test_gap_roc) < 0.05 and abs(val_test_gap_pr) < 0.1:
            print("\n✅ Good generalization! Val and Test performance are similar.")
        else:
            print("\n⚠️  Warning: Possible overfitting or data distribution mismatch.")
        
        # Save model
        model_path = os.path.join(out_dir, 'fraud_detection_no_leakage.pkl')
        joblib.dump(pipeline, model_path)
        print(f"\n[INFO] Model saved to {model_path}")
        
        # Save results
        results_summary = {
            'train': {k: v for k, v in train_results.items() if k != 'y_proba' and k != 'y_true'},
            'val': {k: v for k, v in val_results.items() if k != 'y_proba' and k != 'y_true'},
            'test': {k: v for k, v in test_results.items() if k != 'y_proba' and k != 'y_true'},
            'training_time_sec': training_time,
            'gaps': {
                'train_test_roc_gap': float(train_test_gap_roc),
                'train_test_pr_gap': float(train_test_gap_pr),
                'val_test_roc_gap': float(val_test_gap_roc),
                'val_test_pr_gap': float(val_test_gap_pr)
            }
        }
        
        with open(os.path.join(out_dir, 'results_no_leakage.json'), 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"[INFO] Results saved to {out_dir}/results_no_leakage.json")
    
    print("\n" + "="*80)
    print(" HOÀN THÀNH! ")
    print("="*80)
    
    return {
        'pipeline': pipeline if mode in ['xgboost_smoteenn', 'both'] else None,
        'train_results': train_results if mode in ['xgboost_smoteenn', 'both'] else None,
        'val_results': val_results if mode in ['xgboost_smoteenn', 'both'] else None,
        'test_results': test_results if mode in ['xgboost_smoteenn', 'both'] else None
    }


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    """
    Chạy training với pipeline KHÔNG có data leakage
    
    Key improvements:
    1. Load RAW data (không preprocessing trước)
    2. Split FIRST (train/val/test)
    3. Preprocessing trong Pipeline
    4. SMOTE/ENN chỉ áp dụng trên train set
    5. Val và Test KHÔNG được resample
    """
    
    # Chạy training
    results = main(mode='xgboost_smoteenn')
    
    # Hoặc với custom data path
    # results = main(data_path='your_path.csv', mode='xgboost_smoteenn')
