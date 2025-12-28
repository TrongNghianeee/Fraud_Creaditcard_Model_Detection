# Fraud Detection - No Data Leakage Version

import os

# Limit BLAS/OpenMP threads early (must be set before numpy/scikit-learn imports)
# Helps avoid OpenBLAS OpenMP-loop warnings/hangs in some Linux runtimes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd
import numpy as np
import json
import time
import warnings
import subprocess
import math
from typing import Tuple
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, f1_score, precision_score, recall_score
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")


# Configuration - Feature Selection (FA)

@dataclass
class FAConfig:
    """Configuration for Firefly Algorithm Feature Selection"""
    selection_ratio: float = 0.7
    min_feature_ratio: float = 0.6
    max_feature_ratio: float = 0.8
    min_feature_count: int = 10
    random_state: int = 42
    feature_selection_mode: str = "importance"
    fa_variant: str = "alfa"  # standard | lfa | afa | alfa
    n_fireflies: int = 30
    n_epochs: int = 15
    alpha: float = 0.25
    alpha_decay: float = 0.95
    beta_decay: float = 0.92
    beta0: float = 2.0
    gamma: float = 0.20
    levy_beta: float = 1.5
    levy_scale: float = 0.1
    lambda_feat: float = 0.005
    diversity_threshold: float = 0.1
    patience: int = 6
    validation_strictness: float = 0.8
    overfitting_threshold: float = 0.03


# Data Loading

def load_raw_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load raw data without preprocessing to avoid data leakage"""
    print("Loading data from:", filepath)
    df = pd.read_csv(filepath)
    
    print("Initial shape:", df.shape)
    print("\nTarget distribution:\n", df['is_fraud'].value_counts(normalize=True))
    
    cols_to_drop = ['index', 'Unnamed: 0', 'trans_num']
    df.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')
    y = df['is_fraud']
    X = df.drop('is_fraud', axis=1)
    
    print("\nFeatures:", X.columns.tolist())
    print("Shape:", X.shape)
    
    return X, y

# Feature Engineering

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from datetime columns"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X['trans_date_trans_time'] = pd.to_datetime(X['trans_date_trans_time'])
        X['dob'] = pd.to_datetime(X['dob'])
        
        X['transaction_hour'] = X['trans_date_trans_time'].dt.hour
        X['transaction_day'] = X['trans_date_trans_time'].dt.dayofweek
        X['transaction_month'] = X['trans_date_trans_time'].dt.month
        X['age'] = (X['trans_date_trans_time'] - X['dob']).dt.days // 365
        X.drop(['trans_date_trans_time', 'dob', 'unix_time'], axis=1, inplace=True, errors='ignore')
        
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features"""
    
    def __init__(self):
        self.label_encoders = {}
    
    def fit(self, X, y=None):
        X = X.copy()
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in cat_cols:
            le = LabelEncoder()
            # Handle missing values
            X[col] = X[col].fillna('unknown')
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
        
        return self
    
    def transform(self, X):
        X = X.copy()
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
        num_cols = X.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            self.fill_values[col] = X[col].median()
        
        return self
    
    def transform(self, X):
        X = X.copy()
        for col, fill_val in self.fill_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill_val)
        
        return X


# Pipeline Creation

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Feature Selection using FULL Firefly Algorithm"""
    
    def __init__(self, config: FAConfig = None):
        """Initialize FeatureSelector with FAConfig"""
        if config is None:
            config = FAConfig()
        
        self.config = config
        self.selection_ratio = config.selection_ratio
        self.random_state = config.random_state
        self.min_feature_ratio = config.min_feature_ratio
        self.max_feature_ratio = config.max_feature_ratio
        self.min_feature_count = config.min_feature_count
        self.fa_variant = config.fa_variant.lower()
        if self.fa_variant not in ['standard', 'lfa', 'afa', 'alfa']:
            print(f"[WARN] Unknown FA variant '{self.fa_variant}', fallback to standard")
            self.fa_variant = 'standard'
        
        # FA parameters
        self.n_fireflies = config.n_fireflies
        self.n_epochs = config.n_epochs
        self.alpha = config.alpha
        self.alpha_decay = config.alpha_decay
        self.beta_decay = config.beta_decay
        self.beta0 = config.beta0
        self.gamma = config.gamma
        self.levy_beta = config.levy_beta
        self.levy_scale = config.levy_scale
        self.lambda_feat = config.lambda_feat
        self.diversity_threshold = config.diversity_threshold
        self.patience = config.patience
        
        self.selected_features_ = None
        self.feature_names_ = None
        self.best_fitness_ = -np.inf
        self.fitness_history_ = []
    
    def _initialize_fireflies(self, n_features, target_n_features):
        """Initialize firefly population (binary vectors)"""
        fireflies = []
        for _ in range(self.n_fireflies):
            # Random binary vector
            firefly = np.random.rand(n_features) < (target_n_features / n_features)
            # Ensure minimum features
            if firefly.sum() < self.min_feature_count:
                indices = np.random.choice(n_features, self.min_feature_count, replace=False)
                firefly = np.zeros(n_features, dtype=bool)
                firefly[indices] = True
            fireflies.append(firefly.astype(float))
        return np.array(fireflies)
    
    def _calculate_fitness(self, firefly, X, y, verbose=False):
        """Calculate fitness of a firefly (feature subset)"""
        selected_indices = firefly > 0.5
        n_selected = selected_indices.sum()
        
        # Check minimum features
        if n_selected < self.min_feature_count:
            return -1000.0
        
        # Get selected features
        X_selected = X[:, selected_indices] if not isinstance(X, pd.DataFrame) else X.iloc[:, selected_indices]
        
        # Quick validation using XGBoost (faster and more relevant than RF)
        from xgboost import XGBClassifier
        from sklearn.model_selection import cross_val_score
        
        # Use small XGBoost for fast evaluation
        clf = XGBClassifier(
            n_estimators=50, 
            max_depth=3, 
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=1,
            verbosity=0,
            eval_metric='aucpr'
        )
        
        # Cross-validation score (3-fold takes ~5-10s per evaluation)
        try:
            scores = cross_val_score(clf, X_selected, y, cv=3, scoring='average_precision', n_jobs=1)
            auc_score = scores.mean()
        except:
            auc_score = 0.0
        
        # Fitness = AUC - feature penalty
        fitness = auc_score - self.lambda_feat * (n_selected / len(firefly))
        
        return fitness
    
    def _distance(self, firefly_i, firefly_j):
        """Calculate Euclidean distance between two fireflies"""
        return np.sqrt(np.sum((firefly_i - firefly_j) ** 2))
    
    def _attractiveness(self, distance, beta0_override=None):
        """Calculate attractiveness based on distance"""
        base_beta0 = beta0_override if beta0_override is not None else self.beta0
        return base_beta0 * np.exp(-self.gamma * distance ** 2)

    def _levy_flight_step(self, dim):
        """Generate Lévy flight step using Mantegna's algorithm (heavy-tailed jumps)"""
        beta = self.levy_beta
        sigma_u = (
            math.gamma(1 + beta)
            * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))
        ) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=dim)
        v = np.random.normal(0, 1, size=dim)
        step = u / ((np.abs(v) ** (1 / beta)) + 1e-12)
        return self.levy_scale * step
    
    def _move_firefly(self, firefly_i, firefly_j, alpha, beta0_override=None, use_levy=False):
        """Move firefly i towards firefly j with optional Lévy flight and adaptive beta"""
        distance = self._distance(firefly_i, firefly_j)
        beta = self._attractiveness(distance, beta0_override=beta0_override)
        
        # Movement equation with selectable exploration noise
        random_vector = self._levy_flight_step(len(firefly_i)) if use_levy else (np.random.rand(len(firefly_i)) - 0.5)
        new_position = firefly_i + beta * (firefly_j - firefly_i) + alpha * random_vector
        
        # Convert to binary (0 or 1)
        new_position = (1 / (1 + np.exp(-new_position)) > 0.5).astype(float)
        
        return new_position
    
    def _ensure_diversity(self, fireflies):
        """Ensure diversity in population"""
        unique_fireflies = []
        for firefly in fireflies:
            is_duplicate = False
            for unique in unique_fireflies:
                if np.array_equal(firefly, unique):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_fireflies.append(firefly)
        
        # Add random fireflies if diversity is low
        n_features = len(fireflies[0])
        while len(unique_fireflies) < self.n_fireflies:
            random_firefly = (np.random.rand(n_features) < 0.5).astype(float)
            # Ensure minimum features
            if random_firefly.sum() < self.min_feature_count:
                indices = np.random.choice(n_features, self.min_feature_count, replace=False)
                random_firefly = np.zeros(n_features, dtype=float)
                random_firefly[indices] = 1.0
            unique_fireflies.append(random_firefly)
        
        # Return exactly n_fireflies
        return np.array(unique_fireflies[:self.n_fireflies])
    
    def fit(self, X, y=None):
        """Fit using Firefly Algorithm for feature selection"""
        np.random.seed(self.random_state)
        
        print("\n" + "="*60)
        print(" FIREFLY ALGORITHM - FEATURE SELECTION ")
        print("="*60)
        
        # Get feature names - IMPORTANT: These are post-transformation features
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_array = X.values
        else:
            # If not DataFrame, feature names should match post-transformation
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X
        
        n_features = len(self.feature_names_)
        
        print(f"\nFeature names: {self.feature_names_[:10]}... ({n_features} total)")
        
        # Calculate target number of features
        min_features = max(self.min_feature_count, int(n_features * self.min_feature_ratio))
        max_features = int(n_features * self.max_feature_ratio)
        target_n_features = int(n_features * self.selection_ratio)
        target_n_features = max(min_features, min(target_n_features, max_features))
        
        print(f"\nFA Configuration:")
        print(f"  - Variant: {self.fa_variant.upper()} (standard | LFA | AFA | ALFA)")
        print(f"  - Total features: {n_features}")
        print(f"  - Target features: {target_n_features} ({self.selection_ratio:.1%})")
        print(f"  - Fireflies: {self.n_fireflies}")
        print(f"  - Epochs: {self.n_epochs}")
        print(f"  - Alpha (randomness): {self.alpha} (decay={self.alpha_decay})")
        print(f"  - Beta0 (attraction): {self.beta0} (decay={self.beta_decay if self.fa_variant in ['afa', 'alfa'] else 'none'})")
        print(f"  - Gamma (absorption): {self.gamma}")
        if self.fa_variant in ['lfa', 'alfa']:
            print(f"  - Lévy flight: beta={self.levy_beta}, scale={self.levy_scale}")
        print(f"\nInitializing {self.n_fireflies} fireflies (may take 1-2 minutes)...")
        print(f"Each firefly requires XGBoost training with 3-fold CV")
        
        fireflies = self._initialize_fireflies(n_features, target_n_features)
        
        print("Evaluating initial population...")
        fitness_values = []
        for idx, f in enumerate(fireflies):
            fitness = self._calculate_fitness(f, X_array, y)
            fitness_values.append(fitness)
            if (idx + 1) % 10 == 0:
                print(f"       Progress: {idx+1}/{self.n_fireflies} fireflies evaluated")
        fitness_values = np.array(fitness_values)
        
        # Track best solution
        best_idx = np.argmax(fitness_values)
        best_firefly = fireflies[best_idx].copy()
        self.best_fitness_ = fitness_values[best_idx]
        
        print(f"\nInitial population ready!")
        print(f"  Best fitness: {self.best_fitness_:.4f}")
        print(f"  Features selected: {int(best_firefly.sum())}")
        print(f"\nStarting FA optimization ({self.n_epochs} epochs)...")
        print(f"Estimated time: ~{self.n_epochs * 2}-{self.n_epochs * 3} minutes")
        
        # Early stopping
        no_improvement = 0
        
        # FA iterations
        for epoch in range(self.n_epochs):
            epoch_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{self.n_epochs}")
            print(f"{'='*60}")
            
            # Adaptive alpha/beta per variant
            alpha_t = self.alpha * (self.alpha_decay ** epoch)
            beta0_t = self.beta0 * (self.beta_decay ** epoch) if self.fa_variant in ['afa', 'alfa'] else self.beta0
            use_levy = self.fa_variant in ['lfa', 'alfa']
            
            # Track evaluations in this epoch
            evaluations_count = 0
            
            # For each firefly
            for i in range(self.n_fireflies):
                # Compare with all other fireflies
                moved = False
                for j in range(self.n_fireflies):
                    if fitness_values[j] > fitness_values[i]:
                        # Move firefly i towards brighter firefly j
                        fireflies[i] = self._move_firefly(
                            fireflies[i],
                            fireflies[j],
                            alpha_t,
                            beta0_override=beta0_t,
                            use_levy=use_levy
                        )
                        
                        # Recalculate fitness
                        fitness_values[i] = self._calculate_fitness(fireflies[i], X_array, y)
                        evaluations_count += 1
                        moved = True
                
                # Progress update every 10 fireflies
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{self.n_fireflies} fireflies ({evaluations_count} evaluations)")
            
            # Update best solution
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            
            epoch_time = time.time() - epoch_start_time
            
            if current_best_fitness > self.best_fitness_:
                improvement = current_best_fitness - self.best_fitness_
                self.best_fitness_ = current_best_fitness
                best_firefly = fireflies[current_best_idx].copy()
                no_improvement = 0
                print(f"\n  NEW BEST FITNESS: {self.best_fitness_:.4f} (+{improvement:.4f})")
                print(f"  Features: {int(best_firefly.sum())}/{n_features}")
                print(f"  Time: {epoch_time:.1f}s")
            else:
                no_improvement += 1
                print(f"\n  No improvement (fitness: {self.best_fitness_:.4f})")
                print(f"  Patience: {no_improvement}/{self.patience}")
                print(f"  Time: {epoch_time:.1f}s")
            
            self.fitness_history_.append(self.best_fitness_)
            
            # Ensure diversity every few epochs
            if (epoch + 1) % 5 == 0:
                print(f"\n  Ensuring population diversity...")
                fireflies = self._ensure_diversity(fireflies)
                print(f"  Re-evaluating {self.n_fireflies} fireflies...")
                fitness_values = []
                for idx, f in enumerate(fireflies):
                    fitness_values.append(self._calculate_fitness(f, X_array, y))
                    if (idx + 1) % 10 == 0:
                        print(f"     Progress: {idx+1}/{self.n_fireflies}")
                fitness_values = np.array(fitness_values)
            
            # Early stopping
            if no_improvement >= self.patience:
                print(f"\n{'='*60}")
                print(f"EARLY STOPPING at epoch {epoch+1}")
                print(f"No improvement for {self.patience} consecutive epochs")
                print(f"{'='*60}")
                break
        
        # Get selected features from best firefly
        selected_indices = best_firefly > 0.5
        self.selected_features_ = [self.feature_names_[i] for i in range(n_features) if selected_indices[i]]
        
        print(f"\nFeature Selection Complete!")
        print(f"  Final fitness: {self.best_fitness_:.4f}")
        print(f"  Selected: {len(self.selected_features_)}/{n_features} features")
        print(f"  Selection ratio: {len(self.selected_features_)/n_features:.1%}")
        print("="*60 + "\n")
        
        return self
    
    def transform(self, X):
        """Transform by selecting best features"""
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            # Convert to DataFrame for selection
            df = pd.DataFrame(X, columns=self.feature_names_)
            return df[self.selected_features_].values
# Training & Evaluation

def find_optimal_threshold(y_true, y_proba):
    """Find a single optimal threshold (F1-maximizing) on a validation set."""

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Calculate F1 for each threshold (thresholds has length N, precisions/recalls has length N+1)
    f1_scores = []
    for i in range(len(thresholds)):
        p = precisions[i]
        r = recalls[i]
        if p + r > 0:
            f1 = 2 * (p * r) / (p + r)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    f1_scores = np.array(f1_scores)
    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    return optimal_threshold, best_f1, {
        'precision': float(precisions[best_idx]),
        'recall': float(recalls[best_idx]),
        'f1': best_f1,
        'threshold': optimal_threshold,
    }


def evaluate_model_with_threshold(pipeline, X, y, threshold=0.5, dataset_name="Dataset"):
    """Evaluate model với custom threshold"""
    
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    roc_auc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred)
    
    print(f"\n{dataset_name} Metrics (threshold={threshold:.4f}):")
    print(f"  ROC-AUC:    {roc_auc:.4f}")
    print(f"  PR-AUC:     {pr_auc:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    
    # Ensure all values are native Python types (not numpy)
    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'y_proba': y_proba,  # Keep as array for plotting
        'threshold': float(threshold)
    }


def display_image(image_path):
    """Display image inline or open with default viewer"""
    try:
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img.show()
            print(f"Opened image: {image_path}")
        else:
            print(f"Image not found: {image_path}")
    except Exception as e:
        print(f"Could not display image {image_path}: {e}")


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
    
    print(f"\nSaved curves to {out_dir}/roc_pr_curves_no_leakage.png")


def plot_feature_importance(pipeline, feature_names, out_dir='outputs', top_n=20):

    os.makedirs(out_dir, exist_ok=True)
    
    # Get XGBoost classifier from pipeline
    xgb_model = pipeline.named_steps['classifier']
    
    # Get feature importance
    importance = xgb_model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Feature Importance (Gain)')
    plt.title(f'Top {top_n} Feature Importances - XGBoost')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(out_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[INFO] Saved feature importance plot to {out_dir}/feature_importance.png")
    
    # Save to JSON
    importance_dict = {
        'top_features': top_features.to_dict('records'),
        'all_features': importance_df.to_dict('records')
    }
    
    with open(os.path.join(out_dir, 'feature_importance.json'), 'w') as f:
        json.dump(importance_dict, f, indent=2)
    
    print(f"[INFO] Saved feature importance data to {out_dir}/feature_importance.json")
    
    # Print top features
    print(f"\n{'='*60}")
    print(f" TOP {top_n} MOST IMPORTANT FEATURES ")
    print(f"{'='*60}")
    for idx, row in top_features.iterrows():
        print(f"  {row['feature']:30s} : {row['importance']:.6f}")
    print(f"{'='*60}\n")
    
    return importance_df


# Hyperparameter Tuning

def tune_xgboost_with_optuna(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials=50,
    use_gpu=False,
    scale_pos_weight=None,
):

    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        print("[WARNING] Optuna not installed. Skipping hyperparameter tuning.")
        print("Using default hyperparameters instead")
        return None
    
    print("\n" + "="*60)
    print(" OPTUNA HYPERPARAMETER TUNING ")
    print("="*60)
    print(f"Running {n_trials} trials...")
    
    def objective(trial):
        """Optuna objective function"""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'random_state': 42,
            'verbosity': 0,
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 10.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        }

        # If provided, keep class-imbalance weighting fixed during tuning
        if scale_pos_weight is not None:
            params['scale_pos_weight'] = float(scale_pos_weight)
        
        if use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['device'] = 'cuda'
        else:
            params['tree_method'] = 'hist'
            params['device'] = 'cpu'
            params['n_jobs'] = -1
        
        clf = XGBClassifier(**params)
        clf.fit(X_train, y_train)
        y_val_proba = clf.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_val_proba)
        
        return pr_auc
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    best_score = study.best_value
    
    print(f"\nOptuna tuning complete!")
    print(f"  Best PR-AUC: {best_score:.4f}")
    print(f"  Best parameters:")
    for key, value in best_params.items():
        print(f"    {key}: {value}")
    print("="*60 + "\n")
    
    return best_params


# Pipeline Wrappers

class FraudDetectionPipeline:
    """Complete fraud detection pipeline with preprocessing + classifier + threshold"""
    def __init__(self, preprocessor, classifier, threshold=0.5):
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.threshold = threshold
    
    def predict_proba(self, X):
        """Predict probability for fraud class"""
        X_processed = self.preprocessor.transform(X)
        return self.classifier.predict_proba(X_processed)
    
    def predict(self, X):
        """Predict fraud class with optimal threshold"""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
    
    def get_params(self):
        """Get pipeline parameters"""
        return {
            'threshold': self.threshold,
            'classifier_params': self.classifier.get_params(),
            'preprocessor_steps': list(self.preprocessor.named_steps.keys())
        }


class FullPreprocessingPipeline:
    """Wrapper combining preprocessing + feature selection"""
    def __init__(self, preprocessor, feature_selector):
        self.preprocessor = preprocessor
        self.feature_selector = feature_selector
    
    def transform(self, X):
        """Apply preprocessing then feature selection"""
        X_processed = self.preprocessor.transform(X)
        
        if not isinstance(X_processed, pd.DataFrame):
            feature_names = self.feature_selector.feature_names_
            X_processed = pd.DataFrame(X_processed, columns=feature_names)
        X_selected = X_processed[self.feature_selector.selected_features_]
        return X_selected


# Main Training Function

def main(mode: str = 'xgboost_smoteenn'):
    """Main training function without data leakage"""
    
    print("\n" + "="*80)
    print(" FRAUD DETECTION - NO DATA LEAKAGE ")
    print("="*80)
    print(f"Mode: {mode.upper()}")
    print("="*80 + "\n")
    
    # Create output directory
    out_dir = 'outputs_no_leakage'
    os.makedirs(out_dir, exist_ok=True)
    
    # Load RAW data
    data_path = '/kaggle/input/fraud-detection/fraudTrain.csv'  # Sử dụng fraudTrain.csv để so sánh fair
    X, y = load_raw_data(data_path)
    
    fa_config = FAConfig(
        selection_ratio=0.7,         # Tăng 0.6 → 0.8 (giữ nhiều features hơn)
        min_feature_ratio=0.6,       # Tăng 0.5 → 0.7 (tối thiểu 70% features)
        max_feature_ratio=0.8,       # Tăng 0.7 → 0.9 (tối đa 90% features)
        min_feature_count=12,        # Tăng 10 → 15 (giữ ít nhất 15 features)
        n_fireflies=30,              # Tăng 40 → 50 (population lớn hơn)
        n_epochs=10,                 # Tăng 10 → 15 (nhiều epochs hơn)
        patience=4,                  # Tăng 4 → 6 (kiên nhẫn hơn)
        fa_variant="alfa",          # Mặc định dùng Adaptive + Lévy
        random_state=42
    )
    
    print("\nSplitting data (Train 60% / Val 20% / Test 20%)")
    print("="*60)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.4,
        random_state=42, 
        stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )
    
    print(f"Train: {X_train.shape}, fraud rate: {y_train.mean():.4f}")
    print(f"Val:   {X_val.shape}, fraud rate: {y_val.mean():.4f}")
    print(f"Test:  {X_test.shape}, fraud rate: {y_test.mean():.4f}")

    # NOTE: SMOTE+ENN resampling removed.
    # Modes below train on the original imbalanced data and use scale_pos_weight.
    
    if mode == 'xgboost_smoteenn':
        print("\n" + "="*80)
        print(" XGBOOST (NO FEATURE SELECTION) ")
        print("="*80)
        
        start_time = time.time()
        
        print("\nCreating preprocessing pipeline...")
        pipeline_steps = [
            ('date_features', DateFeatureExtractor()),
            ('missing_handler', MissingValueHandler()),
            ('categorical_encoder', CategoricalEncoder()),
            ('scaler', StandardScaler()),
        ]
        
        from sklearn.pipeline import Pipeline as SkPipeline
        preprocessing_pipeline = SkPipeline(pipeline_steps)
        
        print("Running preprocessing on all sets...")
        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_val_processed = preprocessing_pipeline.transform(X_val)
        X_test_processed = preprocessing_pipeline.transform(X_test)
        
        # Get feature names after preprocessing
        if isinstance(X_train_processed, pd.DataFrame):
            all_features = X_train_processed.columns.tolist()
        else:
            all_features = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
        
        total_features = len(all_features)
        print(f"\nTotal features: {total_features}")
        print(f"Using ALL features (no feature selection)")
        
        print("\n[INFO] Training on original imbalanced data (no SMOTE+ENN)")
        X_train_final, y_train_final = X_train_processed, y_train

        print(f"  - Train: {X_train_final.shape}")
        print(f"  - Fraud rate: {y_train_final.mean():.4f}")
        
        gpu_available = False
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_available = True
                print("\nGPU detected for Optuna tuning")
        except:
            print("\nNo GPU - using CPU for Optuna tuning")
        
        # Calculate scale_pos_weight from ORIGINAL training labels
        pos = int(y_train_final.sum())
        neg = len(y_train_final) - pos
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0

        print("\nRunning Optuna hyperparameter tuning (with fixed scale_pos_weight)...")
        best_params = tune_xgboost_with_optuna(
            X_train_final, y_train_final,
            X_val_processed, y_val,
            n_trials=50,
            use_gpu=gpu_available,
            scale_pos_weight=scale_pos_weight
        )
        
        if best_params is None:
            best_params = {
                'n_estimators': 250,
                'learning_rate': 0.04,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 3.0,
                'reg_lambda': 6.0,
                'min_child_weight': 6.0,
                'gamma': 0.0
            }
        best_params['objective'] = 'binary:logistic'
        best_params['eval_metric'] = 'aucpr'
        best_params['random_state'] = 42
        best_params['verbosity'] = 0
        
        if gpu_available:
            best_params['tree_method'] = 'gpu_hist'
            best_params['device'] = 'cuda'
        else:
            best_params['tree_method'] = 'hist'
            best_params['device'] = 'cpu'
            best_params['n_jobs'] = -1
        
        best_params['scale_pos_weight'] = scale_pos_weight
        
        # STEP 4: Train final model with best params
        print(f"\n[STEP 4] Training final XGBoost with optimized params...")
        print(f"  - scale_pos_weight: {scale_pos_weight:.2f}")
        
        final_classifier = XGBClassifier(**best_params)
        final_classifier.fit(X_train_final, y_train_final)
        
        training_time = time.time() - start_time
        print(f"\n[INFO] Training completed in {training_time:.2f} seconds")
        
        print("\nFinding optimal thresholds on validation set...")
        y_val_proba = final_classifier.predict_proba(X_val_processed)[:, 1]
        
        optimal_threshold, best_f1, threshold_metrics = find_optimal_threshold(
            y_val, y_val_proba
        )
        
        print(f"\nOptimal threshold found on validation set (F1-max):")
        print(f"  - Threshold: {optimal_threshold:.4f}")
        print(f"  - Val Precision: {threshold_metrics['precision']:.4f}")
        print(f"  - Val Recall: {threshold_metrics['recall']:.4f}")
        print(f"  - Val F1-Score: {threshold_metrics['f1']:.4f}")
        
        print("\n" + "="*60)
        print(" EVALUATION RESULTS ")
        print("="*60)
        
        train_results = evaluate_model_with_threshold(
            type('Pipeline', (), {
                'predict_proba': lambda self, X: final_classifier.predict_proba(
                    preprocessing_pipeline.transform(X)
                )
            })(),
            X_train, y_train,
            threshold=optimal_threshold,
            dataset_name="TRAIN"
        )
        
        val_results = evaluate_model_with_threshold(
            type('Pipeline', (), {
                'predict_proba': lambda self, X: final_classifier.predict_proba(
                    preprocessing_pipeline.transform(X)
                )
            })(),
            X_val, y_val,
            threshold=optimal_threshold,
            dataset_name="VALIDATION"
        )
        
        test_results = evaluate_model_with_threshold(
            type('Pipeline', (), {
                'predict_proba': lambda self, X: final_classifier.predict_proba(
                    preprocessing_pipeline.transform(X)
                )
            })(),
            X_test, y_test,
            threshold=optimal_threshold,
            dataset_name="TEST (F1-Optimal)"
        )
        
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
        
        # Display ROC-PR curves
        curves_path = os.path.join(out_dir, 'roc_pr_curves_no_leakage.png')
        print(f"\nDisplaying ROC-PR curves...")
        display_image(curves_path)
        
        print("\nGenerating feature importance plot...")
        
        dummy_pipeline = type('Pipeline', (), {
            'named_steps': {'classifier': final_classifier}
        })()
        
        importance_df = plot_feature_importance(
            dummy_pipeline,
            all_features,
            out_dir,
            top_n=min(20, len(all_features))
        )
        
        # Display feature importance
        importance_path = os.path.join(out_dir, 'feature_importance.png')
        print(f"\nDisplaying feature importance plot...")
        display_image(importance_path)
        
        # Save complete pipeline
        complete_pipeline = FraudDetectionPipeline(
            preprocessing_pipeline,
            final_classifier,
            threshold=optimal_threshold
        )
        
        model_path = os.path.join(out_dir, 'fraud_detection_smoteenn.pkl')
        try:
            joblib.dump(complete_pipeline, model_path)
            print(f"\nModel saved to {model_path}")
            print(f"Pipeline includes:")
            print(f"  - Preprocessor: {list(preprocessing_pipeline.named_steps.keys())}")
            print(f"  - Classifier: XGBoost")
            print(f"  - Optimal threshold: {optimal_threshold:.4f}")
        except Exception as e:
            print(f"\nWarning: Failed to save model: {e}")
            print("Continuing without saving model file...")
        
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
        
        # Convert best_params to native Python types
        best_params_json = {}
        for key, value in best_params.items():
            if isinstance(value, (np.integer, np.floating)):
                best_params_json[key] = float(value)
            else:
                best_params_json[key] = value
        
        # Save results to JSON
        results_json = {
            'mode': mode,
            'train': {k: v for k, v in train_results.items() if k not in ['y_proba', 'y_true']},
            'val': {k: v for k, v in val_results.items() if k not in ['y_proba', 'y_true']},
            'test_f1_optimal': {k: v for k, v in test_results.items() if k not in ['y_proba', 'y_true']},
            'training_time_sec': float(training_time),
            'val_test_gap': {
                'roc_auc': float(val_test_gap_roc),
                'pr_auc': float(val_test_gap_pr)
            },
            'threshold_strategies': {
                'f1_optimal': {
                    'threshold': float(optimal_threshold),
                    'val_metrics': threshold_metrics
                }
            },
            'feature_selection': {
                'total_features': int(total_features),
                'selected_count': int(total_features),
                'selected_features': all_features,
                'note': 'NO feature selection - using ALL features'
            },
            'xgboost_params': best_params_json
        }
        
        with open(os.path.join(out_dir, 'results_no_leakage.json'), 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n[INFO] Results saved to {out_dir}/results_no_leakage.json")
        
        # Print threshold strategies summary
        print("\n" + "="*60)
        print(" THRESHOLD STRATEGIES SUMMARY ")
        print("="*60)
        print("\n1. F1-Optimal Strategy (Balanced):")
        print(f"   Threshold: {optimal_threshold:.4f}")
        print(f"   Val Precision: {threshold_metrics['precision']:.4f}")
        print(f"   Val Recall: {threshold_metrics['recall']:.4f}")
        print(f"   Val F1-Score: {threshold_metrics['f1']:.4f}")
        
        print("="*60 + "\n")
        
        pipeline = complete_pipeline  # Return complete pipeline
    
    # ========================================
    # MODE: XGBoost + FA + SMOTEENN
    # ========================================
    
    elif mode == 'xgboost_fa_smoteenn':
        print("\n" + "="*80)
        print(" XGBOOST + FIREFLY ALGORITHM (NO SMOTE+ENN) ")
        print("="*80)
        
        print("\nFA Configuration:")
        print(f"  - Selection ratio: {fa_config.selection_ratio}")
        print(f"  - Selection mode: {fa_config.feature_selection_mode}")
        print(f"  - Min features: {fa_config.min_feature_count}")
        print(f"  - Random state: {fa_config.random_state}")
        
        start_time = time.time()
        
        print("\nCreating preprocessing pipeline...")
        pipeline_steps = [
            ('date_features', DateFeatureExtractor()),
            ('missing_handler', MissingValueHandler()),
            ('categorical_encoder', CategoricalEncoder()),
            ('scaler', StandardScaler()),
        ]
        
        from sklearn.pipeline import Pipeline as SkPipeline
        preprocessing_pipeline = SkPipeline(pipeline_steps)
        
        print("Running preprocessing on all sets...")
        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_val_processed = preprocessing_pipeline.transform(X_val)
        X_test_processed = preprocessing_pipeline.transform(X_test)
        
        # Get feature names after preprocessing
        if isinstance(X_train_processed, pd.DataFrame):
            all_features = X_train_processed.columns.tolist()
        else:
            all_features = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
        
        total_features = len(all_features)
        print(f"\nTotal features: {total_features}")
        
        print("\nRunning FA Feature Selection on imbalanced, encoded data (before resampling)...")
        
        if fa_config is None:
            fa_config = FAConfig()
        
        feature_selector = FeatureSelector(config=fa_config)
        
        # Convert to DataFrame if needed for FA
        if not isinstance(X_train_processed, pd.DataFrame):
            X_train_processed = pd.DataFrame(X_train_processed, columns=all_features)
        
        # Fit FA on ORIGINAL (imbalanced) training data
        X_train_selected = feature_selector.fit_transform(X_train_processed, y_train)
        selected_features = feature_selector.selected_features_
        
        print(f"\nSelected {len(selected_features)}/{total_features} features")
        print(f"Selected features: {selected_features[:10]}... (showing first 10)")
        
        print("\nApplying feature selection to Val and Test sets...")
        if not isinstance(X_val_processed, pd.DataFrame):
            X_val_processed = pd.DataFrame(X_val_processed, columns=all_features)
        if not isinstance(X_test_processed, pd.DataFrame):
            X_test_processed = pd.DataFrame(X_test_processed, columns=all_features)
        
        X_val_selected = X_val_processed[selected_features]
        X_test_selected = X_test_processed[selected_features]
        
        print(f"  - Train (selected): {X_train_selected.shape}")
        print(f"  - Val (selected): {X_val_selected.shape}")
        print(f"  - Test (selected): {X_test_selected.shape}")

        print("\n[INFO] Training on original imbalanced data (no SMOTE+ENN)")
        X_train_resampled, y_train_resampled = X_train_selected, y_train
        print(f"  - Train (selected): {X_train_resampled.shape}")
        print(f"  - Fraud rate: {y_train_resampled.mean():.4f}")

        # Ensure 2D arrays for XGBoost/Optuna
        X_train_resampled = np.asarray(X_train_resampled)
        X_val_selected = np.asarray(X_val_selected)
        X_test_selected = np.asarray(X_test_selected)
        if X_train_resampled.ndim == 1:
            X_train_resampled = X_train_resampled.reshape(-1, 1)
        if X_val_selected.ndim == 1:
            X_val_selected = X_val_selected.reshape(-1, 1)
        if X_test_selected.ndim == 1:
            X_test_selected = X_test_selected.reshape(-1, 1)

        y_train_resampled = np.asarray(y_train_resampled)
        
        # Check GPU availability
        gpu_available = False
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_available = True
                print("\n[INFO] 🚀 GPU detected for Optuna tuning!")
        except:
            print("\n[INFO] ⚠️  No GPU detected. Using CPU for Optuna tuning.")
        
        # STEP 4: Optuna hyperparameter tuning (on BALANCED + SELECTED features)
        print("\n[STEP 4] Running Optuna hyperparameter tuning...")
        best_params = tune_xgboost_with_optuna(
            X_train_resampled, y_train_resampled,  # Use BALANCED + SELECTED data
            X_val_selected, y_val,
            n_trials=50,
            use_gpu=gpu_available
        )
        
        # If Optuna failed, use default params
        if best_params is None:
            if gpu_available:
                best_params = {
                    'n_estimators': 250,
                    'learning_rate': 0.04,
                    'max_depth': 5,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 3.0,
                    'reg_lambda': 6.0,
                    'min_child_weight': 6.0,
                    'gamma': 0.0
                }
            else:
                best_params = {
                    'n_estimators': 250,
                    'learning_rate': 0.04,
                    'max_depth': 5,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 3.0,
                    'reg_lambda': 6.0,
                    'min_child_weight': 6.0,
                    'gamma': 0.0
                }
        
        # Add base params
        best_params['objective'] = 'binary:logistic'
        best_params['eval_metric'] = 'aucpr'
        best_params['random_state'] = 42
        best_params['verbosity'] = 0
        
        if gpu_available:
            best_params['tree_method'] = 'gpu_hist'
            best_params['device'] = 'cuda'
        else:
            best_params['tree_method'] = 'hist'
            best_params['device'] = 'cpu'
            best_params['n_jobs'] = -1
        
        # Calculate scale_pos_weight (from resampled data)
        pos = int(y_train_resampled.sum())
        neg = len(y_train_resampled) - pos
        scale_pos_weight = neg / pos
        best_params['scale_pos_weight'] = scale_pos_weight
        
        print(f"\nTraining final XGBoost with optimized params...")
        print(f"  - scale_pos_weight: {scale_pos_weight:.2f}")
        print(f"  - Training on balanced data with selected features")
        
        final_classifier = XGBClassifier(**best_params)
        final_classifier.fit(X_train_resampled, y_train_resampled)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        print("\nFinding optimal thresholds on validation set...")
        y_val_proba = final_classifier.predict_proba(X_val_selected)[:, 1]
        
        optimal_threshold, best_f1, threshold_metrics = find_optimal_threshold(
            y_val, y_val_proba
        )
        
        print(f"\nOptimal threshold found on validation set (F1-max):")
        print(f"  - Threshold: {optimal_threshold:.4f}")
        print(f"  - Val Precision: {threshold_metrics['precision']:.4f}")
        print(f"  - Val Recall: {threshold_metrics['recall']:.4f}")
        print(f"  - Val F1-Score: {threshold_metrics['f1']:.4f}")
        
        full_preprocessing = FullPreprocessingPipeline(preprocessing_pipeline, feature_selector)
        
        print("\n" + "="*60)
        print(" EVALUATION RESULTS ")
        print("="*60)
        
        train_results = evaluate_model_with_threshold(
            type('Pipeline', (), {
                'predict_proba': lambda self, X: final_classifier.predict_proba(
                    full_preprocessing.transform(X)
                )
            })(),
            X_train, y_train, 
            threshold=optimal_threshold,
            dataset_name="TRAIN"
        )
        
        val_results = evaluate_model_with_threshold(
            type('Pipeline', (), {
                'predict_proba': lambda self, X: final_classifier.predict_proba(
                    full_preprocessing.transform(X)
                )
            })(),
            X_val, y_val,
            threshold=optimal_threshold,
            dataset_name="VALIDATION"
        )
        
        test_results = evaluate_model_with_threshold(
            type('Pipeline', (), {
                'predict_proba': lambda self, X: final_classifier.predict_proba(
                    full_preprocessing.transform(X)
                )
            })(),
            X_test, y_test,
            threshold=optimal_threshold,
            dataset_name="TEST (F1-Optimal)"
        )
        
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
        
        # Display ROC-PR curves
        curves_path = os.path.join(out_dir, 'roc_pr_curves_no_leakage.png')
        print(f"\nDisplaying ROC-PR curves...")
        display_image(curves_path)
        
        print("\nGenerating feature importance plot...")
        
        # Create a dummy pipeline for feature importance plotting
        dummy_pipeline = type('Pipeline', (), {
            'named_steps': {'classifier': final_classifier}
        })()
        
        importance_df = plot_feature_importance(
            dummy_pipeline,
            selected_features,
            out_dir,
            top_n=min(20, len(selected_features))
        )
        
        # Display feature importance
        importance_path = os.path.join(out_dir, 'feature_importance.png')
        print(f"\nDisplaying feature importance plot...")
        display_image(importance_path)
        
        # Save complete pipeline (preprocessing + feature selection + classifier)
        # Create a wrapper for prediction that combines all components
        complete_pipeline = FraudDetectionPipeline(
            full_preprocessing,  # Use the combined preprocessing + feature selection
            final_classifier,
            threshold=optimal_threshold
        )
        
        model_path = os.path.join(out_dir, 'fraud_detection_fa_smoteenn.pkl')
        try:
            joblib.dump(complete_pipeline, model_path)
            print(f"\nModel saved to {model_path}")
            print(f"Pipeline includes:")
            print(f"  - Preprocessor: {list(preprocessing_pipeline.named_steps.keys())}")
            print(f"  - Feature Selector: FA ({len(selected_features)} features)")
            print(f"  - Classifier: XGBoost")
            print(f"  - Optimal threshold: {optimal_threshold:.4f}")
        except Exception as e:
            print(f"\nWarning: Failed to save model: {e}")
            print("Continuing without saving model file...")
        
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
        
        # Convert best_params to native Python types (for JSON serialization)
        best_params_json = {}
        for key, value in best_params.items():
            if isinstance(value, (np.integer, np.floating)):
                best_params_json[key] = float(value)
            else:
                best_params_json[key] = value
        
        # Save results to JSON
        results_json = {
            'mode': mode,
            'train': {k: v for k, v in train_results.items() if k not in ['y_proba', 'y_true']},
            'val': {k: v for k, v in val_results.items() if k not in ['y_proba', 'y_true']},
            'test_f1_optimal': {k: v for k, v in test_results.items() if k not in ['y_proba', 'y_true']},
            'training_time_sec': float(training_time),
            'val_test_gap': {
                'roc_auc': float(val_test_gap_roc),
                'pr_auc': float(val_test_gap_pr)
            },
            'threshold_strategies': {
                'f1_optimal': {
                    'threshold': float(optimal_threshold),
                    'val_metrics': threshold_metrics
                }
            },
            'feature_selection': {
                'fa_config': {
                    'selection_ratio': float(fa_config.selection_ratio),
                    'selection_mode': fa_config.feature_selection_mode,
                    'min_feature_ratio': float(fa_config.min_feature_ratio),
                    'max_feature_ratio': float(fa_config.max_feature_ratio),
                    'min_feature_count': int(fa_config.min_feature_count),
                    'random_state': int(fa_config.random_state)
                },
                'total_features': int(total_features),
                'selected_count': int(len(selected_features)),
                'selected_features': selected_features
            },
            'xgboost_params': best_params_json
        }
        
        with open(os.path.join(out_dir, 'results_no_leakage.json'), 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\nResults saved to {out_dir}/results_no_leakage.json")
        
        # Print threshold strategies summary
        print("\n" + "="*60)
        print(" THRESHOLD STRATEGIES SUMMARY ")
        print("="*60)
        print("\n1. F1-Optimal Strategy (Balanced):")
        print(f"   Threshold: {optimal_threshold:.4f}")
        print(f"   Val Precision: {threshold_metrics['precision']:.4f}")
        print(f"   Val Recall: {threshold_metrics['recall']:.4f}")
        print(f"   Val F1-Score: {threshold_metrics['f1']:.4f}")
        
        print("="*60 + "\n")
        
        pipeline = complete_pipeline
    
    else:
        print(f"\nERROR: Invalid mode '{mode}'")
        print("Valid modes: 'xgboost_smoteenn' or 'xgboost_fa_smoteenn'")
        return None
    
    print("\n" + "="*80)
    print(" COMPLETED ")
    print("="*80)
    
    return pipeline

if __name__ == "__main__":
    # pipeline = main(mode='xgboost_smoteenn')
    pipeline = main(mode='xgboost_fa_smoteenn')

