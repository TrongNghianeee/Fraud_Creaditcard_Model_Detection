!pip install xgboost==1.7.3
!pip install optuna==3.1.0


# =============================================================================
# FRAUD DETECTION - PSO + XGBOOST (NO SMOTEENN)
# 🎯 TARGET: 12-15 FEATURES | 10 FULL EPOCHS
# =============================================================================

import pandas as pd
import numpy as np
import time
import warnings
import os
import json
from typing import Tuple
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, f1_score, precision_score, recall_score, confusion_matrix
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Tắt cảnh báo
warnings.filterwarnings("ignore")

print("=============================================================================")
print("   FRAUD DETECTION - PSO + XGBOOST (NO SMOTEENN)                            ")
print("   🎯 TARGET: 12-15 FEATURES | 10 FULL EPOCHS                              ")
print("=============================================================================")

# =============================================================================
# PHẦN 1: CẤU HÌNH PSO VÀ HÀM LOAD DỮ LIỆU - UPDATED
# =============================================================================

@dataclass
class PSOConfig:
    """Configuration for Feature Selection using Particle Swarm Optimization"""
    
    # Feature selection parameters - UPDATED
    selection_ratio: float = 0.7           # 🎯 70% (target ~15 features from 21)
    min_feature_ratio: float = 0.6         # Tối thiểu 60%
    max_feature_ratio: float = 0.8         # Tối đa 80%
    min_feature_count: int = 12            # 🎯 TỐI THIỂU 12 đặc trưng
    
    # Random seed
    random_state: int = 42
    
    # PSO parameters - UPDATED TO RUN FULL 10 EPOCHS
    n_particles: int = 30                  # Số lượng hạt
    n_epochs: int = 10                     # 🎯 CHẠY ĐỦ 10 EPOCHS
    w_max: float = 0.9                     # Trọng số quán tính tối đa
    w_min: float = 0.4                     # Trọng số quán tính tối thiểu
    c1: float = 2.0                        # Hệ số nhận thức
    c2: float = 2.0                        # Hệ số xã hội
    lambda_feat: float = 0.005             # 🎯 HỆ SỐ PHẠT (giảm từ 0.01 → 0.005)
    patience: int = 10                     # 🎯 TẮT EARLY STOPPING (patience = epochs)

def load_raw_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load raw data without preprocessing to avoid data leakage."""
    print("[INFO] Loading RAW data from:", filepath)
    df = pd.read_csv(filepath)
    
    print("Initial shape:", df.shape)
    print("\nTarget distribution:\n", df['is_fraud'].value_counts(normalize=True))
    
    # Drop unnecessary columns
    cols_to_drop = ['index', 'Unnamed: 0', 'trans_num']
    df.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')
    
    # Separate target
    y = df['is_fraud']
    X = df.drop('is_fraud', axis=1)
    
    print("\nFeatures:", X.columns.tolist())
    print("Shape:", X.shape)
    
    return X, y

# =============================================================================
# PHẦN 2: CÁC LỚP XỬ LÝ DỮ LIỆU
# =============================================================================

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from datetime columns."""
    
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
    """Encode categorical features."""
    
    def __init__(self):
        self.label_encoders = {}
    
    def fit(self, X, y=None):
        X = X.copy()
        
        # Identify categorical columns
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Fit label encoders
        for col in cat_cols:
            le = LabelEncoder()
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
                X[col] = X[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else 'unknown'
                )
                X[col] = le.transform(X[col])
        
        return X

class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Handle missing values."""
    
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

# =============================================================================
# PHẦN 3: THUẬT TOÁN PSO CHO LỰA CHỌN ĐẶC TRƯNG - UPDATED
# =============================================================================

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Feature Selection using Particle Swarm Optimization (PSO)."""
    
    def __init__(self, config: PSOConfig = None):
        if config is None:
            config = PSOConfig()
        
        self.config = config
        self.selected_features_ = None
        self.feature_names_ = None
        self.best_fitness_ = -np.inf
        self.fitness_history_ = []
    
    def _initialize_swarm(self, n_features, target_n_features):
        """Initialize swarm of particles (binary vectors) and velocities."""
        particles = []
        velocities = []
        for _ in range(self.config.n_particles):
            # Random binary vector
            particle = np.random.rand(n_features) < (target_n_features / n_features)
            # Ensure minimum features
            if particle.sum() < self.config.min_feature_count:
                indices = np.random.choice(n_features, self.config.min_feature_count, replace=False)
                particle = np.zeros(n_features, dtype=bool)
                particle[indices] = True
            particles.append(particle.astype(float))
            # Initialize velocities
            velocities.append(np.random.uniform(-1, 1, n_features))
        return np.array(particles), np.array(velocities)
    
    def _calculate_fitness(self, particle, X, y):
        """Calculate fitness of a particle (feature subset)."""
        selected_indices = particle > 0.5
        n_selected = selected_indices.sum()
        
        # Check minimum features
        if n_selected < self.config.min_feature_count:
            return -1000.0
        
        # Get selected features
        X_selected = X[:, selected_indices] if not isinstance(X, pd.DataFrame) else X.iloc[:, selected_indices]
        
        # Quick validation using XGBoost
        clf = XGBClassifier(
            n_estimators=50, 
            max_depth=3, 
            learning_rate=0.1,
            random_state=self.config.random_state,
            n_jobs=1,
            verbosity=0,
            eval_metric='aucpr'
        )
        
        # Cross-validation score
        try:
            scores = cross_val_score(clf, X_selected, y, cv=3, scoring='average_precision', n_jobs=1)
            auc_score = scores.mean()
        except:
            auc_score = 0.0
        
        # Fitness = AUC - feature penalty
        fitness = auc_score - self.config.lambda_feat * (n_selected / len(particle))
        
        return fitness
    
    def fit(self, X, y=None):
        """Fit using PSO for feature selection - RUN FULL 10 EPOCHS."""
        np.random.seed(self.config.random_state)
        
        print("\n" + "="*60)
        print(" PARTICLE SWARM OPTIMIZATION - FEATURE SELECTION ")
        print(" 🎯 TARGET: 12-15 FEATURES | 10 FULL EPOCHS ")
        print("="*60)
        
        # Get feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_array = X.values
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X
        
        n_features = len(self.feature_names_)
        
        print(f"\n[INFO] Total features before selection: {n_features}")
        print(f"[INFO] Running PSO on data (fraud rate: {y.mean():.4f})")
        
        # Calculate target number of features
        min_features = max(self.config.min_feature_count, int(n_features * self.config.min_feature_ratio))
        max_features = int(n_features * self.config.max_feature_ratio)
        target_n_features = int(n_features * self.config.selection_ratio)
        target_n_features = max(min_features, min(target_n_features, max_features))
        
        print(f"\n[INFO] PSO Configuration:")
        print(f"  - Total features: {n_features}")
        print(f"  - Target features: {target_n_features} (minimum: {self.config.min_feature_count})")
        print(f"  - Particles: {self.config.n_particles}")
        print(f"  - Epochs: {self.config.n_epochs} (NO EARLY STOPPING)")
        print(f"  - Inertia (w): {self.config.w_max} to {self.config.w_min}")
        print(f"  - Lambda penalty: {self.config.lambda_feat}")
        
        print(f"\n[INFO] ⏳ Initializing {self.config.n_particles} particles...")
        
        # Initialize swarm
        particles, velocities = self._initialize_swarm(n_features, target_n_features)
        
        print(f"[INFO] 🔥 Evaluating initial swarm...")
        fitness_values = []
        for idx, p in enumerate(particles):
            fitness = self._calculate_fitness(p, X_array, y)
            fitness_values.append(fitness)
            if (idx + 1) % 10 == 0:
                print(f"       Progress: {idx+1}/{self.config.n_particles} particles evaluated")
        fitness_values = np.array(fitness_values)
        
        # Personal bests
        p_bests = particles.copy()
        p_best_fitness = fitness_values.copy()
        
        # Global best
        g_best_idx = np.argmax(fitness_values)
        g_best = particles[g_best_idx].copy()
        self.best_fitness_ = fitness_values[g_best_idx]
        
        print(f"\n[INFO] ✅ Initial swarm ready!")
        print(f"  - Best fitness: {self.best_fitness_:.4f}")
        print(f"  - Features selected: {int(g_best.sum())}")
        print(f"\n[INFO] 🚀 Starting PSO optimization ({self.config.n_epochs} FULL EPOCHS)...")
        
        no_improvement = 0
        
        # PSO iterations - RUN FULL 10 EPOCHS (NO EARLY STOPPING)
        for epoch in range(self.config.n_epochs):
            epoch_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"📍 EPOCH {epoch+1}/{self.config.n_epochs}")
            print(f"{'='*60}")
            
            # Update inertia weight
            w = self.config.w_max - (self.config.w_max - self.config.w_min) * (epoch / self.config.n_epochs)
            
            evaluations_count = 0
            
            # Update each particle
            for i in range(self.config.n_particles):
                r1 = np.random.rand(n_features)
                r2 = np.random.rand(n_features)
                
                # Update velocity
                velocities[i] = w * velocities[i] + \
                                self.config.c1 * r1 * (p_bests[i] - particles[i]) + \
                                self.config.c2 * r2 * (g_best - particles[i])
                
                # Update position (sigmoid for binary)
                particles[i] = 1 / (1 + np.exp(- (particles[i] + velocities[i])))
                particles[i] = (particles[i] > 0.5).astype(float)
                
                # Evaluate fitness
                fitness = self._calculate_fitness(particles[i], X_array, y)
                evaluations_count += 1
                
                # Update personal best
                if fitness > p_best_fitness[i]:
                    p_bests[i] = particles[i].copy()
                    p_best_fitness[i] = fitness
                
                if (i + 1) % 10 == 0:
                    print(f"  🔄 Processed {i+1}/{self.config.n_particles} particles ({evaluations_count} evaluations)")
            
            # Update global best
            g_best_idx = np.argmax(p_best_fitness)
            current_best_fitness = p_best_fitness[g_best_idx]
            
            epoch_time = time.time() - epoch_start_time
            
            if current_best_fitness > self.best_fitness_:
                improvement = current_best_fitness - self.best_fitness_
                self.best_fitness_ = current_best_fitness
                g_best = p_bests[g_best_idx].copy()
                no_improvement = 0
                print(f"\n  ✨ NEW BEST FITNESS: {self.best_fitness_:.4f} (+{improvement:.4f})")
                print(f"     Features: {int(g_best.sum())}/{n_features}")
                print(f"     Time: {epoch_time:.1f}s")
            else:
                no_improvement += 1
                print(f"\n  ⏸️  No improvement (fitness: {self.best_fitness_:.4f})")
                print(f"     Consecutive no-improvement: {no_improvement}")
                print(f"     Time: {epoch_time:.1f}s")
            
            self.fitness_history_.append(self.best_fitness_)
            
            # 🎯 REMOVED EARLY STOPPING - ALWAYS RUN FULL 10 EPOCHS
        
        print(f"\n{'='*60}")
        print(f"✅ COMPLETED ALL {self.config.n_epochs} EPOCHS!")
        print(f"{'='*60}")
        
        # Select features from global best
        selected_indices = g_best > 0.5
        self.selected_features_ = [self.feature_names_[i] for i in range(n_features) if selected_indices[i]]
        
        print(f"\n[INFO] ✅ Feature Selection Complete!")
        print(f"  - Final fitness: {self.best_fitness_:.4f}")
        print(f"  - Selected: {len(self.selected_features_)}/{n_features} features")
        print(f"  - Selection ratio: {len(self.selected_features_)/n_features:.1%}")
        print(f"  - Selected features: {self.selected_features_}")
        print("="*60 + "\n")
        
        return self
    
    def transform(self, X):
        """Transform data by selecting best features."""
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            # Convert to DataFrame for selection
            df = pd.DataFrame(X, columns=self.feature_names_)
            return df[self.selected_features_].values

# =============================================================================
# PHẦN 4: TỐI ƯU HÓA HYPERPARAMETER VỚI OPTUNA
# =============================================================================

def tune_xgboost_with_optuna(X_train, y_train, X_val, y_val, n_trials=50, use_gpu=False):
    """Tune XGBoost hyperparameters using Optuna."""
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        print("[WARNING] Optuna not installed. Using default hyperparameters.")
        return None
    
    print("\n" + "="*60)
    print(" OPTUNA - HYPERPARAMETER TUNING ")
    print("="*60)
    print(f"[INFO] Running {n_trials} trials...")
    
    def objective(trial):
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
    print(f"\n[INFO] ✅ Best PR-AUC: {study.best_value:.4f}")
    print("="*60 + "\n")
    
    return best_params

# =============================================================================
# PHẦN 5: TÌM NGƯỠNG TỐI ƯU VÀ ĐÁNH GIÁ
# =============================================================================

def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """Find optimal threshold."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    f1_scores = []
    for i in range(len(thresholds)):
        if precisions[i] + recalls[i] > 0:
            f1 = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
        else:
            f1 = 0.0
        f1_scores.append(f1)
    
    f1_scores = np.array(f1_scores)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    
    return float(optimal_threshold), float(f1_scores[best_idx]), {
        'precision': float(precisions[best_idx]),
        'recall': float(recalls[best_idx]),
        'f1': float(f1_scores[best_idx]),
        'threshold': float(optimal_threshold)
    }

def evaluate_model_with_threshold(pipeline, X, y, threshold=0.5, dataset_name="Dataset"):
    """Evaluate model with threshold."""
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
    
    return {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'y_proba': y_proba,
        'threshold': float(threshold)
    }

# =============================================================================
# PHẦN 6: VISUALIZATION
# =============================================================================

def plot_curves(results_dict, out_dir='outputs'):
    """Plot ROC and PR curves."""
    os.makedirs(out_dir, exist_ok=True)
    
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
    plt.savefig(os.path.join(out_dir, 'roc_pr_curves.png'), dpi=150)
    plt.close()
    
    print(f"[INFO] Saved curves to {out_dir}/roc_pr_curves.png")

def plot_feature_importance(pipeline, feature_names, out_dir='outputs', top_n=20):
    """Plot feature importance."""
    os.makedirs(out_dir, exist_ok=True)
    
    xgb_model = pipeline.named_steps['classifier'] if hasattr(pipeline, 'named_steps') else pipeline
    importance = xgb_model.feature_importances_
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_features)))
    plt.barh(range(len(top_features)), top_features['importance'].values, color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_importance.png'), dpi=150)
    plt.close()
    
    print(f"\n{'='*60}")
    print(f" TOP {min(top_n, len(feature_names))} FEATURES ")
    print(f"{'='*60}")
    for idx, row in top_features.iterrows():
        print(f"  {row['feature']:30s} : {row['importance']:.6f}")
    print(f"{'='*60}\n")
    
    return importance_df

# =============================================================================
# PHẦN 7: PIPELINE WRAPPER
# =============================================================================

class FraudDetectionPipeline:
    """Complete pipeline wrapper."""
    def __init__(self, preprocessor, classifier, threshold=0.5):
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.threshold = threshold
    
    def predict_proba(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.classifier.predict_proba(X_processed)
    
    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

# =============================================================================
# PHẦN 8: HÀM MAIN - PSO + XGBOOST (NO SMOTEENN)
# =============================================================================

def main():
    """Main training function - PSO + XGBoost (NO SMOTEENN)."""
    
    print("\n" + "="*80)
    print(" 🎯 WORKFLOW: PSO → XGBoost (NO SMOTEENN) ")
    print(" 🎯 TARGET: 12-15 FEATURES | 10 FULL EPOCHS ")
    print("="*80)
    
    out_dir = 'outputs_pso_xgboost'
    os.makedirs(out_dir, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load RAW Data
    # =========================================================================
    print("\n[STEP 1] Loading RAW data...")
    data_path = '/kaggle/input/fraud-detection/fraudTrain.csv'
    X, y = load_raw_data(data_path)
    
    # Config - 🎯 UPDATED VALUES
    pso_config = PSOConfig(
        selection_ratio=0.7,               # 🎯 70%
        min_feature_ratio=0.6,             # 🎯 60%
        max_feature_ratio=0.8,             # 🎯 80%
        min_feature_count=12,              # 🎯 MINIMUM 12 FEATURES
        n_particles=30,
        n_epochs=10,                       # 🎯 RUN FULL 10 EPOCHS
        patience=10,                       # 🎯 NO EARLY STOPPING
        lambda_feat=0.005,                 # 🎯 PENALTY 0.005
        random_state=42
    )
    
    # =========================================================================
    # STEP 2: Split Data (60% Train / 20% Val / 20% Test)
    # =========================================================================
    print("\n[STEP 2] Splitting data (60/20/20)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape}, fraud: {y_train.mean():.4f}")
    print(f"Val:   {X_val.shape}, fraud: {y_val.mean():.4f}")
    print(f"Test:  {X_test.shape}, fraud: {y_test.mean():.4f}")
    
    # =========================================================================
    # STEP 3: Preprocessing Pipeline (FIT on TRAIN only)
    # =========================================================================
    print("\n[STEP 3] Preprocessing (FIT on TRAIN only)...")
    preprocessing_pipeline = SkPipeline([
        ('date_features', DateFeatureExtractor()),
        ('missing_handler', MissingValueHandler()),
        ('categorical_encoder', CategoricalEncoder()),
        ('scaler', StandardScaler())
    ])
    
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_val_processed = preprocessing_pipeline.transform(X_val)
    X_test_processed = preprocessing_pipeline.transform(X_test)
    
    print(f"[INFO] After preprocessing: {X_train_processed.shape[1]} features")
    
    # =========================================================================
    # STEP 4: PSO Feature Selection (on IMBALANCED data)
    # =========================================================================
    print("\n[STEP 4] 🔥 PSO Feature Selection...")
    print(f"[INFO] Running PSO on IMBALANCED data (fraud rate: {y_train.mean():.4f})")
    
    start_time = time.time()
    
    # Convert to DataFrame for PSO
    feature_names = [f'feature_{i}' for i in range(X_train_processed.shape[1])]
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
    
    # Run PSO
    feature_selector = FeatureSelector(config=pso_config)
    X_train_selected = feature_selector.fit_transform(X_train_processed_df, y_train)
    
    selected_features = feature_selector.selected_features_
    
    print(f"\n[INFO] ✅ PSO completed!")
    print(f"  - Selected {len(selected_features)} features from {len(feature_names)}")
    print(f"  - Selected features: {selected_features}")
    
    # =========================================================================
    # STEP 5: Apply Feature Selection to All Sets
    # =========================================================================
    print("\n[STEP 5] Applying feature selection to Val/Test...")
    X_val_df = pd.DataFrame(X_val_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    
    X_train_selected = feature_selector.transform(X_train_processed_df)
    X_val_selected = feature_selector.transform(X_val_df)
    X_test_selected = feature_selector.transform(X_test_df)
    
    print(f"[INFO] Train: {X_train_selected.shape}")
    print(f"[INFO] Val:   {X_val_selected.shape}")
    print(f"[INFO] Test:  {X_test_selected.shape}")
    
    # =========================================================================
    # STEP 6: GPU Check
    # =========================================================================
    gpu_available = False
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        gpu_available = (result.returncode == 0)
        if gpu_available:
            print("\n[INFO] 🚀 GPU detected!")
    except:
        print("\n[INFO] No GPU, using CPU")
    
    # =========================================================================
    # STEP 7: Optuna Hyperparameter Tuning
    # =========================================================================
    print("\n[STEP 6] Hyperparameter tuning with Optuna (50 trials)...")
    best_params = tune_xgboost_with_optuna(
        X_train_selected,       # IMBALANCED data (no SMOTEENN)
        y_train, 
        X_val_selected,         # IMBALANCED validation
        y_val, 
        n_trials=50, 
        use_gpu=gpu_available
    )
    
    if best_params is None:
        print("[WARNING] Using default parameters")
        best_params = {
            'n_estimators': 250, 
            'learning_rate': 0.04, 
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1.0,
            'reg_lambda': 10.0,
            'min_child_weight': 3.0,
            'gamma': 0.5
        }
    
    # Add fixed parameters
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'random_state': 42,
        'verbosity': 0,
        'tree_method': 'gpu_hist' if gpu_available else 'hist',
        'device': 'cuda' if gpu_available else 'cpu'
    })
    
    if not gpu_available:
        best_params['n_jobs'] = -1
    
    # =========================================================================
    # STEP 8: Train Final XGBoost Model
    # =========================================================================
    print("\n[STEP 7] Training final XGBoost model...")
    
    # Calculate scale_pos_weight for imbalanced data
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    best_params['scale_pos_weight'] = neg / pos
    
    print(f"[INFO] scale_pos_weight: {best_params['scale_pos_weight']:.4f}")
    
    final_classifier = XGBClassifier(**best_params)
    final_classifier.fit(X_train_selected, y_train)
    
    training_time = time.time() - start_time
    print(f"[INFO] ✅ Training completed in {training_time:.2f}s")
    
    # =========================================================================
    # STEP 9: Threshold Optimization (on Validation)
    # =========================================================================
    print("\n[STEP 8] Threshold optimization on validation set...")
    y_val_proba = final_classifier.predict_proba(X_val_selected)[:, 1]
    optimal_threshold, best_f1, threshold_metrics = find_optimal_threshold(
        y_val, y_val_proba, metric='f1'
    )
    
    print(f"[INFO] Optimal threshold: {optimal_threshold:.4f}")
    print(f"[INFO] Val F1-Score: {threshold_metrics['f1']:.4f}")
    print(f"[INFO] Val Precision: {threshold_metrics['precision']:.4f}")
    print(f"[INFO] Val Recall: {threshold_metrics['recall']:.4f}")
    
    # =========================================================================
    # STEP 10: Final Evaluation
    # =========================================================================
    print("\n[STEP 9] Final evaluation on all sets...")
    
    # Create full preprocessing pipeline
    full_preprocessing_pipeline = SkPipeline([
        ('date_features', preprocessing_pipeline.named_steps['date_features']),
        ('missing_handler', preprocessing_pipeline.named_steps['missing_handler']),
        ('categorical_encoder', preprocessing_pipeline.named_steps['categorical_encoder']),
        ('scaler', preprocessing_pipeline.named_steps['scaler']),
        ('feature_selector', feature_selector)
    ])
    
    # Wrapper for evaluation
    class EvaluationWrapper:
        def __init__(self, preprocessor, classifier):
            self.preprocessor = preprocessor
            self.classifier = classifier
        
        def predict_proba(self, X):
            X_processed = self.preprocessor.transform(X)
            return self.classifier.predict_proba(X_processed)
    
    eval_pipeline = EvaluationWrapper(full_preprocessing_pipeline, final_classifier)
    
    # Evaluate on all sets
    train_results = evaluate_model_with_threshold(
        eval_pipeline, X_train, y_train, optimal_threshold, "TRAIN"
    )
    val_results = evaluate_model_with_threshold(
        eval_pipeline, X_val, y_val, optimal_threshold, "VAL"
    )
    test_results = evaluate_model_with_threshold(
        eval_pipeline, X_test, y_test, optimal_threshold, "TEST"
    )
    
    # Add ground truth for plotting
    train_results['y_true'] = y_train
    val_results['y_true'] = y_val
    test_results['y_true'] = y_test
    
    # =========================================================================
    # STEP 11: Visualizations
    # =========================================================================
    print("\n[STEP 10] Generating visualizations...")
    
    # ROC & PR Curves
    plot_curves({
        'Train': train_results, 
        'Val': val_results, 
        'Test': test_results
    }, out_dir)
    
    # Feature Importance
    importance_wrapper = type('Pipeline', (), {
        'named_steps': {'classifier': final_classifier}
    })()
    importance_df = plot_feature_importance(
        importance_wrapper, 
        selected_features, 
        out_dir, 
        top_n=len(selected_features)
    )
    
    # Confusion Matrix
    y_test_pred = (final_classifier.predict_proba(X_test_selected)[:, 1] >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_test, y_test_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Fraud', 'Fraud'], 
                yticklabels=['Not Fraud', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Test Set Confusion Matrix (Threshold={optimal_threshold:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    print(f"[INFO] Confusion matrix saved to {out_dir}/confusion_matrix.png")
    
    # PSO Fitness History
    if hasattr(feature_selector, 'fitness_history_') and len(feature_selector.fitness_history_) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(feature_selector.fitness_history_) + 1), 
                 feature_selector.fitness_history_, 
                 marker='o', linewidth=2, markersize=8)
        plt.xlabel('Epoch')
        plt.ylabel('Best Fitness')
        plt.title('PSO Convergence (10 Epochs)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'pso_convergence.png'), dpi=150)
        plt.close()
        print(f"[INFO] PSO convergence plot saved to {out_dir}/pso_convergence.png")
    
    # =========================================================================
    # STEP 12: Save Model & Results
    # =========================================================================
    print("\n[STEP 11] Saving model and results...")
    
    # Save complete pipeline
    complete_pipeline = FraudDetectionPipeline(
        full_preprocessing_pipeline, 
        final_classifier, 
        optimal_threshold
    )
    joblib.dump(complete_pipeline, os.path.join(out_dir, 'fraud_detection_model.pkl'))
    print(f"[INFO] Model saved to {out_dir}/fraud_detection_model.pkl")
    
    # Save results to JSON
    results_json = {
        'workflow': 'PSO + XGBoost (NO SMOTEENN)',
        'workflow_steps': [
            '1. Load RAW data',
            '2. Split (60/20/20)',
            '3. Preprocessing (FIT on train)',
            '4. PSO Feature Selection (on IMBALANCED train)',
            '5. Apply selection to Val/Test',
            '6. Optuna tuning (IMBALANCED train, IMBALANCED val)',
            '7. Train XGBoost (IMBALANCED train with scale_pos_weight)',
            '8. Threshold optimization (IMBALANCED val)',
            '9. Final evaluation'
        ],
        'pso_config': {
            'target_features': '12-15',
            'selection_ratio': pso_config.selection_ratio,
            'min_feature_ratio': pso_config.min_feature_ratio,
            'max_feature_ratio': pso_config.max_feature_ratio,
            'min_feature_count': pso_config.min_feature_count,
            'epochs': pso_config.n_epochs,
            'particles': pso_config.n_particles,
            'lambda_feat': pso_config.lambda_feat,
            'no_early_stopping': True,
            'ran_on': 'IMBALANCED data'
        },
        'data_splits': {
            'train': {
                'shape': list(X_train.shape),
                'fraud_rate': float(y_train.mean())
            },
            'val': {
                'shape': list(X_val.shape),
                'fraud_rate': float(y_val.mean())
            },
            'test': {
                'shape': list(X_test.shape),
                'fraud_rate': float(y_test.mean())
            }
        },
        'selected_features': selected_features,
        'n_features_selected': len(selected_features),
        'best_xgboost_params': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                                for k, v in best_params.items()},
        'threshold': float(optimal_threshold),
        'train_metrics': {k: float(v) for k, v in train_results.items() 
                         if k not in ['y_proba', 'y_true']},
        'val_metrics': {k: float(v) for k, v in val_results.items() 
                       if k not in ['y_proba', 'y_true']},
        'test_metrics': {k: float(v) for k, v in test_results.items() 
                        if k not in ['y_proba', 'y_true']},
        'training_time_sec': float(training_time),
        'overfitting_check': {
            'train_test_f1_gap': float(train_results['f1'] - test_results['f1']),
            'train_test_auc_gap': float(train_results['roc_auc'] - test_results['roc_auc']),
            'val_test_f1_gap': float(val_results['f1'] - test_results['f1'])
        }
    }
    
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"[INFO] Results saved to {out_dir}/results.json")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print(" ✅ TRAINING COMPLETED!")
    print("="*80)
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Selected Features: {len(selected_features)}/{len(feature_names)}")
    print(f"   PSO Epochs: {pso_config.n_epochs} (FULL, no early stopping)")
    print(f"   Training Time: {training_time:.2f}s")
    print(f"   Optimal Threshold: {optimal_threshold:.4f}")
    print(f"\n   TEST SET PERFORMANCE:")
    print(f"   - ROC-AUC:   {test_results['roc_auc']:.4f}")
    print(f"   - PR-AUC:    {test_results['pr_auc']:.4f}")
    print(f"   - Precision: {test_results['precision']:.4f}")
    print(f"   - Recall:    {test_results['recall']:.4f}")
    print(f"   - F1-Score:  {test_results['f1']:.4f}")
    print(f"\n   OVERFITTING CHECK:")
    print(f"   - Train-Test F1 gap: {train_results['f1'] - test_results['f1']:.4f}")
    print(f"   - Val-Test F1 gap:   {val_results['f1'] - test_results['f1']:.4f}")
    print(f"\n📁 All outputs saved to: {out_dir}/")
    print("="*80 + "\n")
    
    return complete_pipeline

# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    pipeline = main()