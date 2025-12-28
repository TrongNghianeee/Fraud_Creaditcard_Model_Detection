import os

# Limit BLAS/OpenMP threads early (must be set before numpy/scikit-learn imports)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
import subprocess
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from xgboost import XGBClassifier

import joblib

warnings.filterwarnings("ignore")


def find_optimal_threshold(y_true, y_proba):
    """Find F1-maximizing threshold on a validation set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # thresholds has length N, precisions/recalls have length N+1
    if len(thresholds) == 0:
        return 0.5, 0.0, {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    f1_scores = []
    for i in range(len(thresholds)):
        p = float(precisions[i])
        r = float(recalls[i])
        f1 = 0.0 if (p + r) == 0 else (2 * p * r) / (p + r)
        f1_scores.append(f1)

    f1_scores = np.asarray(f1_scores, dtype=float)
    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    metrics = {
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "f1": best_f1,
    }
    return optimal_threshold, best_f1, metrics


def load_raw_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load raw data without preprocessing (avoid leakage)."""
    print("Loading data from:", filepath)
    df = pd.read_csv(filepath)

    print("Initial shape:", df.shape)
    if "is_fraud" not in df.columns:
        raise ValueError("Missing target column 'is_fraud' in input CSV")

    print("\nTarget distribution:\n", df["is_fraud"].value_counts(normalize=True))

    cols_to_drop = ["index", "Unnamed: 0", "trans_num"]
    df.drop(cols_to_drop, axis=1, inplace=True, errors="ignore")

    y = df["is_fraud"]
    X = df.drop("is_fraud", axis=1)

    print("\nFeatures:", X.columns.tolist())
    print("Shape:", X.shape)

    return X, y


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from datetime columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["trans_date_trans_time"] = pd.to_datetime(X["trans_date_trans_time"], errors="coerce")
        X["dob"] = pd.to_datetime(X["dob"], errors="coerce")

        X["transaction_hour"] = X["trans_date_trans_time"].dt.hour
        X["transaction_day"] = X["trans_date_trans_time"].dt.dayofweek
        X["transaction_month"] = X["trans_date_trans_time"].dt.month
        X["age"] = (X["trans_date_trans_time"] - X["dob"]).dt.days // 365

        X.drop(["trans_date_trans_time", "dob", "unix_time"], axis=1, inplace=True, errors="ignore")
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features using LabelEncoder per column."""

    def __init__(self):
        self.label_encoders = {}

    def fit(self, X, y=None):
        X = X.copy()
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = X[col].fillna("unknown")
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col, le in self.label_encoders.items():
            if col in X.columns:
                X[col] = X[col].fillna("unknown")
                X[col] = X[col].astype(str).apply(lambda x: x if x in le.classes_ else "unknown")
                X[col] = le.transform(X[col])
        return X


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Fill numeric missing values with median learned from training set."""

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


def _detect_gpu() -> bool:
    """Best-effort detection of NVIDIA GPU (for XGBoost gpu_hist)."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def tune_xgboost_with_optuna(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials=50,
    use_gpu=False,
    scale_pos_weight=None,
):
    """Optuna tuning for PR-AUC. Returns best_params or None if optuna missing.

    If scale_pos_weight is provided (float), it will be used as a fixed param.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        print("[WARNING] Optuna not installed. Skipping hyperparameter tuning.")
        return None

    print("\n" + "=" * 60)
    print(" OPTUNA HYPERPARAMETER TUNING (PR-AUC) ")
    print("=" * 60)
    print(f"Running {n_trials} trials...")

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": 42,
            "verbosity": 0,
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 20.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }

        if scale_pos_weight is not None:
            params["scale_pos_weight"] = float(scale_pos_weight)

        if use_gpu:
            params["tree_method"] = "gpu_hist"
            params["device"] = "cuda"
        else:
            params["tree_method"] = "hist"
            params["device"] = "cpu"
            params["n_jobs"] = -1

        clf = XGBClassifier(**params)
        clf.fit(X_train, y_train)
        y_val_proba = clf.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_val_proba)
        return pr_auc

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\nOptuna tuning complete!")
    print(f"  Best PR-AUC: {study.best_value:.4f}")
    print("  Best parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("=" * 60 + "\n")

    return study.best_params


def evaluate_model_with_threshold(model, X, y, threshold=0.5, dataset_name="Dataset"):
    """Evaluate model with a given threshold (threshold may be optimized elsewhere)."""
    y_proba = model.predict_proba(X)[:, 1]
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
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
    }


class FraudDetectionPipeline:
    """Preprocessor + classifier wrapper."""

    def __init__(self, preprocessor, classifier, threshold=0.5):
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.threshold = float(threshold)

    def predict_proba(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.classifier.predict_proba(X_processed)

    def predict(self, X, threshold=None):
        thr = self.threshold if threshold is None else float(threshold)
        proba = self.predict_proba(X)[:, 1]
        return (proba >= thr).astype(int)


def _train_one(
    *,
    preprocessing_pipeline,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    mode_label: str,
    use_optuna: bool,
    n_trials: int,
    gpu_available: bool,
    use_scale_pos_weight: bool,
    threshold_strategy: str,
    fixed_threshold: float,
    out_dir: str,
):
    """Train one experiment variant and save model + metrics."""
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_val_processed = preprocessing_pipeline.transform(X_val)
    X_test_processed = preprocessing_pipeline.transform(X_test)

    scale_pos_weight = None
    if use_scale_pos_weight:
        pos = float(np.sum(y_train))
        neg = float(len(y_train) - np.sum(y_train))
        if pos > 0:
            scale_pos_weight = neg / pos
        else:
            scale_pos_weight = 1.0

    if use_optuna:
        best_params = tune_xgboost_with_optuna(
            X_train_processed,
            y_train,
            X_val_processed,
            y_val,
            n_trials=n_trials,
            use_gpu=gpu_available,
            scale_pos_weight=scale_pos_weight,
        )
    else:
        best_params = None

    if best_params is None:
        best_params = {
            "n_estimators": 250,
            "learning_rate": 0.04,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 3.0,
            "reg_lambda": 6.0,
            "min_child_weight": 6.0,
            "gamma": 0.0,
        }

    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "aucpr"
    best_params["random_state"] = 42
    best_params["verbosity"] = 0

    if gpu_available:
        best_params["tree_method"] = "gpu_hist"
        best_params["device"] = "cuda"
    else:
        best_params["tree_method"] = "hist"
        best_params["device"] = "cpu"
        best_params["n_jobs"] = -1

    if scale_pos_weight is not None:
        best_params["scale_pos_weight"] = float(scale_pos_weight)

    print(f"\nTraining final XGBoost: {mode_label}")
    if scale_pos_weight is None:
        print("  - scale_pos_weight: NOT USED")
    else:
        print(f"  - scale_pos_weight: {scale_pos_weight:.4f}")

    final_classifier = XGBClassifier(**best_params)
    final_classifier.fit(X_train_processed, y_train)

    threshold_strategy = str(threshold_strategy).lower().strip()
    if threshold_strategy not in {"auto", "fixed", "f1_optimal"}:
        raise ValueError("threshold_strategy must be one of: auto, fixed, f1_optimal")

    resolved_threshold_strategy = threshold_strategy
    if threshold_strategy == "auto":
        # Default behavior:
        # - no_optuna runs usually want a stable 0.5 threshold
        # - optuna runs commonly want best F1 threshold on validation
        resolved_threshold_strategy = "f1_optimal" if use_optuna else "fixed"

    fixed_threshold = float(fixed_threshold)
    if not (0.0 < fixed_threshold < 1.0):
        raise ValueError("fixed_threshold must be in (0, 1)")

    best_f1 = None
    threshold_metrics = None
    if resolved_threshold_strategy == "f1_optimal":
        y_val_proba = final_classifier.predict_proba(X_val_processed)[:, 1]
        optimal_threshold, best_f1, threshold_metrics = find_optimal_threshold(y_val, y_val_proba)

        print("\nOptimal threshold found on validation set (F1-max):")
        print(f"  - Threshold: {optimal_threshold:.4f}")
        print(f"  - Val Precision: {threshold_metrics['precision']:.4f}")
        print(f"  - Val Recall: {threshold_metrics['recall']:.4f}")
        print(f"  - Val F1-Score: {threshold_metrics['f1']:.4f}")
    else:
        optimal_threshold = fixed_threshold
        print("\nUsing FIXED threshold (no optimization):")
        print(f"  - Threshold: {optimal_threshold:.4f}")

    pipeline = FraudDetectionPipeline(preprocessing_pipeline, final_classifier, threshold=optimal_threshold)

    print("\n" + "=" * 60)
    print(f" EVALUATION RESULTS ({mode_label}) ")
    print("=" * 60)

    train_metrics = evaluate_model_with_threshold(pipeline, X_train, y_train, threshold=optimal_threshold, dataset_name="TRAIN")
    val_metrics = evaluate_model_with_threshold(pipeline, X_val, y_val, threshold=optimal_threshold, dataset_name="VALIDATION")
    test_metrics = evaluate_model_with_threshold(pipeline, X_test, y_test, threshold=optimal_threshold, dataset_name="TEST")

    model_path = os.path.join(out_dir, f"fraud_xgboost_{mode_label}.pkl")
    try:
        joblib.dump(pipeline, model_path)
        print(f"\n[INFO] Model saved to {model_path}")
    except Exception as e:
        print(f"\n[WARNING] Failed to save model: {e}")

    best_params_json = {}
    for key, value in best_params.items():
        if isinstance(value, (np.integer, np.floating)):
            best_params_json[key] = float(value)
        else:
            best_params_json[key] = value

    threshold_block = (
        {
            "threshold_strategies": {
                "f1_optimal": {
                    "threshold": float(optimal_threshold),
                    "val_metrics": threshold_metrics,
                    "best_f1": float(best_f1),
                }
            }
        }
        if resolved_threshold_strategy == "f1_optimal"
        else {
            "threshold_strategies": {
                "fixed": {
                    "threshold": float(optimal_threshold),
                    "val_metrics": None,
                    "best_f1": None,
                }
            }
        }
    )

    results_json = {
        "mode": mode_label,
        **threshold_block,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "xgboost_params": best_params_json,
        "notes": {
            "scale_pos_weight": "USED" if use_scale_pos_weight else "NOT USED",
            "threshold_strategy": resolved_threshold_strategy,
            "fixed_threshold": float(fixed_threshold),
            "optuna": "ENABLED" if use_optuna else "DISABLED",
        },
    }

    results_path = os.path.join(out_dir, f"results_{mode_label}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    print(f"\n[INFO] Results saved to {results_path}")
    return pipeline, results_json


def main(
    mode: str,
    data_path: str,
    n_trials: int = 50,
    threshold_strategy: str = "auto",
    fixed_threshold: float = 0.5,
    out_dir: str = "outputs_xgboost",
):
    """Train/evaluate XGBoost for comparisons.

    - xgboost_no_optuna: uses scale_pos_weight + threshold optimization (F1-max)
    - xgboost_optuna: runs 2 variants to compare imbalance handling:
        (a) with scale_pos_weight
        (b) without scale_pos_weight
      Both variants use Optuna tuning + threshold optimization.
    """

    print("\n" + "=" * 80)
    print(" FRAUD DETECTION - XGBOOST (COMPARISON RUNNER) ")
    print("=" * 80)
    print(f"Mode: {mode}")
    print("=" * 80 + "\n")

    os.makedirs(out_dir, exist_ok=True)

    X, y = load_raw_data(data_path)

    print("\nSplitting data (Train 60% / Val 20% / Test 20%)")
    print("=" * 60)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    print(f"Train: {X_train.shape}, fraud rate: {y_train.mean():.4f}")
    print(f"Val:   {X_val.shape}, fraud rate: {y_val.mean():.4f}")
    print(f"Test:  {X_test.shape}, fraud rate: {y_test.mean():.4f}")

    print("\nCreating preprocessing pipeline...")
    preprocessing_pipeline = SkPipeline(
        [
            ("date_features", DateFeatureExtractor()),
            ("missing_handler", MissingValueHandler()),
            ("categorical_encoder", CategoricalEncoder()),
            ("scaler", StandardScaler()),
        ]
    )

    start_time = time.time()

    gpu_available = _detect_gpu()
    if gpu_available:
        print("\n[INFO] GPU detected (nvidia-smi ok)")
    else:
        print("\n[INFO] No GPU detected - using CPU")

    pipelines = {}
    results = {}

    if mode == "xgboost_no_optuna":
        label = "xgboost_no_optuna_spw"
        pipe, res = _train_one(
            preprocessing_pipeline=preprocessing_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            mode_label=label,
            use_optuna=False,
            n_trials=n_trials,
            gpu_available=gpu_available,
            use_scale_pos_weight=True,
            threshold_strategy=threshold_strategy,
            fixed_threshold=fixed_threshold,
            out_dir=out_dir,
        )
        pipelines[label] = pipe
        results[label] = res

    elif mode == "xgboost_optuna":
        # Two variants for imbalance comparison under Optuna tuning
        for label, use_spw in [
            ("xgboost_optuna_spw", True),
            ("xgboost_optuna_no_spw", False),
        ]:
            pipe, res = _train_one(
                preprocessing_pipeline=preprocessing_pipeline,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                mode_label=label,
                use_optuna=True,
                n_trials=n_trials,
                gpu_available=gpu_available,
                use_scale_pos_weight=use_spw,
                threshold_strategy=threshold_strategy,
                fixed_threshold=fixed_threshold,
                out_dir=out_dir,
            )
            pipelines[label] = pipe
            results[label] = res
    else:
        raise ValueError("mode must be 'xgboost_optuna' or 'xgboost_no_optuna'")

    training_time = time.time() - start_time
    print(f"\n[INFO] Total run time: {training_time:.2f} seconds")
    return pipelines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="xgboost_no_optuna",
        choices=["xgboost_no_optuna", "xgboost_optuna"],
        help=(
            "xgboost_no_optuna: fixed hyperparams + scale_pos_weight + threshold optimize; "
            "xgboost_optuna: Optuna tuning and runs both with/without scale_pos_weight"
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/kaggle/input/fraud-detection/fraudTrain.csv",
        help="Path to CSV containing 'is_fraud'",
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument(
        "--threshold-strategy",
        type=str,
        default="auto",
        choices=["auto", "fixed", "f1_optimal"],
        help=(
            "auto: no_optuna uses fixed 0.5; optuna uses f1_optimal on validation. "
            "fixed: always use --fixed-threshold. "
            "f1_optimal: optimize threshold for max F1 on validation"
        ),
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.5,
        help="Threshold used when threshold-strategy is fixed (must be between 0 and 1)",
    )
    parser.add_argument("--out-dir", type=str, default="outputs_xgboost")

    # Jupyter/Colab kernels often pass extra args like: -f <kernel.json>
    # Use parse_known_args to ignore them.
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[INFO] Ignoring unknown argv: {unknown}")

    main(
        mode=args.mode,
        data_path=args.data_path,
        n_trials=args.n_trials,
        threshold_strategy=args.threshold_strategy,
        fixed_threshold=args.fixed_threshold,
        out_dir=args.out_dir,
    )
