#!/usr/bin/env python3
"""
Step 3: Model Development & Training
Train XGBoost regression models for 9 ADMET properties with appropriate transformations.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
import json
import os
from pathlib import Path
import time

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
BASE_DIR = Path('/app/sandbox/session_20251217_085238_bf1de403d101')
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'results' / 'models'
RESULTS_DIR = BASE_DIR / 'results'

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("STEP 3: MODEL DEVELOPMENT & TRAINING")
print("="*80)
print()

# Load featurized training data
print("Loading training data...")
train_df = pd.read_csv(DATA_DIR / 'train_featurized.csv')
print(f"Training data shape: {train_df.shape}")
print()

# Define target columns mapping (short name -> actual column name in CSV)
TARGET_MAPPING = {
    'LogD': 'LogD',
    'KSol': 'KSol',
    'MLM': 'MLM CLint',
    'HLM': 'HLM CLint',
    'Peff': 'Caco-2 Permeability Efflux',
    'Papp': 'Caco-2 Permeability Papp A>B',
    'MPPB': 'MPPB',
    'MBPB': 'MBPB',
    'MGMB': 'MGMB'
}

# Define which targets need log transformation (skewed positive properties)
LOG_TRANSFORM_TARGETS = ['KSol', 'MLM', 'HLM', 'Peff', 'Papp']

# Identify feature columns (fingerprints and descriptors)
feature_cols = [c for c in train_df.columns if c.startswith('fp_') or c.startswith('desc_')]
print(f"Number of features: {len(feature_cols)}")
print(f"  - Fingerprint features: {len([c for c in feature_cols if c.startswith('fp_')])}")
print(f"  - Descriptor features: {len([c for c in feature_cols if c.startswith('desc_')])}")
print()

# XGBoost hyperparameters
xgb_params = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_jobs': -1,
    'random_state': 42,
    'tree_method': 'hist'  # Faster training
}

print("XGBoost Parameters:")
for k, v in xgb_params.items():
    print(f"  {k}: {v}")
print()


def calculate_ma_rae(y_true, y_pred):
    """
    Calculate Mean Absolute Relative Error (MA-RAE).
    MA-RAE = mean(|y_true - y_pred| / |y_true|)
    """
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan

    relative_errors = np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])
    return np.mean(relative_errors)


def train_model_with_cv(target_short_name, target_col_name, X, y, n_folds=5):
    """
    Train XGBoost model with cross-validation.

    Parameters:
    -----------
    target_short_name : str
        Short name for the target (e.g., 'LogD', 'KSol')
    target_col_name : str
        Actual column name in the dataframe
    X : DataFrame
        Feature matrix
    y : Series
        Target values (original scale)
    n_folds : int
        Number of cross-validation folds

    Returns:
    --------
    dict : Dictionary containing model, CV scores, and training info
    """
    print(f"\n{'='*80}")
    print(f"Training model for: {target_short_name} ({target_col_name})")
    print(f"{'='*80}")

    # Check if transformation is needed
    needs_log_transform = target_short_name in LOG_TRANSFORM_TARGETS

    # Apply transformation if needed
    if needs_log_transform:
        y_transformed = np.log1p(y)  # log(1 + x)
        print(f"Applied log1p transformation")
        print(f"  Original scale - Mean: {y.mean():.3f}, Std: {y.std():.3f}, Range: [{y.min():.3f}, {y.max():.3f}]")
        print(f"  Transformed scale - Mean: {y_transformed.mean():.3f}, Std: {y_transformed.std():.3f}, Range: [{y_transformed.min():.3f}, {y_transformed.max():.3f}]")
    else:
        y_transformed = y.copy()
        print(f"No transformation applied (original scale)")
        print(f"  Mean: {y.mean():.3f}, Std: {y.std():.3f}, Range: [{y.min():.3f}, {y.max():.3f}]")

    print(f"Training samples: {len(y)}")
    print(f"Features: {X.shape[1]}")
    print()

    # Cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_mae_scores = []

    print(f"Performing {n_folds}-Fold Cross-Validation...")
    fold_start = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        # Split data
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y_transformed.iloc[train_idx], y_transformed.iloc[val_idx]
        y_val_original = y.iloc[val_idx]  # Keep original scale for metric calculation

        # Train model
        model_fold = xgb.XGBRegressor(**xgb_params)
        model_fold.fit(X_train_fold, y_train_fold, verbose=False)

        # Predict on validation set (transformed scale)
        y_val_pred_transformed = model_fold.predict(X_val_fold)

        # Inverse transform predictions if needed
        if needs_log_transform:
            y_val_pred_original = np.expm1(y_val_pred_transformed)  # exp(x) - 1
        else:
            y_val_pred_original = y_val_pred_transformed

        # Calculate MA-RAE on original scale
        ma_rae = calculate_ma_rae(y_val_original.values, y_val_pred_original)
        cv_mae_scores.append(ma_rae)

        # Print progress every fold
        print(f"  Fold {fold_idx}/{n_folds}: MA-RAE = {ma_rae:.4f}")

    fold_time = time.time() - fold_start
    print(f"\nCross-Validation completed in {fold_time:.1f}s")
    print(f"  Mean MA-RAE: {np.mean(cv_mae_scores):.4f} ± {np.std(cv_mae_scores):.4f}")

    # Train final model on all available data
    print(f"\nTraining final model on all {len(y)} samples...")
    final_model = xgb.XGBRegressor(**xgb_params)
    final_model.fit(X, y_transformed, verbose=False)

    # Save model using booster's save method
    model_path = MODEL_DIR / f'model_{target_short_name}.json'
    final_model.get_booster().save_model(str(model_path))
    print(f"Model saved to: {model_path}")

    # Return results
    return {
        'target_short_name': target_short_name,
        'target_col_name': target_col_name,
        'model': final_model,
        'model_path': str(model_path),
        'cv_ma_rae_mean': np.mean(cv_mae_scores),
        'cv_ma_rae_std': np.std(cv_mae_scores),
        'training_samples': len(y),
        'needs_log_transform': needs_log_transform,
        'n_features': X.shape[1]
    }


# Train models for all targets
print("\n" + "="*80)
print("TRAINING MODELS FOR ALL 9 ADMET PROPERTIES")
print("="*80)

results = []
overall_start = time.time()

for target_short, target_col in TARGET_MAPPING.items():
    # Filter to samples with non-null target values
    valid_mask = train_df[target_col].notna()
    n_valid = valid_mask.sum()

    if n_valid == 0:
        print(f"\nSKIPPING {target_short}: No valid samples")
        continue

    # Prepare data
    X_target = train_df.loc[valid_mask, feature_cols].reset_index(drop=True)
    y_target = train_df.loc[valid_mask, target_col].reset_index(drop=True)

    # Train model
    result = train_model_with_cv(target_short, target_col, X_target, y_target)
    results.append(result)

overall_time = time.time() - overall_start

print("\n" + "="*80)
print(f"ALL MODELS TRAINED SUCCESSFULLY in {overall_time:.1f}s")
print("="*80)

# Create performance summary
print("\n" + "="*80)
print("MODEL PERFORMANCE SUMMARY")
print("="*80)
print()

performance_df = pd.DataFrame([
    {
        'Target': r['target_short_name'],
        'CV_MA_RAE_Mean': r['cv_ma_rae_mean'],
        'CV_MA_RAE_Std': r['cv_ma_rae_std'],
        'Training_Samples': r['training_samples']
    }
    for r in results
])

print(performance_df.to_string(index=False))
print()

# Save performance CSV
perf_path = RESULTS_DIR / 'model_performance.csv'
performance_df.to_csv(perf_path, index=False)
print(f"Performance report saved to: {perf_path}")
print()

# Generate text summary
summary_lines = [
    "="*80,
    "MODEL TRAINING SUMMARY",
    "="*80,
    "",
    f"Training Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    f"Total Training Time: {overall_time:.1f} seconds",
    "",
    "Models Trained: 9 ADMET Properties",
    "Algorithm: XGBoost Regressor",
    f"Hyperparameters: {xgb_params}",
    "",
    "="*80,
    "PERFORMANCE METRICS (5-Fold Cross-Validation)",
    "="*80,
    "",
]

# Add table
summary_lines.append(f"{'Target':<10} {'MA-RAE Mean':>12} {'MA-RAE Std':>12} {'Samples':>10}")
summary_lines.append("-" * 50)
for _, row in performance_df.iterrows():
    summary_lines.append(
        f"{row['Target']:<10} {row['CV_MA_RAE_Mean']:>12.4f} {row['CV_MA_RAE_Std']:>12.4f} {row['Training_Samples']:>10.0f}"
    )

summary_lines.append("")
summary_lines.append("="*80)
summary_lines.append("STATISTICAL SUMMARY")
summary_lines.append("="*80)
summary_lines.append(f"Average MA-RAE across all targets: {performance_df['CV_MA_RAE_Mean'].mean():.4f}")
summary_lines.append(f"Best performing target: {performance_df.loc[performance_df['CV_MA_RAE_Mean'].idxmin(), 'Target']} "
                     f"(MA-RAE = {performance_df['CV_MA_RAE_Mean'].min():.4f})")
summary_lines.append(f"Worst performing target: {performance_df.loc[performance_df['CV_MA_RAE_Mean'].idxmax(), 'Target']} "
                     f"(MA-RAE = {performance_df['CV_MA_RAE_Mean'].max():.4f})")
summary_lines.append("")

# Identify high error targets (MA-RAE > 0.5)
high_error = performance_df[performance_df['CV_MA_RAE_Mean'] > 0.5]
if len(high_error) > 0:
    summary_lines.append("="*80)
    summary_lines.append("TARGETS WITH HIGH ERROR RATES (MA-RAE > 0.5)")
    summary_lines.append("="*80)
    for _, row in high_error.iterrows():
        summary_lines.append(f"  - {row['Target']}: MA-RAE = {row['CV_MA_RAE_Mean']:.4f}")
        # Get the corresponding result
        res = next(r for r in results if r['target_short_name'] == row['Target'])
        if res['needs_log_transform']:
            summary_lines.append(f"    (Log-transformed target, {res['training_samples']} samples)")
        else:
            summary_lines.append(f"    (Original scale, {res['training_samples']} samples)")
    summary_lines.append("")
else:
    summary_lines.append("All targets achieved MA-RAE < 0.5 (good performance)")
    summary_lines.append("")

# Data sparsity note
summary_lines.append("="*80)
summary_lines.append("DATA AVAILABILITY")
summary_lines.append("="*80)
sparse_targets = performance_df[performance_df['Training_Samples'] < 1000]
if len(sparse_targets) > 0:
    summary_lines.append("Targets with limited data (< 1000 samples):")
    for _, row in sparse_targets.iterrows():
        summary_lines.append(f"  - {row['Target']}: {row['Training_Samples']:.0f} samples")
else:
    summary_lines.append("All targets have adequate sample sizes (>= 1000 samples)")
summary_lines.append("")

# Target transformations
summary_lines.append("="*80)
summary_lines.append("TARGET TRANSFORMATIONS APPLIED")
summary_lines.append("="*80)
summary_lines.append("Log-transformed targets (log1p):")
for target in LOG_TRANSFORM_TARGETS:
    if target in performance_df['Target'].values:
        summary_lines.append(f"  - {target}")
summary_lines.append("")
summary_lines.append("Original scale targets:")
for target in TARGET_MAPPING.keys():
    if target not in LOG_TRANSFORM_TARGETS and target in performance_df['Target'].values:
        summary_lines.append(f"  - {target}")
summary_lines.append("")

summary_lines.append("="*80)
summary_lines.append("OUTPUT FILES")
summary_lines.append("="*80)
summary_lines.append(f"Models saved to: {MODEL_DIR}/")
for r in results:
    summary_lines.append(f"  - model_{r['target_short_name']}.json")
summary_lines.append("")
summary_lines.append(f"Performance metrics: {perf_path}")
summary_lines.append("="*80)

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

# Save summary
summary_path = RESULTS_DIR / 'training_summary.txt'
with open(summary_path, 'w') as f:
    f.write(summary_text)
print(f"\nSummary report saved to: {summary_path}")

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE")
print("="*80)
print(f"\nModels saved: {len(results)}/9")
print(f"Performance report: {perf_path}")
print(f"Training summary: {summary_path}")
