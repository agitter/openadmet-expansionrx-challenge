#!/usr/bin/env python3
"""
OpenADMET ExpansionRx Blind Challenge - Baseline LightGBM Models
Purpose: Train single-task LightGBM models per property with hyperparameter tuning
         Establish baseline performance metrics
Date: 2025-12-19
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("=" * 80)
print("BASELINE LIGHTGBM MODELS")
print("=" * 80)

# Load processed data
print("\nLoading processed data...")
train_features = pd.read_pickle('train_features.pkl')
test_features = pd.read_pickle('test_features.pkl')
train_targets = pd.read_pickle('train_targets.pkl')
train_indices = np.load('train_indices.npy')
val_indices = np.load('val_indices.npy')

target_cols = ['LogD', 'KSOL', 'HLM CLint', 'MLM CLint', 'Caco-2 Permeability Papp A>B',
               'Caco-2 Permeability Efflux', 'MPPB', 'MBPB', 'MGMB']

print(f"Training features shape: {train_features.shape}")
print(f"Test features shape: {test_features.shape}")
print(f"Train split: {len(train_indices)}, Val split: {len(val_indices)}")

# Remove constant features
print("\nRemoving constant features...")
const_cols = train_features.columns[train_features.std() == 0].tolist()
train_features = train_features.drop(columns=const_cols)
test_features = test_features.drop(columns=const_cols)
print(f"Removed {len(const_cols)} constant features")
print(f"Final feature count: {train_features.shape[1]}")

# Handle inf values
train_features = train_features.replace([np.inf, -np.inf], np.nan)
test_features = test_features.replace([np.inf, -np.inf], np.nan)

# Fill NaN with median
train_median = train_features.median()
train_features = train_features.fillna(train_median)
test_features = test_features.fillna(train_median)

# Split data
X_train_full = train_features.values
X_train = X_train_full[train_indices]
X_val = X_train_full[val_indices]
X_test = test_features.values

print(f"\nTrain set: {X_train.shape}")
print(f"Val set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# Define evaluation metrics
def compute_metrics(y_true, y_pred):
    """Compute Spearman correlation and MA-RAE"""
    # Remove NaN
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() < 10:
        return np.nan, np.nan

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Spearman correlation
    spearman, _ = stats.spearmanr(y_true, y_pred)

    # MA-RAE (Mean Absolute Relative Absolute Error)
    # MA-RAE = mean(|y_true - y_pred| / |y_true - median(y_true)|)
    median_y = np.median(y_true)
    denom = np.abs(y_true - median_y)
    denom = np.where(denom < 1e-8, 1e-8, denom)  # Avoid division by zero
    ma_rae = np.mean(np.abs(y_true - y_pred) / denom)

    return spearman, ma_rae


# Base LightGBM parameters (optimized for molecular property prediction)
base_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'max_depth': 10,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'seed': 42,
    'n_jobs': -1,
}

# Train models for each property
print("\n" + "=" * 80)
print("TRAINING SINGLE-TASK MODELS")
print("=" * 80)

models = {}
scalers = {}
results = []
val_predictions = {}
test_predictions = {}

for target in target_cols:
    print(f"\n{'='*60}")
    print(f"Training model for: {target}")
    print(f"{'='*60}")

    # Get target values
    y_full = train_targets[target].values
    y_train = y_full[train_indices]
    y_val = y_full[val_indices]

    # Check data availability
    train_mask = ~np.isnan(y_train)
    val_mask = ~np.isnan(y_val)

    n_train = train_mask.sum()
    n_val = val_mask.sum()

    print(f"  Training samples: {n_train} / {len(y_train)}")
    print(f"  Validation samples: {n_val} / {len(y_val)}")

    if n_train < 50:
        print(f"  SKIPPING: Insufficient training data")
        models[target] = None
        scalers[target] = None
        continue

    # Z-score normalize target
    train_mean = np.nanmean(y_train)
    train_std = np.nanstd(y_train)
    if train_std < 1e-8:
        train_std = 1.0

    scalers[target] = {'mean': train_mean, 'std': train_std}

    y_train_scaled = (y_train - train_mean) / train_std
    y_val_scaled = (y_val - train_mean) / train_std

    # Create datasets (only non-NaN values)
    X_train_target = X_train[train_mask]
    y_train_target = y_train_scaled[train_mask]

    X_val_target = X_val[val_mask]
    y_val_target = y_val_scaled[val_mask]

    train_data = lgb.Dataset(X_train_target, label=y_train_target)

    if n_val >= 20:
        val_data = lgb.Dataset(X_val_target, label=y_val_target, reference=train_data)
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model = lgb.train(
            base_params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=callbacks
        )
    else:
        model = lgb.train(
            base_params,
            train_data,
            num_boost_round=300
        )

    models[target] = model
    print(f"  Best iteration: {model.best_iteration if hasattr(model, 'best_iteration') else 'N/A'}")

    # Predict on validation set
    val_pred_scaled = model.predict(X_val)
    val_pred = val_pred_scaled * train_std + train_mean
    val_predictions[target] = val_pred

    # Compute validation metrics
    spearman, ma_rae = compute_metrics(y_val, val_pred)
    print(f"  Validation Spearman: {spearman:.4f}")
    print(f"  Validation MA-RAE: {ma_rae:.4f}")

    results.append({
        'target': target,
        'n_train': n_train,
        'n_val': n_val,
        'spearman': spearman,
        'ma_rae': ma_rae,
        'best_iter': model.best_iteration if hasattr(model, 'best_iteration') else 0
    })

    # Predict on test set
    test_pred_scaled = model.predict(X_test)
    test_pred = test_pred_scaled * train_std + train_mean
    test_predictions[target] = test_pred

# Summary results
print("\n" + "=" * 80)
print("BASELINE MODEL RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Calculate weighted average Spearman (weighted by n_val)
valid_results = results_df.dropna()
if len(valid_results) > 0:
    weights = valid_results['n_val'].values
    weighted_spearman = np.average(valid_results['spearman'].values, weights=weights)
    weighted_ma_rae = np.average(valid_results['ma_rae'].values, weights=weights)
    print(f"\nWeighted avg Spearman: {weighted_spearman:.4f}")
    print(f"Weighted avg MA-RAE: {weighted_ma_rae:.4f}")

    # Simple average
    avg_spearman = valid_results['spearman'].mean()
    avg_ma_rae = valid_results['ma_rae'].mean()
    print(f"\nSimple avg Spearman: {avg_spearman:.4f}")
    print(f"Simple avg MA-RAE: {avg_ma_rae:.4f}")

# Save models
print("\n" + "=" * 80)
print("SAVING MODELS AND PREDICTIONS")
print("=" * 80)

# Save models
with open('baseline_models.pkl', 'wb') as f:
    pickle.dump({'models': models, 'scalers': scalers}, f)
print("Saved: baseline_models.pkl")

# Save validation predictions
val_pred_df = pd.DataFrame(val_predictions)
val_pred_df.to_csv('baseline_val_predictions.csv', index=False)
print("Saved: baseline_val_predictions.csv")

# Create test predictions dataframe
test_mol_info = pd.read_csv('test_mol_info.csv')
test_pred_df = test_mol_info.copy()

for target in target_cols:
    if target in test_predictions:
        test_pred_df[target] = test_predictions[target]
    else:
        # For missing models, use training median
        test_pred_df[target] = train_targets[target].median()

# Ensure column order matches training data
train_original = pd.read_csv('expansion_data_train.csv')
test_pred_df = test_pred_df[train_original.columns]

# Save first baseline predictions
test_pred_df.to_csv('baseline_predictions.csv', index=False)
print("Saved: baseline_predictions.csv")

# Also save as potential final predictions
test_pred_df.to_csv('admet_predictions_v1.csv', index=False)
print("Saved: admet_predictions_v1.csv")

# Save results summary
results_df.to_csv('baseline_results.csv', index=False)
print("Saved: baseline_results.csv")

print("\n" + "=" * 80)
print("BASELINE TRAINING COMPLETE")
print("=" * 80)
