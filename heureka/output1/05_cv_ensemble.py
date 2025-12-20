#!/usr/bin/env python3
"""
OpenADMET ExpansionRx Blind Challenge - Cross-Validation Ensemble
Purpose: Use proper cross-validation with all data for more robust predictions
         Special handling for difficult properties (HLM CLint)
Date: 2025-12-19
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import pickle
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 80)
print("CROSS-VALIDATION ENSEMBLE MODEL")
print("=" * 80)

# Load data
train_features = pd.read_pickle('train_features.pkl')
test_features = pd.read_pickle('test_features.pkl')
train_targets = pd.read_pickle('train_targets.pkl')

target_cols = ['LogD', 'KSOL', 'HLM CLint', 'MLM CLint', 'Caco-2 Permeability Papp A>B',
               'Caco-2 Permeability Efflux', 'MPPB', 'MBPB', 'MGMB']

# Remove constant features
const_cols = train_features.columns[train_features.std() == 0].tolist()
train_features = train_features.drop(columns=const_cols)
test_features = test_features.drop(columns=const_cols)

# Handle NaN
train_features = train_features.replace([np.inf, -np.inf], np.nan)
test_features = test_features.replace([np.inf, -np.inf], np.nan)
train_median = train_features.median()
train_features = train_features.fillna(train_median)
test_features = test_features.fillna(train_median)

X_train = train_features.values
X_test = test_features.values

feature_cols = train_features.columns.tolist()
rdkit_cols = [c for c in feature_cols if c.startswith('rdkit_')]
morgan2_cols = [c for c in feature_cols if c.startswith('morgan2_')]
morgan3_cols = [c for c in feature_cols if c.startswith('morgan3_')]
maccs_cols = [c for c in feature_cols if c.startswith('maccs_')]
smiles_cols = [c for c in feature_cols if c.startswith('smiles_')]

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")

def compute_metrics(y_true, y_pred):
    """Compute Spearman and MA-RAE"""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() < 10:
        return np.nan, np.nan

    y_true_c = y_true[mask]
    y_pred_c = y_pred[mask]

    spearman, _ = stats.spearmanr(y_true_c, y_pred_c)
    mae = np.mean(np.abs(y_true_c - y_pred_c))
    mad = np.mean(np.abs(y_true_c - np.mean(y_true_c)))
    ma_rae = mae / mad if mad > 1e-8 else np.nan

    return spearman, ma_rae


# Configuration for each target (property-specific tuning)
target_configs = {
    'LogD': [
        {'num_leaves': 63, 'learning_rate': 0.05, 'max_depth': 10, 'feature_fraction': 0.8},
        {'num_leaves': 127, 'learning_rate': 0.03, 'max_depth': 12, 'feature_fraction': 0.7},
        {'num_leaves': 31, 'learning_rate': 0.08, 'max_depth': 8, 'feature_fraction': 0.9},
    ],
    'KSOL': [
        {'num_leaves': 63, 'learning_rate': 0.05, 'max_depth': 10, 'feature_fraction': 0.8},
        {'num_leaves': 47, 'learning_rate': 0.04, 'max_depth': 9, 'feature_fraction': 0.75},
    ],
    'HLM CLint': [  # Problematic - use log transform and more regularization
        {'num_leaves': 31, 'learning_rate': 0.03, 'max_depth': 6, 'feature_fraction': 0.6, 'min_child_samples': 50},
        {'num_leaves': 15, 'learning_rate': 0.02, 'max_depth': 5, 'feature_fraction': 0.5, 'min_child_samples': 100},
        {'num_leaves': 47, 'learning_rate': 0.01, 'max_depth': 7, 'feature_fraction': 0.7, 'min_child_samples': 30},
    ],
    'MLM CLint': [
        {'num_leaves': 63, 'learning_rate': 0.04, 'max_depth': 10, 'feature_fraction': 0.8},
        {'num_leaves': 31, 'learning_rate': 0.06, 'max_depth': 8, 'feature_fraction': 0.75},
    ],
    'Caco-2 Permeability Papp A>B': [
        {'num_leaves': 47, 'learning_rate': 0.04, 'max_depth': 8, 'feature_fraction': 0.75},
        {'num_leaves': 31, 'learning_rate': 0.05, 'max_depth': 7, 'feature_fraction': 0.8},
    ],
    'Caco-2 Permeability Efflux': [
        {'num_leaves': 63, 'learning_rate': 0.04, 'max_depth': 10, 'feature_fraction': 0.8},
        {'num_leaves': 47, 'learning_rate': 0.05, 'max_depth': 8, 'feature_fraction': 0.75},
    ],
    'MPPB': [
        {'num_leaves': 63, 'learning_rate': 0.04, 'max_depth': 10, 'feature_fraction': 0.8},
        {'num_leaves': 47, 'learning_rate': 0.05, 'max_depth': 8, 'feature_fraction': 0.85},
    ],
    'MBPB': [
        {'num_leaves': 47, 'learning_rate': 0.04, 'max_depth': 8, 'feature_fraction': 0.75},
        {'num_leaves': 31, 'learning_rate': 0.05, 'max_depth': 7, 'feature_fraction': 0.8},
    ],
    'MGMB': [  # Very limited data
        {'num_leaves': 15, 'learning_rate': 0.03, 'max_depth': 5, 'feature_fraction': 0.6, 'min_child_samples': 10},
        {'num_leaves': 7, 'learning_rate': 0.02, 'max_depth': 4, 'feature_fraction': 0.5, 'min_child_samples': 5},
    ],
}


def get_base_params():
    return {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1,
    }


def train_cv_model(X, y, params, n_folds=5, n_rounds=500):
    """Train with cross-validation, return OOF predictions and test predictions"""
    valid_mask = ~np.isnan(y)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    if len(y_valid) < 50:
        return None, None, None

    oof_predictions = np.full(len(y_valid), np.nan)
    test_predictions = []
    fold_models = []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_valid)):
        X_tr, X_val = X_valid[train_idx], X_valid[val_idx]
        y_tr, y_val = y_valid[train_idx], y_valid[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_rounds,
            valid_sets=[val_data],
            callbacks=callbacks
        )

        oof_predictions[val_idx] = model.predict(X_val)
        test_predictions.append(model.predict(X_test))
        fold_models.append(model)

    # Average test predictions across folds
    test_pred = np.mean(test_predictions, axis=0)

    return oof_predictions, test_pred, fold_models


print("\n" + "=" * 80)
print("TRAINING CV ENSEMBLE FOR EACH TARGET")
print("=" * 80)

all_cv_preds = {}
all_test_preds = {}
all_scalers = {}
all_models = {}
results = []

for target in target_cols:
    print(f"\n{'='*60}")
    print(f"Target: {target}")
    print(f"{'='*60}")

    y = train_targets[target].values
    n_valid = (~np.isnan(y)).sum()
    print(f"  Available samples: {n_valid}")

    if n_valid < 50:
        print(f"  SKIPPING: Insufficient data")
        all_test_preds[target] = np.full(len(X_test), train_targets[target].median())
        continue

    # Z-score normalization
    y_mean = np.nanmean(y)
    y_std = np.nanstd(y)
    if y_std < 1e-8:
        y_std = 1.0
    all_scalers[target] = {'mean': y_mean, 'std': y_std}
    y_scaled = (y - y_mean) / y_std

    # Special handling for HLM CLint - use log transform
    use_log = target in ['HLM CLint', 'MLM CLint', 'Caco-2 Permeability Efflux']
    if use_log:
        # Log-transform positive values
        y_log = np.log1p(np.maximum(y, 0))
        y_log_mean = np.nanmean(y_log)
        y_log_std = np.nanstd(y_log)
        if y_log_std < 1e-8:
            y_log_std = 1.0
        y_for_training = (y_log - y_log_mean) / y_log_std
        all_scalers[target]['log_mean'] = y_log_mean
        all_scalers[target]['log_std'] = y_log_std
        all_scalers[target]['use_log'] = True
    else:
        y_for_training = y_scaled
        all_scalers[target]['use_log'] = False

    # Get configs for this target
    configs = target_configs.get(target, [{'num_leaves': 63, 'learning_rate': 0.05, 'max_depth': 10, 'feature_fraction': 0.8}])

    model_oof_preds = []
    model_test_preds = []
    target_models = []

    for i, config in enumerate(configs):
        params = get_base_params()
        params.update(config)

        # Train with CV
        oof_pred, test_pred, fold_models = train_cv_model(X_train, y_for_training, params)

        if oof_pred is None:
            continue

        # Denormalize
        if use_log:
            oof_pred_denorm = np.expm1(oof_pred * y_log_std + y_log_mean)
            test_pred_denorm = np.expm1(test_pred * y_log_std + y_log_mean)
        else:
            oof_pred_denorm = oof_pred * y_std + y_mean
            test_pred_denorm = test_pred * y_std + y_mean

        model_oof_preds.append(oof_pred_denorm)
        model_test_preds.append(test_pred_denorm)
        target_models.append(fold_models)

        # Compute metrics on valid subset
        valid_mask = ~np.isnan(y)
        spearman, ma_rae = compute_metrics(y[valid_mask], oof_pred_denorm)
        print(f"    Config {i+1}: Spearman={spearman:.4f}, MA-RAE={ma_rae:.4f}")

    if len(model_oof_preds) == 0:
        print("  No valid models")
        all_test_preds[target] = np.full(len(X_test), y_mean)
        continue

    all_models[target] = target_models

    # Ensemble predictions
    oof_array = np.column_stack(model_oof_preds)
    test_array = np.column_stack(model_test_preds)

    # Simple average
    avg_oof = np.mean(oof_array, axis=1)
    avg_test = np.mean(test_array, axis=1)

    valid_mask = ~np.isnan(y)
    spearman_avg, ma_rae_avg = compute_metrics(y[valid_mask], avg_oof)
    print(f"  Average ensemble: Spearman={spearman_avg:.4f}, MA-RAE={ma_rae_avg:.4f}")

    # Weighted average by performance
    weights = []
    for pred in model_oof_preds:
        sp, _ = compute_metrics(y[valid_mask], pred)
        weights.append(max(0.01, sp if not np.isnan(sp) else 0.01))
    weights = np.array(weights) / sum(weights)

    weighted_oof = np.average(oof_array, axis=1, weights=weights)
    weighted_test = np.average(test_array, axis=1, weights=weights)
    spearman_wt, ma_rae_wt = compute_metrics(y[valid_mask], weighted_oof)
    print(f"  Weighted ensemble: Spearman={spearman_wt:.4f}, MA-RAE={ma_rae_wt:.4f}")

    # Use best ensemble
    if spearman_wt >= spearman_avg:
        final_oof = weighted_oof
        final_test = weighted_test
        best_spearman = spearman_wt
        best_ma_rae = ma_rae_wt
    else:
        final_oof = avg_oof
        final_test = avg_test
        best_spearman = spearman_avg
        best_ma_rae = ma_rae_avg

    all_cv_preds[target] = final_oof
    all_test_preds[target] = final_test

    results.append({
        'target': target,
        'n_samples': n_valid,
        'spearman': best_spearman,
        'ma_rae': best_ma_rae
    })

# Summary
print("\n" + "=" * 80)
print("CV ENSEMBLE RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

avg_spearman = results_df['spearman'].mean()
avg_ma_rae = results_df['ma_rae'].mean()
print(f"\nAverage Spearman: {avg_spearman:.4f}")
print(f"Average MA-RAE: {avg_ma_rae:.4f}")

# Weighted averages
weights = results_df['n_samples'].values
weighted_spearman = np.average(results_df['spearman'].values, weights=weights)
weighted_ma_rae = np.average(results_df['ma_rae'].values, weights=weights)
print(f"\nWeighted avg Spearman: {weighted_spearman:.4f}")
print(f"Weighted avg MA-RAE: {weighted_ma_rae:.4f}")

# Create predictions dataframe
print("\n" + "=" * 80)
print("SAVING PREDICTIONS")
print("=" * 80)

test_mol_info = pd.read_csv('test_mol_info.csv')
pred_df = test_mol_info.copy()

for target in target_cols:
    pred_df[target] = all_test_preds[target]

# Ensure column order
train_original = pd.read_csv('expansion_data_train.csv')
pred_df = pred_df[train_original.columns]

# Save
pred_df.to_csv('cv_ensemble_predictions.csv', index=False)
print("Saved: cv_ensemble_predictions.csv")

pred_df.to_csv('admet_predictions_v3.csv', index=False)
print("Saved: admet_predictions_v3.csv")

results_df.to_csv('cv_ensemble_results.csv', index=False)
print("Saved: cv_ensemble_results.csv")

with open('cv_ensemble_models.pkl', 'wb') as f:
    pickle.dump({'models': all_models, 'scalers': all_scalers}, f)
print("Saved: cv_ensemble_models.pkl")

print("\n" + "=" * 80)
print("CV ENSEMBLE COMPLETE")
print("=" * 80)
