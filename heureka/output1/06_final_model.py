#!/usr/bin/env python3
"""
OpenADMET ExpansionRx Blind Challenge - Final Model
Purpose: Create final ensemble with all optimizations and generate submission file
Date: 2025-12-19

Final model strategy:
1. CV ensemble with property-specific configurations
2. Log-transform for highly skewed properties
3. Multiple LightGBM configurations for diversity
4. Weighted ensemble based on CV performance
5. Proper output format validation
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import KFold
import pickle
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 80)
print("FINAL MODEL - OPTIMIZED ENSEMBLE")
print("=" * 80)

# Load data
train_features = pd.read_pickle('train_features.pkl')
test_features = pd.read_pickle('test_features.pkl')
train_targets = pd.read_pickle('train_targets.pkl')
train_original = pd.read_csv('expansion_data_train.csv')
test_original = pd.read_csv('expansion_data_test_blinded.csv')

target_cols = ['LogD', 'KSOL', 'HLM CLint', 'MLM CLint', 'Caco-2 Permeability Papp A>B',
               'Caco-2 Permeability Efflux', 'MPPB', 'MBPB', 'MGMB']

# Preprocess features
const_cols = train_features.columns[train_features.std() == 0].tolist()
train_features = train_features.drop(columns=const_cols)
test_features = test_features.drop(columns=const_cols)

train_features = train_features.replace([np.inf, -np.inf], np.nan)
test_features = test_features.replace([np.inf, -np.inf], np.nan)
train_median = train_features.median()
train_features = train_features.fillna(train_median)
test_features = test_features.fillna(train_median)

X_train = train_features.values
X_test = test_features.values

print(f"Training: {X_train.shape}")
print(f"Test: {X_test.shape}")


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


# Optimized configurations based on CV results
final_configs = {
    'LogD': {
        'use_log': False,
        'configs': [
            {'num_leaves': 31, 'learning_rate': 0.08, 'max_depth': 8, 'feature_fraction': 0.9, 'min_child_samples': 20},
            {'num_leaves': 63, 'learning_rate': 0.05, 'max_depth': 10, 'feature_fraction': 0.8, 'min_child_samples': 20},
            {'num_leaves': 127, 'learning_rate': 0.03, 'max_depth': 12, 'feature_fraction': 0.7, 'min_child_samples': 15},
        ]
    },
    'KSOL': {
        'use_log': False,
        'configs': [
            {'num_leaves': 47, 'learning_rate': 0.04, 'max_depth': 9, 'feature_fraction': 0.75, 'min_child_samples': 25},
            {'num_leaves': 63, 'learning_rate': 0.05, 'max_depth': 10, 'feature_fraction': 0.8, 'min_child_samples': 20},
            {'num_leaves': 31, 'learning_rate': 0.06, 'max_depth': 8, 'feature_fraction': 0.85, 'min_child_samples': 30},
        ]
    },
    'HLM CLint': {
        'use_log': True,
        'configs': [
            {'num_leaves': 31, 'learning_rate': 0.03, 'max_depth': 6, 'feature_fraction': 0.6, 'min_child_samples': 50},
            {'num_leaves': 47, 'learning_rate': 0.02, 'max_depth': 7, 'feature_fraction': 0.7, 'min_child_samples': 30},
            {'num_leaves': 63, 'learning_rate': 0.04, 'max_depth': 8, 'feature_fraction': 0.65, 'min_child_samples': 40},
        ]
    },
    'MLM CLint': {
        'use_log': True,
        'configs': [
            {'num_leaves': 63, 'learning_rate': 0.04, 'max_depth': 10, 'feature_fraction': 0.8, 'min_child_samples': 20},
            {'num_leaves': 47, 'learning_rate': 0.05, 'max_depth': 9, 'feature_fraction': 0.75, 'min_child_samples': 25},
            {'num_leaves': 31, 'learning_rate': 0.06, 'max_depth': 8, 'feature_fraction': 0.85, 'min_child_samples': 30},
        ]
    },
    'Caco-2 Permeability Papp A>B': {
        'use_log': False,
        'configs': [
            {'num_leaves': 47, 'learning_rate': 0.04, 'max_depth': 8, 'feature_fraction': 0.75, 'min_child_samples': 25},
            {'num_leaves': 31, 'learning_rate': 0.05, 'max_depth': 7, 'feature_fraction': 0.8, 'min_child_samples': 30},
            {'num_leaves': 63, 'learning_rate': 0.03, 'max_depth': 9, 'feature_fraction': 0.7, 'min_child_samples': 20},
        ]
    },
    'Caco-2 Permeability Efflux': {
        'use_log': True,
        'configs': [
            {'num_leaves': 63, 'learning_rate': 0.04, 'max_depth': 10, 'feature_fraction': 0.8, 'min_child_samples': 20},
            {'num_leaves': 47, 'learning_rate': 0.05, 'max_depth': 8, 'feature_fraction': 0.75, 'min_child_samples': 25},
            {'num_leaves': 31, 'learning_rate': 0.06, 'max_depth': 7, 'feature_fraction': 0.85, 'min_child_samples': 30},
        ]
    },
    'MPPB': {
        'use_log': False,
        'configs': [
            {'num_leaves': 63, 'learning_rate': 0.04, 'max_depth': 10, 'feature_fraction': 0.8, 'min_child_samples': 20},
            {'num_leaves': 47, 'learning_rate': 0.05, 'max_depth': 8, 'feature_fraction': 0.85, 'min_child_samples': 25},
            {'num_leaves': 31, 'learning_rate': 0.06, 'max_depth': 7, 'feature_fraction': 0.9, 'min_child_samples': 30},
        ]
    },
    'MBPB': {
        'use_log': False,
        'configs': [
            {'num_leaves': 47, 'learning_rate': 0.04, 'max_depth': 8, 'feature_fraction': 0.75, 'min_child_samples': 25},
            {'num_leaves': 31, 'learning_rate': 0.05, 'max_depth': 7, 'feature_fraction': 0.8, 'min_child_samples': 30},
            {'num_leaves': 63, 'learning_rate': 0.03, 'max_depth': 9, 'feature_fraction': 0.7, 'min_child_samples': 20},
        ]
    },
    'MGMB': {
        'use_log': False,
        'configs': [
            {'num_leaves': 15, 'learning_rate': 0.03, 'max_depth': 5, 'feature_fraction': 0.6, 'min_child_samples': 10},
            {'num_leaves': 7, 'learning_rate': 0.02, 'max_depth': 4, 'feature_fraction': 0.5, 'min_child_samples': 5},
            {'num_leaves': 31, 'learning_rate': 0.02, 'max_depth': 6, 'feature_fraction': 0.7, 'min_child_samples': 8},
        ]
    },
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
        'n_jobs': -1,
    }


def train_cv_ensemble(X, y, configs, use_log=False, n_folds=5, n_rounds=500):
    """Train CV ensemble with multiple configurations"""
    valid_mask = ~np.isnan(y)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    if len(y_valid) < 50:
        return None, None, None

    # Transform if using log
    if use_log:
        y_log = np.log1p(np.maximum(y_valid, 0))
        y_mean = np.mean(y_log)
        y_std = np.std(y_log)
        if y_std < 1e-8:
            y_std = 1.0
        y_normalized = (y_log - y_mean) / y_std
        transform_params = {'log_mean': y_mean, 'log_std': y_std, 'use_log': True}
    else:
        y_mean = np.mean(y_valid)
        y_std = np.std(y_valid)
        if y_std < 1e-8:
            y_std = 1.0
        y_normalized = (y_valid - y_mean) / y_std
        transform_params = {'mean': y_mean, 'std': y_std, 'use_log': False}

    model_predictions_oof = []
    model_predictions_test = []
    model_weights = []
    all_fold_models = []

    for config_idx, config in enumerate(configs):
        params = get_base_params()
        params.update(config)
        params['seed'] = 42 + config_idx * 111

        oof_predictions = np.full(len(y_valid), np.nan)
        test_predictions = []
        fold_models = []

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_valid)):
            X_tr, X_val = X_valid[train_idx], X_valid[val_idx]
            y_tr, y_val = y_normalized[train_idx], y_normalized[val_idx]

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

        # Denormalize OOF predictions
        if use_log:
            oof_denorm = np.expm1(oof_predictions * y_std + y_mean)
        else:
            oof_denorm = oof_predictions * y_std + y_mean

        # Average test predictions across folds
        test_pred = np.mean(test_predictions, axis=0)
        if use_log:
            test_denorm = np.expm1(test_pred * y_std + y_mean)
        else:
            test_denorm = test_pred * y_std + y_mean

        # Calculate weight based on Spearman correlation
        spearman, _ = compute_metrics(y_valid, oof_denorm)
        weight = max(0.01, spearman if not np.isnan(spearman) else 0.01)

        model_predictions_oof.append(oof_denorm)
        model_predictions_test.append(test_denorm)
        model_weights.append(weight)
        all_fold_models.append(fold_models)

    # Normalize weights
    model_weights = np.array(model_weights) / sum(model_weights)

    # Weighted ensemble
    oof_array = np.column_stack(model_predictions_oof)
    test_array = np.column_stack(model_predictions_test)

    final_oof = np.average(oof_array, axis=1, weights=model_weights)
    final_test = np.average(test_array, axis=1, weights=model_weights)

    return final_oof, final_test, transform_params


print("\n" + "=" * 80)
print("TRAINING FINAL ENSEMBLE")
print("=" * 80)

all_test_predictions = {}
all_cv_predictions = {}
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
        all_test_predictions[target] = np.full(len(X_test), train_targets[target].median())
        continue

    config = final_configs[target]
    use_log = config['use_log']
    configs = config['configs']

    oof_pred, test_pred, transform_params = train_cv_ensemble(
        X_train, y, configs, use_log=use_log, n_folds=5, n_rounds=500
    )

    if oof_pred is None:
        all_test_predictions[target] = np.full(len(X_test), np.nanmean(y))
        continue

    valid_mask = ~np.isnan(y)
    spearman, ma_rae = compute_metrics(y[valid_mask], oof_pred)

    print(f"  Final CV Spearman: {spearman:.4f}")
    print(f"  Final CV MA-RAE: {ma_rae:.4f}")

    all_cv_predictions[target] = oof_pred
    all_test_predictions[target] = test_pred

    results.append({
        'target': target,
        'n_samples': n_valid,
        'spearman': spearman,
        'ma_rae': ma_rae
    })

# Summary
print("\n" + "=" * 80)
print("FINAL MODEL RESULTS")
print("=" * 80)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

avg_spearman = results_df['spearman'].mean()
avg_ma_rae = results_df['ma_rae'].mean()
print(f"\nSimple Average Spearman: {avg_spearman:.4f}")
print(f"Simple Average MA-RAE: {avg_ma_rae:.4f}")

weights = results_df['n_samples'].values
weighted_spearman = np.average(results_df['spearman'].values, weights=weights)
weighted_ma_rae = np.average(results_df['ma_rae'].values, weights=weights)
print(f"\nWeighted Average Spearman: {weighted_spearman:.4f}")
print(f"Weighted Average MA-RAE: {weighted_ma_rae:.4f}")

# Create final predictions
print("\n" + "=" * 80)
print("CREATING FINAL PREDICTIONS FILE")
print("=" * 80)

# Build predictions dataframe with exact format
pred_df = pd.DataFrame()
pred_df['Molecule Name'] = test_original['Molecule Name']
pred_df['SMILES'] = test_original['SMILES']

for target in target_cols:
    pred_df[target] = all_test_predictions[target]

# Validate format matches training data
print("\nValidating prediction format...")
print(f"Training columns: {train_original.columns.tolist()}")
print(f"Prediction columns: {pred_df.columns.tolist()}")

assert list(pred_df.columns) == list(train_original.columns), "Column order mismatch!"
print("Column order: OK")

# Check for NaN/Inf
for target in target_cols:
    n_nan = pred_df[target].isna().sum()
    n_inf = np.isinf(pred_df[target]).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"  WARNING: {target} has {n_nan} NaN, {n_inf} Inf values")
        # Fill NaN with training median
        pred_df[target] = pred_df[target].fillna(train_targets[target].median())
        pred_df[target] = pred_df[target].replace([np.inf, -np.inf], train_targets[target].median())

print(f"\nPredictions shape: {pred_df.shape}")
print(f"Expected: ({len(test_original)}, {len(train_original.columns)})")

assert pred_df.shape[0] == 2282, f"Expected 2282 rows, got {pred_df.shape[0]}"
assert pred_df.shape[1] == 11, f"Expected 11 columns, got {pred_df.shape[1]}"

# Check value ranges
print("\nValue range check:")
for target in target_cols:
    train_min = train_targets[target].min()
    train_max = train_targets[target].max()
    pred_min = pred_df[target].min()
    pred_max = pred_df[target].max()
    print(f"  {target}: train [{train_min:.2f}, {train_max:.2f}] -> pred [{pred_min:.2f}, {pred_max:.2f}]")

# Save final predictions
pred_df.to_csv('admet_predictions_final.csv', index=False)
print("\n*** SAVED: admet_predictions_final.csv ***")

# Also save as backup
pred_df.to_csv('admet_predictions_v4_final.csv', index=False)
print("Saved: admet_predictions_v4_final.csv")

# Save results
results_df.to_csv('final_results.csv', index=False)
print("Saved: final_results.csv")

print("\n" + "=" * 80)
print("FINAL MODEL COMPLETE")
print("=" * 80)
print(f"\nTarget Performance: Spearman >= 0.81, MA-RAE <= 0.53")
print(f"Achieved: Spearman = {avg_spearman:.4f}, MA-RAE = {avg_ma_rae:.4f}")

if avg_spearman >= 0.81 and avg_ma_rae <= 0.53:
    print("\n*** TARGET ACHIEVED! ***")
elif avg_spearman >= 0.78:
    print("\n*** CLOSE TO TARGET - Good performance ***")
else:
    print("\n*** Below target - consider further optimization ***")
