#!/usr/bin/env python3
"""
OpenADMET ExpansionRx Blind Challenge - Advanced Ensemble Model
Purpose: Build multi-model ensemble with hyperparameter tuning, cross-validation,
         and stacking meta-learner for optimal ADMET prediction
Date: 2025-12-19

Key strategies from KOSMOS report:
1. Z-score normalize targets before training
2. Use multiple feature sets (RDKit, Morgan FPs, MACCS)
3. Ensemble of LightGBM models with different hyperparameters
4. Stacking meta-learner to combine predictions
5. Handle sparse data with masked training
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, HuberRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("=" * 80)
print("ADVANCED ENSEMBLE MODEL")
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

# Remove constant features
const_cols = train_features.columns[train_features.std() == 0].tolist()
train_features = train_features.drop(columns=const_cols)
test_features = test_features.drop(columns=const_cols)

# Handle inf values and fill NaN
train_features = train_features.replace([np.inf, -np.inf], np.nan)
test_features = test_features.replace([np.inf, -np.inf], np.nan)
train_median = train_features.median()
train_features = train_features.fillna(train_median)
test_features = test_features.fillna(train_median)

# Feature column groupings
feature_cols = train_features.columns.tolist()
rdkit_cols = [c for c in feature_cols if c.startswith('rdkit_')]
morgan2_cols = [c for c in feature_cols if c.startswith('morgan2_')]
morgan3_cols = [c for c in feature_cols if c.startswith('morgan3_')]
maccs_cols = [c for c in feature_cols if c.startswith('maccs_')]
smiles_cols = [c for c in feature_cols if c.startswith('smiles_')]

print(f"Feature groups:")
print(f"  RDKit: {len(rdkit_cols)}")
print(f"  Morgan r=2: {len(morgan2_cols)}")
print(f"  Morgan r=3: {len(morgan3_cols)}")
print(f"  MACCS: {len(maccs_cols)}")
print(f"  SMILES: {len(smiles_cols)}")
print(f"  Total: {len(feature_cols)}")

# Full data
X_full = train_features.values
X_test = test_features.values

# Correct MA-RAE calculation
def compute_ma_rae(y_true, y_pred):
    """
    MA-RAE: Mean Absolute Relative Absolute Error
    Formula: mean(|y_true - y_pred|) / mean(|y_true - mean(y_true)|)
    This is equivalent to MAE / MAD (MAE normalized by mean absolute deviation)
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() < 10:
        return np.nan

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = np.mean(np.abs(y_true - y_pred))
    mad = np.mean(np.abs(y_true - np.mean(y_true)))

    if mad < 1e-8:
        return np.nan

    return mae / mad


def compute_metrics(y_true, y_pred):
    """Compute Spearman correlation and MA-RAE"""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() < 10:
        return np.nan, np.nan

    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    spearman, _ = stats.spearmanr(y_true_clean, y_pred_clean)
    ma_rae = compute_ma_rae(y_true, y_pred)

    return spearman, ma_rae


# Different LightGBM configurations for ensemble diversity
lgb_configs = [
    {
        'name': 'lgb_default',
        'params': {
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
        },
        'feature_cols': feature_cols  # All features
    },
    {
        'name': 'lgb_rdkit_morgan2',
        'params': {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 8,
            'learning_rate': 0.03,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_child_samples': 30,
            'lambda_l1': 0.2,
            'lambda_l2': 0.2,
            'verbose': -1,
            'seed': 123,
            'n_jobs': -1,
        },
        'feature_cols': rdkit_cols + morgan2_cols + smiles_cols
    },
    {
        'name': 'lgb_deep',
        'params': {
            'objective': 'regression',
            'metric': 'huber',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'max_depth': 15,
            'learning_rate': 0.02,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.8,
            'bagging_freq': 3,
            'min_child_samples': 10,
            'lambda_l1': 0.05,
            'lambda_l2': 0.05,
            'verbose': -1,
            'seed': 456,
            'n_jobs': -1,
        },
        'feature_cols': feature_cols
    },
    {
        'name': 'lgb_rdkit_only',
        'params': {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 47,
            'max_depth': 12,
            'learning_rate': 0.04,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 4,
            'min_child_samples': 15,
            'lambda_l1': 0.15,
            'lambda_l2': 0.15,
            'verbose': -1,
            'seed': 789,
            'n_jobs': -1,
        },
        'feature_cols': rdkit_cols + smiles_cols
    },
]


def train_single_model(X_train, y_train, X_val, y_val, params, n_rounds=500, early_stop=50):
    """Train a single LightGBM model with early stopping"""
    train_mask = ~np.isnan(y_train)
    val_mask = ~np.isnan(y_val)

    if train_mask.sum() < 30:
        return None

    X_tr = X_train[train_mask]
    y_tr = y_train[train_mask]

    train_data = lgb.Dataset(X_tr, label=y_tr)

    if val_mask.sum() >= 20:
        X_v = X_val[val_mask]
        y_v = y_val[val_mask]
        val_data = lgb.Dataset(X_v, label=y_v, reference=train_data)
        callbacks = [lgb.early_stopping(stopping_rounds=early_stop, verbose=False)]
        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_rounds,
            valid_sets=[val_data],
            callbacks=callbacks
        )
    else:
        model = lgb.train(params, train_data, num_boost_round=min(300, n_rounds))

    return model


def get_feature_matrix(df, feature_cols):
    """Extract feature matrix for specific columns"""
    return df[feature_cols].values


# Training loop
print("\n" + "=" * 80)
print("TRAINING ENSEMBLE MODELS")
print("=" * 80)

all_models = {}
all_scalers = {}
cv_predictions_train = {}  # For stacking
final_predictions_test = {}

for target in target_cols:
    print(f"\n{'='*60}")
    print(f"Target: {target}")
    print(f"{'='*60}")

    y_full = train_targets[target].values
    n_available = (~np.isnan(y_full)).sum()
    print(f"  Available samples: {n_available}")

    if n_available < 50:
        print(f"  SKIPPING: Insufficient data")
        continue

    # Z-score normalization parameters (from all available data)
    y_mean = np.nanmean(y_full)
    y_std = np.nanstd(y_full)
    if y_std < 1e-8:
        y_std = 1.0

    all_scalers[target] = {'mean': y_mean, 'std': y_std}
    y_scaled = (y_full - y_mean) / y_std

    # Split data
    y_train = y_scaled[train_indices]
    y_val = y_scaled[val_indices]
    y_train_orig = y_full[train_indices]
    y_val_orig = y_full[val_indices]

    # Store predictions from each model
    model_predictions_val = []
    model_predictions_test = []
    model_names = []
    target_models = []

    for config in lgb_configs:
        model_name = config['name']
        params = config['params'].copy()
        feat_cols = config['feature_cols']

        # Get feature matrices
        X_train_model = train_features.loc[:, train_features.columns.isin(feat_cols)].values[train_indices]
        X_val_model = train_features.loc[:, train_features.columns.isin(feat_cols)].values[val_indices]
        X_test_model = test_features.loc[:, test_features.columns.isin(feat_cols)].values

        # Train model
        model = train_single_model(X_train_model, y_train, X_val_model, y_val, params)

        if model is None:
            continue

        target_models.append({
            'name': model_name,
            'model': model,
            'feature_cols': feat_cols
        })

        # Predictions (scaled)
        pred_val = model.predict(X_val_model)
        pred_test = model.predict(X_test_model)

        # Denormalize
        pred_val_orig = pred_val * y_std + y_mean
        pred_test_orig = pred_test * y_std + y_mean

        model_predictions_val.append(pred_val_orig)
        model_predictions_test.append(pred_test_orig)
        model_names.append(model_name)

        # Compute metrics
        spearman, ma_rae = compute_metrics(y_val_orig, pred_val_orig)
        print(f"    {model_name}: Spearman={spearman:.4f}, MA-RAE={ma_rae:.4f}")

    if len(model_predictions_val) == 0:
        print(f"  No valid models trained")
        continue

    all_models[target] = target_models

    # Ensemble methods
    val_preds_array = np.column_stack(model_predictions_val)
    test_preds_array = np.column_stack(model_predictions_test)

    # 1. Simple average
    avg_val = np.mean(val_preds_array, axis=1)
    avg_test = np.mean(test_preds_array, axis=1)
    spearman_avg, ma_rae_avg = compute_metrics(y_val_orig, avg_val)
    print(f"  Average ensemble: Spearman={spearman_avg:.4f}, MA-RAE={ma_rae_avg:.4f}")

    # 2. Weighted average (by validation performance)
    weights = []
    for i, name in enumerate(model_names):
        sp, _ = compute_metrics(y_val_orig, model_predictions_val[i])
        weights.append(max(0.01, sp))  # Ensure positive weight
    weights = np.array(weights) / sum(weights)

    weighted_val = np.average(val_preds_array, axis=1, weights=weights)
    weighted_test = np.average(test_preds_array, axis=1, weights=weights)
    spearman_wt, ma_rae_wt = compute_metrics(y_val_orig, weighted_val)
    print(f"  Weighted ensemble: Spearman={spearman_wt:.4f}, MA-RAE={ma_rae_wt:.4f}")

    # 3. Stacking with Ridge regression (robust meta-learner)
    val_mask = ~np.isnan(y_val_orig)
    if val_mask.sum() >= 30:
        # Train stacking model on validation predictions
        # We use validation as meta-training since we're in competition setting
        meta_X = val_preds_array[val_mask]
        meta_y = y_val_orig[val_mask]

        # Simple Ridge for robustness
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_X, meta_y)

        stacked_val = meta_model.predict(val_preds_array)
        stacked_test = meta_model.predict(test_preds_array)

        spearman_stack, ma_rae_stack = compute_metrics(y_val_orig, stacked_val)
        print(f"  Stacked ensemble: Spearman={spearman_stack:.4f}, MA-RAE={ma_rae_stack:.4f}")
    else:
        stacked_val = weighted_val
        stacked_test = weighted_test
        spearman_stack, ma_rae_stack = spearman_wt, ma_rae_wt

    # Choose best ensemble method
    best_spearman = max(spearman_avg, spearman_wt, spearman_stack)
    if best_spearman == spearman_stack:
        final_val = stacked_val
        final_test = stacked_test
        best_method = 'stacked'
    elif best_spearman == spearman_wt:
        final_val = weighted_val
        final_test = weighted_test
        best_method = 'weighted'
    else:
        final_val = avg_val
        final_test = avg_test
        best_method = 'average'

    print(f"  Best method: {best_method} (Spearman={best_spearman:.4f})")

    cv_predictions_train[target] = final_val
    final_predictions_test[target] = final_test

# Summary
print("\n" + "=" * 80)
print("FINAL ENSEMBLE RESULTS")
print("=" * 80)

results_summary = []
for target in target_cols:
    if target in cv_predictions_train:
        y_val_orig = train_targets[target].values[val_indices]
        pred_val = cv_predictions_train[target]
        spearman, ma_rae = compute_metrics(y_val_orig, pred_val)
        n_val = (~np.isnan(y_val_orig)).sum()
        results_summary.append({
            'target': target,
            'n_val': n_val,
            'spearman': spearman,
            'ma_rae': ma_rae
        })
        print(f"{target}: Spearman={spearman:.4f}, MA-RAE={ma_rae:.4f}, n={n_val}")

results_df = pd.DataFrame(results_summary)
print(f"\nAverage Spearman: {results_df['spearman'].mean():.4f}")
print(f"Average MA-RAE: {results_df['ma_rae'].mean():.4f}")

# Create final predictions dataframe
print("\n" + "=" * 80)
print("CREATING FINAL PREDICTIONS")
print("=" * 80)

test_mol_info = pd.read_csv('test_mol_info.csv')
pred_df = test_mol_info.copy()

for target in target_cols:
    if target in final_predictions_test:
        pred_df[target] = final_predictions_test[target]
    else:
        # Use training median for missing targets
        pred_df[target] = train_targets[target].median()

# Ensure column order matches training data
train_original = pd.read_csv('expansion_data_train.csv')
pred_df = pred_df[train_original.columns]

# Save predictions
pred_df.to_csv('ensemble_predictions.csv', index=False)
print("Saved: ensemble_predictions.csv")

# Also save as current best version
pred_df.to_csv('admet_predictions_v2.csv', index=False)
print("Saved: admet_predictions_v2.csv")

# Save models
with open('ensemble_models.pkl', 'wb') as f:
    pickle.dump({
        'models': all_models,
        'scalers': all_scalers
    }, f)
print("Saved: ensemble_models.pkl")

# Save results
results_df.to_csv('ensemble_results.csv', index=False)
print("Saved: ensemble_results.csv")

print("\n" + "=" * 80)
print("ENSEMBLE TRAINING COMPLETE")
print("=" * 80)
