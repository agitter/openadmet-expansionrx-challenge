#!/usr/bin/env python3
"""
Retrain Baseline LightGBM Models on Full Training Set

This script trains 9 separate LightGBM models (one per property) on the
COMPLETE training dataset. These models will be used to generate predictions
for the blind test set.

Uses same hyperparameters and preprocessing as cross-validation:
- 500 trees, learning_rate=0.05, max_depth=8
- Log-transform for skewed properties: HLM CLint, MLM CLint, Caco-2 Efflux, MBPB
- Same molecular features: Morgan (2048) + RDKit (217)

Author: K-Dense Coding Agent
Date: 2025-12-05
"""

import os
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

# Session directory
SESSION_DIR = "/app/sandbox/session_20251205_152206_4285cc85e60d"

# Properties requiring log-transform (same as CV)
LOG_TRANSFORM_PROPERTIES = ['HLM CLint', 'MLM CLint', 'Caco-2 Permeability Efflux', 'MBPB']

print("=" * 80)
print("Retrain Baseline LightGBM Models on Full Training Set")
print("=" * 80)

# Load training data
print("\n1. Loading training data and features...")
with open(f"{SESSION_DIR}/results/train_data.pkl", 'rb') as f:
    train_df = pickle.load(f)

with open(f"{SESSION_DIR}/results/baseline_features_train.pkl", 'rb') as f:
    feature_data = pickle.load(f)

# Extract target properties (all columns except Molecule Name and SMILES)
target_properties = [col for col in train_df.columns if col not in ['Molecule Name', 'SMILES']]
X = feature_data['features']

print(f"   ✓ Training molecules: {len(train_df):,}")
print(f"   ✓ Features shape: {X.shape}")
print(f"   ✓ Target properties: {len(target_properties)}")

# Create output directory
model_dir = f"{SESSION_DIR}/results/baseline_models"
os.makedirs(model_dir, exist_ok=True)
print(f"   ✓ Model directory: {model_dir}")

# Train models for each property
print("\n2. Training models on full training set...")
print("   Using same hyperparameters as CV:")
print("   - n_estimators: 500")
print("   - learning_rate: 0.05")
print("   - max_depth: 8")
print("   - num_leaves: 31")
print("   - random_state: 42")

trained_models = {}
training_summary = []

for prop_idx, property_name in enumerate(target_properties, 1):
    print(f"\n   [{prop_idx}/9] Training model for: {property_name}")

    # Get target values (remove missing data)
    y = train_df[property_name].values
    valid_mask = ~np.isnan(y)

    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    n_samples = len(y_valid)
    missing_pct = 100 * (1 - n_samples / len(y))

    print(f"       Samples: {n_samples:,} ({missing_pct:.1f}% missing)")

    # Apply log-transform if needed
    apply_log = property_name in LOG_TRANSFORM_PROPERTIES
    if apply_log:
        y_train = np.log10(y_valid + 1)
        print(f"       Transform: log10(x + 1) applied")
    else:
        y_train = y_valid
        print(f"       Transform: None")

    # Train model
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        random_state=42,
        verbose=-1,
        force_col_wise=True
    )

    model.fit(X_valid, y_train)

    # Store model
    model_name = property_name.replace(' ', '_').replace('>', 'to')
    trained_models[property_name] = {
        'model': model,
        'apply_log_transform': apply_log,
        'n_train_samples': n_samples,
        'model_name': model_name
    }

    # Save individual model
    model_path = f"{model_dir}/model_{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(trained_models[property_name], f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"       ✓ Model saved: model_{model_name}.pkl")

    training_summary.append({
        'Property': property_name,
        'N_Samples': n_samples,
        'Missing_Pct': missing_pct,
        'Log_Transform': apply_log,
        'Model_File': f"model_{model_name}.pkl"
    })

# Save training summary
summary_df = pd.DataFrame(training_summary)
summary_path = f"{model_dir}/training_summary.csv"
summary_df.to_csv(summary_path, index=False)

print(f"\n3. Training complete!")
print(f"   ✓ Models trained: {len(trained_models)}")
print(f"   ✓ Models saved to: {model_dir}")
print(f"   ✓ Training summary: {summary_path}")

print("\n" + "=" * 80)
print("✓ Baseline model retraining complete!")
print("=" * 80)
print("\nTraining Summary:")
print(summary_df.to_string(index=False))
