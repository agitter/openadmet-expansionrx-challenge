#!/usr/bin/env python3
"""
Generate Baseline Model Predictions for Test Set

This script loads the 9 trained LightGBM models and generates predictions
for all test molecules. Applies inverse log-transform where needed.

Author: K-Dense Coding Agent
Date: 2025-12-05
"""

import os
import pickle
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Session directory
SESSION_DIR = "/app/sandbox/session_20251205_152206_4285cc85e60d"

print("=" * 80)
print("Generate Baseline Predictions for Test Set")
print("=" * 80)

# Load test data
print("\n1. Loading test data...")
test_df = pd.read_csv(f"{SESSION_DIR}/user_data/expansion_data_test_blinded.csv")
print(f"   ✓ Test molecules: {len(test_df):,}")

# Load test features
print("\n2. Loading test features...")
with open(f"{SESSION_DIR}/results/baseline_features_test.pkl", 'rb') as f:
    test_features = pickle.load(f)

X_test = test_features['features']
test_mol_names = test_features['molecule_names']
test_smiles = test_features['smiles']

print(f"   ✓ Features shape: {X_test.shape}")

# Load training data to get property names in correct order
print("\n3. Loading training data for property names...")
with open(f"{SESSION_DIR}/results/train_data.pkl", 'rb') as f:
    train_df = pickle.load(f)

target_properties = [col for col in train_df.columns if col not in ['Molecule Name', 'SMILES']]
print(f"   ✓ Target properties: {target_properties}")

# Load models and generate predictions
print("\n4. Loading models and generating predictions...")
model_dir = f"{SESSION_DIR}/results/baseline_models"

predictions = {}

for i, property_name in enumerate(target_properties, 1):
    print(f"   [{i}/9] Predicting {property_name}...")

    # Load model
    model_name = property_name.replace(' ', '_').replace('>', 'to')
    model_path = f"{model_dir}/model_{model_name}.pkl"

    with open(model_path, 'rb') as f:
        model_info = pickle.load(f)

    model = model_info['model']
    apply_log = model_info['apply_log_transform']

    # Generate predictions
    preds = model.predict(X_test)

    # Apply inverse log-transform if needed
    if apply_log:
        preds = np.power(10, preds) - 1
        print(f"       Transform: Applied inverse log10 transform")
    else:
        print(f"       Transform: None")

    predictions[property_name] = preds
    print(f"       ✓ Predictions range: [{preds.min():.3f}, {preds.max():.3f}]")

# Create predictions dataframe
print("\n5. Creating predictions dataframe...")
pred_df = pd.DataFrame({
    'Molecule Name': test_mol_names,
    'SMILES': test_smiles
})

for prop in target_properties:
    pred_df[prop] = predictions[prop]

print(f"   ✓ Predictions shape: {pred_df.shape}")
print(f"   ✓ Columns: {list(pred_df.columns)}")

# Save predictions
output_path = f"{SESSION_DIR}/results/baseline_test_predictions.csv"
pred_df.to_csv(output_path, index=False)

print(f"\n6. Predictions saved:")
print(f"   ✓ File: {output_path}")
print(f"   ✓ Shape: {pred_df.shape}")

# Summary statistics
print("\n7. Prediction summary statistics:")
for prop in target_properties:
    vals = pred_df[prop].values
    print(f"   {prop:30s}: mean={vals.mean():8.2f}, std={vals.std():8.2f}, "
          f"min={vals.min():8.2f}, max={vals.max():8.2f}")

print("\n" + "=" * 80)
print("✓ Baseline predictions complete!")
print("=" * 80)
