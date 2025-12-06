#!/usr/bin/env python3
"""
Create Ensemble Predictions

This script combines baseline and GNN predictions using simple arithmetic mean.
Literature suggests ensembles provide 5-10% improvement over single models.

Author: K-Dense Coding Agent
Date: 2025-12-05
"""

import pandas as pd
import numpy as np

# Session directory
SESSION_DIR = "/app/sandbox/session_20251205_152206_4285cc85e60d"

print("=" * 80)
print("Create Ensemble Predictions")
print("=" * 80)

# Load predictions
print("\n1. Loading model predictions...")
baseline_df = pd.read_csv(f"{SESSION_DIR}/results/baseline_test_predictions.csv")
gnn_df = pd.read_csv(f"{SESSION_DIR}/results/gnn_test_predictions.csv")

print(f"   ✓ Baseline predictions: {baseline_df.shape}")
print(f"   ✓ GNN predictions: {gnn_df.shape}")

# Verify same molecules in same order
assert (baseline_df['Molecule Name'] == gnn_df['Molecule Name']).all(), "Molecule names don't match!"
assert (baseline_df['SMILES'] == gnn_df['SMILES']).all(), "SMILES don't match!"

print("   ✓ Verified: Same molecules in same order")

# Get property columns
id_cols = ['Molecule Name', 'SMILES']
property_cols = [col for col in baseline_df.columns if col not in id_cols]

print(f"   ✓ Properties to ensemble: {len(property_cols)}")

# Create ensemble predictions (arithmetic mean)
print("\n2. Creating ensemble predictions (arithmetic mean)...")
ensemble_df = baseline_df[id_cols].copy()

for prop in property_cols:
    baseline_vals = baseline_df[prop].values
    gnn_vals = gnn_df[prop].values

    # Simple arithmetic mean
    ensemble_vals = (baseline_vals + gnn_vals) / 2

    ensemble_df[prop] = ensemble_vals

    # Print comparison
    print(f"   {prop:30s}:")
    print(f"      Baseline: mean={baseline_vals.mean():8.2f}, std={baseline_vals.std():8.2f}")
    print(f"      GNN:      mean={gnn_vals.mean():8.2f}, std={gnn_vals.std():8.2f}")
    print(f"      Ensemble: mean={ensemble_vals.mean():8.2f}, std={ensemble_vals.std():8.2f}")

print(f"\n   ✓ Ensemble predictions shape: {ensemble_df.shape}")

# Save ensemble predictions
output_path = f"{SESSION_DIR}/results/ensemble_test_predictions.csv"
ensemble_df.to_csv(output_path, index=False)

print(f"\n3. Ensemble predictions saved:")
print(f"   ✓ File: {output_path}")
print(f"   ✓ Shape: {ensemble_df.shape}")

# Summary statistics
print("\n4. Ensemble prediction summary:")
for prop in property_cols:
    vals = ensemble_df[prop].values
    print(f"   {prop:30s}: mean={vals.mean():8.2f}, std={vals.std():8.2f}, "
          f"min={vals.min():8.2f}, max={vals.max():8.2f}")

print("\n" + "=" * 80)
print("✓ Ensemble predictions complete!")
print("=" * 80)
