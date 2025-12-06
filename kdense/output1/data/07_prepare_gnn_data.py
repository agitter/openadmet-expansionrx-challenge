#!/usr/bin/env python3
"""
GNN Data Preparation with Z-Score Normalization
================================================

CRITICAL: This script implements Z-score normalization for all 9 target properties.
From literature review (Kosmos AI Discovery 2): Z-score normalization is ESSENTIAL
for multi-task GNN success. Without it, gradient imbalance occurs and KSOL dominates
training, leading to near-random predictions (mean Spearman ~0.03).

With proper normalization: mean Spearman 0.8175+ across all endpoints.

Author: K-Dense Coding Agent
Date: 2025-12-05
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("GNN Data Preparation with Z-Score Normalization")
print("=" * 80)

# Define paths
BASE_DIR = Path("/app/sandbox/session_20251205_152206_4285cc85e60d")
TRAIN_DATA_PATH = BASE_DIR / "results" / "train_data.pkl"
OUTPUT_CSV_PATH = BASE_DIR / "workflow" / "gnn_train_data_normalized.csv"
SCALER_PATH = BASE_DIR / "results" / "target_scaler.pkl"

# Load training data
print("\n1. Loading training data...")
print(f"   Input: {TRAIN_DATA_PATH}")

with open(TRAIN_DATA_PATH, 'rb') as f:
    train_data = pickle.load(f)

print(f"   ✓ Loaded: {train_data.shape[0]:,} molecules")
print(f"   ✓ Columns: {list(train_data.columns)}")

# Define target properties (9 ADMET endpoints)
target_properties = [
    'LogD',
    'KSOL',
    'HLM CLint',
    'MLM CLint',
    'Caco-2 Permeability Papp A>B',
    'Caco-2 Permeability Efflux',
    'MPPB',
    'MBPB',
    'MGMB'
]

# Verify all target properties are present
missing_targets = [t for t in target_properties if t not in train_data.columns]
if missing_targets:
    raise ValueError(f"Missing target properties: {missing_targets}")

print(f"\n   ✓ All 9 target properties found")

# Check data availability
print("\n2. Data availability summary:")
for prop in target_properties:
    n_available = train_data[prop].notna().sum()
    pct_available = 100 * n_available / len(train_data)
    n_missing = train_data[prop].isna().sum()
    pct_missing = 100 * n_missing / len(train_data)
    print(f"   {prop:35s}: {n_available:5d} available ({pct_available:5.1f}%), "
          f"{n_missing:5d} missing ({pct_missing:5.1f}%)")

# Extract SMILES and targets
print("\n3. Extracting SMILES and target values...")
smiles = train_data['SMILES'].values
targets_df = train_data[target_properties].copy()

print(f"   ✓ SMILES: {len(smiles):,} molecules")
print(f"   ✓ Targets shape: {targets_df.shape}")

# CRITICAL: Z-Score Normalization
# ================================
# Calculate mean and std for each property using ONLY available (non-missing) data
# This is essential to prevent KSOL from dominating the gradient during training

print("\n4. Computing Z-score normalization parameters (CRITICAL STEP)...")
print("   NOTE: Calculating mean/std only on available (non-missing) values")

scaler_params = {}
normalized_targets_df = pd.DataFrame(index=targets_df.index)

for prop in target_properties:
    # Get non-missing values
    non_missing_values = targets_df[prop].dropna()

    if len(non_missing_values) == 0:
        print(f"   WARNING: {prop} has NO available values - skipping normalization")
        scaler_params[prop] = {'mean': 0.0, 'std': 1.0, 'n_samples': 0}
        normalized_targets_df[prop] = targets_df[prop]  # Keep as NaN
        continue

    # Calculate mean and std on non-missing values only
    mean = non_missing_values.mean()
    std = non_missing_values.std()
    n_samples = len(non_missing_values)

    # Store scaler parameters
    scaler_params[prop] = {
        'mean': float(mean),
        'std': float(std),
        'n_samples': int(n_samples)
    }

    # Apply z-score normalization: z = (x - mean) / std
    # NaN values remain NaN (not affected by normalization)
    normalized_targets_df[prop] = (targets_df[prop] - mean) / std

    # Statistics for reporting
    normalized_values = normalized_targets_df[prop].dropna()

    print(f"   {prop:35s}:")
    print(f"      Original - Mean: {mean:10.3f}, Std: {std:10.3f}, N: {n_samples:5d}")
    print(f"      Normalized - Mean: {normalized_values.mean():7.3f}, Std: {normalized_values.std():7.3f}")

# Verify normalization was successful
print("\n5. Validating normalized data...")
validation_passed = True

for prop in target_properties:
    normalized_values = normalized_targets_df[prop].dropna()

    if len(normalized_values) == 0:
        continue

    # Check for infinite or extreme values
    if np.isinf(normalized_values).any():
        print(f"   ✗ ERROR: {prop} contains infinite values after normalization")
        validation_passed = False

    if (np.abs(normalized_values) > 100).any():
        print(f"   ⚠ WARNING: {prop} contains extreme normalized values (|z| > 100)")
        print(f"      Max |z|: {np.abs(normalized_values).max():.2f}")

    # Verify normalization worked (mean ≈ 0, std ≈ 1)
    norm_mean = normalized_values.mean()
    norm_std = normalized_values.std()

    if abs(norm_mean) > 0.01:
        print(f"   ⚠ WARNING: {prop} normalized mean not close to 0: {norm_mean:.6f}")

    if abs(norm_std - 1.0) > 0.01:
        print(f"   ⚠ WARNING: {prop} normalized std not close to 1: {norm_std:.6f}")

if validation_passed:
    print("   ✓ All validations passed")
else:
    raise ValueError("Normalization validation failed - see errors above")

# Save scaler parameters
print(f"\n6. Saving scaler parameters...")
print(f"   Output: {SCALER_PATH}")

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler_params, f)

print(f"   ✓ Scaler saved successfully")
print(f"   ✓ Contains normalization params for {len(scaler_params)} properties")

# Create output CSV for Chemprop
print(f"\n7. Creating CSV file for Chemprop...")
print(f"   Output: {OUTPUT_CSV_PATH}")

# Combine SMILES and normalized targets
output_df = pd.DataFrame({'smiles': smiles})

# Add normalized target columns
for prop in target_properties:
    # Chemprop expects lowercase column names for targets
    # Use simplified names to avoid special characters in column names
    simple_name = prop.replace(' ', '_').replace('>', 'to')
    output_df[simple_name] = normalized_targets_df[prop]

print(f"   ✓ Output shape: {output_df.shape}")
print(f"   ✓ Columns: {list(output_df.columns)}")

# Check for missing values
n_missing_per_col = output_df.isna().sum()
print(f"\n   Missing values per column:")
for col in output_df.columns:
    if col != 'smiles':
        n_missing = n_missing_per_col[col]
        pct_missing = 100 * n_missing / len(output_df)
        print(f"      {col:40s}: {n_missing:5d} ({pct_missing:5.1f}%)")

# Save CSV
output_df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\n   ✓ CSV file saved successfully")
print(f"   ✓ File size: {OUTPUT_CSV_PATH.stat().st_size / 1024:.1f} KB")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Loaded {len(smiles):,} training molecules")
print(f"✓ Computed Z-score normalization for 9 target properties")
print(f"✓ Saved scaler parameters to: {SCALER_PATH.name}")
print(f"✓ Created Chemprop-ready CSV: {OUTPUT_CSV_PATH.name}")
print(f"✓ Output dimensions: {output_df.shape[0]:,} rows × {output_df.shape[1]} columns")
print("\nCRITICAL CONFIRMATION:")
print("  ✓ Z-score normalization applied (mean ≈ 0, std ≈ 1 for each property)")
print("  ✓ Missing values preserved as NaN (Chemprop handles internally)")
print("  ✓ Scaler saved for inverse transform during test prediction")
print("\n✓ Data preparation complete - ready for GNN training")
print("=" * 80)
