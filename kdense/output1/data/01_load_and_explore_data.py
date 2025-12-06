#!/usr/bin/env python3
"""
Step 1: Load and Explore Molecular Property Prediction Datasets
Loads training, test, and leaderboard data and performs initial validation.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Set reproducibility
np.random.seed(42)

# Define paths using absolute session directory
SESSION_DIR = Path("/app/sandbox/session_20251205_152206_4285cc85e60d")
USER_DATA_DIR = SESSION_DIR / "user_data"
RESULTS_DIR = SESSION_DIR / "results"

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(exist_ok=True)
print(f"✓ Results directory ready: {RESULTS_DIR}")

# Define file paths
train_file = USER_DATA_DIR / "expansion_data_train.csv"
test_file = USER_DATA_DIR / "expansion_data_test_blinded.csv"
leaderboard_file = USER_DATA_DIR / "current_leaderboard_2025_12_05.csv"

print("\n" + "="*70)
print("STEP 1: DATA LOADING AND VALIDATION")
print("="*70)

# Load training data
print(f"\n[1/3] Loading training data from: {train_file}")
df_train = pd.read_csv(train_file)
print(f"✓ Training data loaded successfully")
print(f"   Shape: {df_train.shape} (rows, columns)")
print(f"   Columns: {list(df_train.columns)}")
print(f"   Memory usage: {df_train.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Load test data
print(f"\n[2/3] Loading test data from: {test_file}")
df_test = pd.read_csv(test_file)
print(f"✓ Test data loaded successfully")
print(f"   Shape: {df_test.shape} (rows, columns)")
print(f"   Columns: {list(df_test.columns)}")

# Load leaderboard data
print(f"\n[3/3] Loading leaderboard data from: {leaderboard_file}")
df_leaderboard = pd.read_csv(leaderboard_file)
print(f"✓ Leaderboard data loaded successfully")
print(f"   Shape: {df_leaderboard.shape} (rows, columns)")
print(f"   Columns: {list(df_leaderboard.columns)}")

# Preview training data
print("\n" + "="*70)
print("TRAINING DATA PREVIEW")
print("="*70)
print("\nFirst 3 rows:")
print(df_train.head(3))

print("\nData types:")
print(df_train.dtypes)

# Identify target property columns
# Typically molecular property columns are numeric and not identifiers
print("\n" + "="*70)
print("IDENTIFYING TARGET PROPERTY COLUMNS")
print("="*70)

# Common identifier columns to exclude
identifier_cols = ['SMILES', 'smiles', 'ID', 'id', 'compound_id', 'mol_id', 'Name', 'name']
numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()

# Filter out likely identifier columns
target_properties = [col for col in numeric_cols if col not in identifier_cols]

print(f"\nNumeric columns found: {len(numeric_cols)}")
print(f"Target property columns identified: {len(target_properties)}")
print(f"Target properties: {target_properties}")

if len(target_properties) == 9:
    print("✓ Found exactly 9 target properties as expected!")
else:
    print(f"⚠ Warning: Expected 9 target properties, but found {len(target_properties)}")
    print("   Will proceed with identified numeric columns")

# Save summary
summary = {
    'train_shape': df_train.shape,
    'test_shape': df_test.shape,
    'leaderboard_shape': df_leaderboard.shape,
    'target_properties': target_properties,
    'train_columns': list(df_train.columns),
    'test_columns': list(df_test.columns)
}

print("\n" + "="*70)
print("DATA LOADING COMPLETE")
print("="*70)
print(f"✓ All datasets loaded successfully")
print(f"✓ Training samples: {df_train.shape[0]:,}")
print(f"✓ Test samples: {df_test.shape[0]:,}")
print(f"✓ Leaderboard entries: {df_leaderboard.shape[0]:,}")
print(f"✓ Target properties: {len(target_properties)}")

# Save datasets as pickle for fast loading in next steps
print("\n[Saving processed data...]")
df_train.to_pickle(RESULTS_DIR / "train_data.pkl")
df_test.to_pickle(RESULTS_DIR / "test_data.pkl")
df_leaderboard.to_pickle(RESULTS_DIR / "leaderboard_data.pkl")

# Save target property list
import json
with open(RESULTS_DIR / "target_properties.json", 'w') as f:
    json.dump(target_properties, f, indent=2)

print(f"✓ Data saved to: {RESULTS_DIR}")
print("\nData loading script completed successfully!")
