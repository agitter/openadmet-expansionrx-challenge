#!/usr/bin/env python3
"""
Prepare Final Submission File

This script creates the final test_predictions.csv file from ensemble predictions.
Validates format, checks for invalid values, and applies sanity checks.

Author: K-Dense Coding Agent
Date: 2025-12-05
"""

import pandas as pd
import numpy as np

# Session directory
SESSION_DIR = "/app/sandbox/session_20251205_152206_4285cc85e60d"

print("=" * 80)
print("Prepare Final Submission File")
print("=" * 80)

# Load ensemble predictions (best performing approach)
print("\n1. Loading ensemble predictions...")
ensemble_df = pd.read_csv(f"{SESSION_DIR}/results/ensemble_test_predictions.csv")
print(f"   ✓ Shape: {ensemble_df.shape}")
print(f"   ✓ Columns: {list(ensemble_df.columns)}")

# Validate format
print("\n2. Validating format...")
required_cols = ['Molecule Name', 'SMILES']
property_cols = [col for col in ensemble_df.columns if col not in required_cols]

print(f"   ✓ Required columns present: {all(col in ensemble_df.columns for col in required_cols)}")
print(f"   ✓ Number of properties: {len(property_cols)}")
print(f"   ✓ Number of molecules: {len(ensemble_df)}")

# Check for missing values
print("\n3. Checking for invalid values...")
has_nan = ensemble_df[property_cols].isna().any().any()
has_inf = np.isinf(ensemble_df[property_cols].values).any()

print(f"   NaN values: {has_nan}")
print(f"   Inf values: {has_inf}")

if has_nan or has_inf:
    print("\n   ⚠ Found invalid values! Cleaning...")

    for prop in property_cols:
        vals = ensemble_df[prop].values

        # Replace NaN with median
        if np.isnan(vals).any():
            median_val = np.nanmedian(vals)
            n_nan = np.isnan(vals).sum()
            print(f"      {prop}: Replacing {n_nan} NaN values with median ({median_val:.2f})")
            vals[np.isnan(vals)] = median_val

        # Replace Inf with clipped values
        if np.isinf(vals).any():
            finite_vals = vals[np.isfinite(vals)]
            if len(finite_vals) > 0:
                max_val = np.percentile(finite_vals, 99)
                min_val = np.percentile(finite_vals, 1)
                n_inf = np.isinf(vals).sum()
                print(f"      {prop}: Replacing {n_inf} Inf values with clipped range [{min_val:.2f}, {max_val:.2f}]")
                vals[np.isinf(vals) & (vals > 0)] = max_val
                vals[np.isinf(vals) & (vals < 0)] = min_val

        ensemble_df[prop] = vals

# Apply sanity checks based on training data ranges
print("\n4. Applying sanity checks...")
with open(f"{SESSION_DIR}/results/train_data.pkl", 'rb') as f:
    import pickle
    train_df = pickle.load(f)

for prop in property_cols:
    if prop in train_df.columns:
        train_vals = train_df[prop].dropna().values
        if len(train_vals) > 0:
            # Get reasonable bounds (5th to 95th percentile, with some buffer)
            train_min = np.percentile(train_vals, 1)
            train_max = np.percentile(train_vals, 99)
            buffer = (train_max - train_min) * 0.5  # 50% buffer

            lower_bound = train_min - buffer
            upper_bound = train_max + buffer

            test_vals = ensemble_df[prop].values
            n_clipped_low = (test_vals < lower_bound).sum()
            n_clipped_high = (test_vals > upper_bound).sum()

            if n_clipped_low > 0 or n_clipped_high > 0:
                print(f"   {prop}:")
                print(f"      Training range: [{train_min:.2f}, {train_max:.2f}]")
                print(f"      Allowed range (with buffer): [{lower_bound:.2f}, {upper_bound:.2f}]")
                if n_clipped_low > 0:
                    print(f"      Clipping {n_clipped_low} values below {lower_bound:.2f}")
                if n_clipped_high > 0:
                    print(f"      Clipping {n_clipped_high} values above {upper_bound:.2f}")

                # Clip values
                ensemble_df[prop] = np.clip(test_vals, lower_bound, upper_bound)

# Create final submission file
print("\n5. Creating final submission file...")
submission_df = ensemble_df.copy()

output_path = f"{SESSION_DIR}/results/test_predictions.csv"
submission_df.to_csv(output_path, index=False)

print(f"   ✓ File saved: {output_path}")
print(f"   ✓ Shape: {submission_df.shape}")

# Final verification
print("\n6. Final verification...")
final_has_nan = submission_df[property_cols].isna().any().any()
final_has_inf = np.isinf(submission_df[property_cols].values).any()

print(f"   ✓ NaN values: {final_has_nan}")
print(f"   ✓ Inf values: {final_has_inf}")
print(f"   ✓ All molecules present: {len(submission_df) == 2282}")
print(f"   ✓ All properties present: {len(property_cols) == 9}")

# Summary statistics
print("\n7. Final prediction summary:")
for prop in property_cols:
    vals = submission_df[prop].values
    print(f"   {prop:30s}: mean={vals.mean():8.2f}, std={vals.std():8.2f}, "
          f"min={vals.min():8.2f}, max={vals.max():8.2f}")

print("\n" + "=" * 80)
print("✓ Final submission file ready!")
print("=" * 80)
print(f"\nFinal deliverable: {output_path}")
print(f"Shape: {submission_df.shape}")
print(f"Format: [Molecule Name, SMILES, {', '.join(property_cols)}]")
