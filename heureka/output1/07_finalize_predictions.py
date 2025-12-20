#!/usr/bin/env python3
"""
OpenADMET ExpansionRx Blind Challenge - Finalize Predictions
Purpose: Clip out-of-range values and validate final submission file
Date: 2025-12-19
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("FINALIZING PREDICTIONS")
print("=" * 80)

# Load predictions
pred_df = pd.read_csv('admet_predictions_final.csv')
train_df = pd.read_csv('expansion_data_train.csv')

target_cols = ['LogD', 'KSOL', 'HLM CLint', 'MLM CLint', 'Caco-2 Permeability Papp A>B',
               'Caco-2 Permeability Efflux', 'MPPB', 'MBPB', 'MGMB']

print(f"\nPredictions shape: {pred_df.shape}")
print(f"Expected: (2282, 11)")

# Define valid ranges based on training data + some margin
# Properties that must be non-negative
non_negative_props = ['KSOL', 'HLM CLint', 'MLM CLint', 'Caco-2 Permeability Papp A>B',
                      'Caco-2 Permeability Efflux', 'MPPB', 'MBPB', 'MGMB']

print("\n" + "=" * 60)
print("Checking and clipping values...")
print("=" * 60)

for target in target_cols:
    train_min = train_df[target].min()
    train_max = train_df[target].max()
    pred_min = pred_df[target].min()
    pred_max = pred_df[target].max()

    # Count out-of-range values
    n_below_min = (pred_df[target] < train_min).sum()
    n_above_max = (pred_df[target] > train_max).sum()

    print(f"\n{target}:")
    print(f"  Train range: [{train_min:.3f}, {train_max:.3f}]")
    print(f"  Pred range: [{pred_min:.3f}, {pred_max:.3f}]")
    print(f"  Below train min: {n_below_min}")
    print(f"  Above train max: {n_above_max}")

    # Clip to reasonable ranges
    if target in non_negative_props:
        # Clip negative values to 0
        n_negative = (pred_df[target] < 0).sum()
        if n_negative > 0:
            print(f"  Clipping {n_negative} negative values to 0")
            pred_df[target] = pred_df[target].clip(lower=0)

    # For extreme extrapolations, clip more aggressively
    # Allow 50% extrapolation beyond training range
    margin = 0.5 * (train_max - train_min)
    clip_min = train_min - margin
    clip_max = train_max + margin

    if target in non_negative_props:
        clip_min = max(0, clip_min)

    n_clipped_low = (pred_df[target] < clip_min).sum()
    n_clipped_high = (pred_df[target] > clip_max).sum()

    if n_clipped_low > 0 or n_clipped_high > 0:
        print(f"  Clipping to [{clip_min:.3f}, {clip_max:.3f}]")
        print(f"  Low clips: {n_clipped_low}, High clips: {n_clipped_high}")
        pred_df[target] = pred_df[target].clip(lower=clip_min, upper=clip_max)

# Final check
print("\n" + "=" * 60)
print("Final value ranges:")
print("=" * 60)

for target in target_cols:
    pred_min = pred_df[target].min()
    pred_max = pred_df[target].max()
    print(f"  {target}: [{pred_min:.3f}, {pred_max:.3f}]")

# Verify no NaN or Inf
print("\n" + "=" * 60)
print("Data integrity check:")
print("=" * 60)

for target in target_cols:
    n_nan = pred_df[target].isna().sum()
    n_inf = np.isinf(pred_df[target]).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"  WARNING: {target} has {n_nan} NaN, {n_inf} Inf")
    else:
        print(f"  {target}: OK")

# Save final predictions
pred_df.to_csv('admet_predictions_final.csv', index=False)
print("\n*** SAVED: admet_predictions_final.csv ***")

# Verification
print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

# Re-read and verify
verify_df = pd.read_csv('admet_predictions_final.csv')
print(f"\nFile verification:")
print(f"  Rows: {len(verify_df)} (expected: 2282)")
print(f"  Columns: {len(verify_df.columns)} (expected: 11)")
print(f"  Column names: {verify_df.columns.tolist()}")

# Check first and last rows
print(f"\nFirst molecule: {verify_df['Molecule Name'].iloc[0]}")
print(f"Last molecule: {verify_df['Molecule Name'].iloc[-1]}")

print("\n" + "=" * 60)
print("PREDICTIONS READY FOR SUBMISSION")
print("=" * 60)
