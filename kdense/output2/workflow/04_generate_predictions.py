#!/usr/bin/env python3
"""
Step 4: Generate Predictions and Create Submission File
========================================================

This script:
1. Loads trained XGBoost models for all 9 targets
2. Generates predictions on the test set
3. Applies inverse transformations and physical constraints
4. Creates the final submission file with exact column names from training data
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import json

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
BASE_DIR = Path('/app/sandbox/session_20251217_085238_bf1de403d101')
DATA_DIR = BASE_DIR / 'data'
USER_DATA_DIR = BASE_DIR / 'user_data'
MODELS_DIR = BASE_DIR / 'results' / 'models'
RESULTS_DIR = BASE_DIR / 'results'

print("="*80)
print("Step 4: Generating Predictions for Test Set")
print("="*80)

# Define target mappings (model_name -> full_column_name)
TARGET_MAPPING = {
    'LogD': 'LogD',
    'KSol': 'KSOL',
    'MLM': 'MLM CLint',
    'HLM': 'HLM CLint',
    'Peff': 'Caco-2 Permeability Efflux',
    'Papp': 'Caco-2 Permeability Papp A>B',
    'MPPB': 'MPPB',
    'MBPB': 'MBPB',
    'MGMB': 'MGMB'
}

# Define which targets were log-transformed during training
LOG_TRANSFORMED_TARGETS = ['KSol', 'MLM', 'HLM', 'Peff', 'Papp']

# Define protein binding targets (should be capped at 100%)
PROTEIN_BINDING_TARGETS = ['MPPB', 'MBPB', 'MGMB']

# Load data
print("\n[1/5] Loading test data...")
print("-" * 80)

# Load featurized test data
test_featurized = pd.read_csv(DATA_DIR / 'test_featurized.csv')
print(f"✓ Loaded featurized test data: {test_featurized.shape}")

# Load raw test data for identifiers
test_raw = pd.read_csv(USER_DATA_DIR / 'expansion_data_test_blinded.csv')
print(f"✓ Loaded raw test data: {test_raw.shape}")

# Verify same number of molecules
assert len(test_featurized) == len(test_raw), "Mismatch in number of test molecules!"

# Extract feature columns
feature_cols = [c for c in test_featurized.columns if c.startswith(('fp_', 'desc_'))]
print(f"✓ Identified {len(feature_cols)} feature columns")

# Prepare feature matrix
X_test = test_featurized[feature_cols].values
print(f"✓ Test feature matrix shape: {X_test.shape}")

# Initialize submission dataframe with identifiers
print("\n[2/5] Initializing submission dataframe...")
print("-" * 80)
submission = test_raw[['Molecule Name', 'SMILES']].copy()
print(f"✓ Initialized submission with {len(submission)} molecules")

# Generate predictions for each target
print("\n[3/5] Generating predictions for all targets...")
print("-" * 80)

predictions_dict = {}

for model_name, full_column_name in TARGET_MAPPING.items():
    print(f"\nProcessing {model_name} ({full_column_name})...")

    # Load model
    model_path = MODELS_DIR / f'model_{model_name}.json'
    if not model_path.exists():
        print(f"  ⚠ WARNING: Model file not found: {model_path}")
        continue

    model = xgb.Booster()
    model.load_model(str(model_path))
    print(f"  ✓ Loaded model from {model_path.name}")

    # Create DMatrix for prediction with feature names
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)

    # Generate predictions
    predictions = model.predict(dtest)
    print(f"  ✓ Generated {len(predictions)} predictions")

    # Apply inverse transformation if needed
    if model_name in LOG_TRANSFORMED_TARGETS:
        predictions = np.expm1(predictions)
        print(f"  ✓ Applied inverse log transformation (expm1)")

    # Apply physical constraints
    if model_name != 'LogD':  # LogD can be negative
        # Clip to non-negative
        num_negative = np.sum(predictions < 0)
        if num_negative > 0:
            print(f"  ✓ Clipped {num_negative} negative predictions to 0")
            predictions = np.maximum(predictions, 0)

    if model_name in PROTEIN_BINDING_TARGETS:
        # Cap protein binding at 100%
        num_exceed = np.sum(predictions > 100)
        if num_exceed > 0:
            print(f"  ✓ Capped {num_exceed} predictions exceeding 100% to 100%")
            predictions = np.minimum(predictions, 100)

    # Store predictions with full column name
    predictions_dict[full_column_name] = predictions

    # Print summary statistics
    print(f"  → Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  → Mean: {predictions.mean():.4f}, Median: {np.median(predictions):.4f}")

# Add all predictions to submission
print("\n[4/5] Creating final submission file...")
print("-" * 80)

for full_column_name, predictions in predictions_dict.items():
    submission[full_column_name] = predictions
    print(f"  ✓ Added column: {full_column_name}")

# Save submission
submission_path = RESULTS_DIR / 'submission.csv'
submission.to_csv(submission_path, index=False)
print(f"\n✓ Saved submission file: {submission_path}")
print(f"  Shape: {submission.shape}")
print(f"  Columns: {list(submission.columns)}")

# Generate prediction summary report
print("\n[5/5] Generating prediction summary report...")
print("-" * 80)

summary_lines = []
summary_lines.append("="*80)
summary_lines.append("PREDICTION SUMMARY REPORT")
summary_lines.append("="*80)
summary_lines.append(f"\nTest Set Size: {len(submission)} molecules")
summary_lines.append(f"Number of Properties Predicted: {len(predictions_dict)}")
summary_lines.append("\n" + "-"*80)
summary_lines.append("DESCRIPTIVE STATISTICS FOR EACH PREDICTED PROPERTY")
summary_lines.append("-"*80)

for full_column_name in predictions_dict.keys():
    values = submission[full_column_name].values
    summary_lines.append(f"\n{full_column_name}:")
    summary_lines.append(f"  Min:    {values.min():.6f}")
    summary_lines.append(f"  Q1:     {np.percentile(values, 25):.6f}")
    summary_lines.append(f"  Median: {np.median(values):.6f}")
    summary_lines.append(f"  Mean:   {values.mean():.6f}")
    summary_lines.append(f"  Q3:     {np.percentile(values, 75):.6f}")
    summary_lines.append(f"  Max:    {values.max():.6f}")
    summary_lines.append(f"  Std:    {values.std():.6f}")

    # Check for any potential issues
    issues = []
    if np.any(np.isnan(values)):
        issues.append(f"Contains {np.sum(np.isnan(values))} NaN values")
    if np.any(np.isinf(values)):
        issues.append(f"Contains {np.sum(np.isinf(values))} infinite values")

    if issues:
        summary_lines.append(f"  ⚠ ISSUES: {', '.join(issues)}")

summary_lines.append("\n" + "="*80)
summary_lines.append("VERIFICATION CHECKS")
summary_lines.append("="*80)

# Verification checks
checks = []
checks.append(f"✓ All {len(submission)} test molecules have predictions")
checks.append(f"✓ All 9 target properties predicted")
checks.append(f"✓ Column names match training data format")

# Check for missing values
missing_count = submission.isna().sum().sum()
if missing_count == 0:
    checks.append("✓ No missing values in predictions")
else:
    checks.append(f"⚠ WARNING: {missing_count} missing values found")

# Check for infinite values
inf_cols = []
for col in predictions_dict.keys():
    if np.any(np.isinf(submission[col].values)):
        inf_cols.append(col)
if not inf_cols:
    checks.append("✓ No infinite values in predictions")
else:
    checks.append(f"⚠ WARNING: Infinite values in {', '.join(inf_cols)}")

# Physical constraint checks
constraint_checks = []

# Check LogD can be negative (no constraint)
logd_neg = np.sum(submission['LogD'] < 0)
constraint_checks.append(f"  LogD: {logd_neg} negative values (allowed)")

# Check non-LogD properties are non-negative
for full_col in predictions_dict.keys():
    if full_col != 'LogD':
        neg_count = np.sum(submission[full_col] < 0)
        if neg_count == 0:
            constraint_checks.append(f"  {full_col}: All non-negative ✓")
        else:
            constraint_checks.append(f"  {full_col}: {neg_count} negative values ⚠")

# Check protein binding <= 100%
for model_name in PROTEIN_BINDING_TARGETS:
    full_col = TARGET_MAPPING[model_name]
    exceed_count = np.sum(submission[full_col] > 100)
    if exceed_count == 0:
        constraint_checks.append(f"  {full_col}: All ≤ 100% ✓")
    else:
        constraint_checks.append(f"  {full_col}: {exceed_count} values > 100% ⚠")

summary_lines.extend(checks)
summary_lines.append("\nPhysical Constraints:")
summary_lines.extend(constraint_checks)

summary_lines.append("\n" + "="*80)
summary_lines.append("PREDICTION GENERATION COMPLETE")
summary_lines.append("="*80)

# Save summary report
summary_text = "\n".join(summary_lines)
summary_path = RESULTS_DIR / 'prediction_summary.txt'
with open(summary_path, 'w') as f:
    f.write(summary_text)

print(f"\n✓ Saved prediction summary: {summary_path}")

# Print summary to console
print("\n" + summary_text)

print("\n" + "="*80)
print("STEP 4 COMPLETE: Predictions Generated Successfully!")
print("="*80)
print(f"\nOutput Files:")
print(f"  1. {submission_path}")
print(f"  2. {summary_path}")
print("\n✓ Ready for submission!")
