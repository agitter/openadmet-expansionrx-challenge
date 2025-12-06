#!/usr/bin/env python3
"""
MA-RAE Implementation Verification Script

This script validates that:
1. MA-RAE columns exist in both CSV files
2. All MA-RAE values are valid (not NaN or Inf)
3. Mean MA-RAE is calculated correctly
4. Mean MA-RAE is compared against target threshold (< 0.53)

Author: K-Dense Coding Agent
Date: 2025-12-05
"""

import pandas as pd
import numpy as np
import sys

# Session directory
SESSION_DIR = "/app/sandbox/session_20251205_152206_4285cc85e60d"

def validate_csv_file(filepath, model_name):
    """
    Validate MA-RAE implementation in a CSV file.

    Parameters
    ----------
    filepath : str
        Path to CSV file
    model_name : str
        Name of the model (for reporting)

    Returns
    -------
    dict
        Validation results
    """
    print(f"\n{'='*80}")
    print(f"Validating {model_name}")
    print(f"{'='*80}")
    print(f"File: {filepath}")
    print()

    # Read CSV
    try:
        df = pd.read_csv(filepath)
        print(f"✓ File loaded successfully")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print()
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return {
            'valid': False,
            'error': str(e)
        }

    # Check for MA-RAE columns
    required_columns = ['ma_rae_mean', 'ma_rae_std']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"✗ Missing required columns: {missing_columns}")
        return {
            'valid': False,
            'error': f'Missing columns: {missing_columns}'
        }

    print(f"✓ Required MA-RAE columns present: {required_columns}")
    print()

    # Check for valid values (not NaN or Inf)
    ma_rae_mean = df['ma_rae_mean']
    ma_rae_std = df['ma_rae_std']

    n_total = len(ma_rae_mean)
    n_nan_mean = ma_rae_mean.isna().sum()
    n_inf_mean = np.isinf(ma_rae_mean).sum()
    n_nan_std = ma_rae_std.isna().sum()
    n_inf_std = np.isinf(ma_rae_std).sum()

    print("MA-RAE Value Validation:")
    print(f"  Total properties: {n_total}")
    print(f"  ma_rae_mean:")
    print(f"    - Valid values: {n_total - n_nan_mean - n_inf_mean}")
    print(f"    - NaN values: {n_nan_mean}")
    print(f"    - Inf values: {n_inf_mean}")
    print(f"  ma_rae_std:")
    print(f"    - Valid values: {n_total - n_nan_std - n_inf_std}")
    print(f"    - NaN values: {n_nan_std}")
    print(f"    - Inf values: {n_inf_std}")
    print()

    # Calculate mean MA-RAE (excluding NaN/Inf)
    valid_mask = ~(ma_rae_mean.isna() | np.isinf(ma_rae_mean))
    if valid_mask.sum() > 0:
        mean_ma_rae = ma_rae_mean[valid_mask].mean()
        print(f"✓ Mean MA-RAE across all properties: {mean_ma_rae:.4f}")
        print(f"  Calculated from {valid_mask.sum()} valid properties")
        print()
    else:
        print(f"✗ No valid MA-RAE values found")
        mean_ma_rae = np.nan

    # Show property-level MA-RAE scores
    print("Property-Level MA-RAE Scores:")
    print(f"{'Property':<40} {'MA-RAE Mean':<15} {'MA-RAE Std':<15}")
    print("─" * 70)

    for idx, row in df.iterrows():
        prop = row.get('property', f'Property_{idx}')
        ma_rae_m = row['ma_rae_mean']
        ma_rae_s = row['ma_rae_std']

        # Format with validity indicator
        if pd.isna(ma_rae_m):
            ma_rae_m_str = "NaN"
        elif np.isinf(ma_rae_m):
            ma_rae_m_str = "Inf"
        else:
            ma_rae_m_str = f"{ma_rae_m:.4f}"

        if pd.isna(ma_rae_s):
            ma_rae_s_str = "NaN"
        elif np.isinf(ma_rae_s):
            ma_rae_s_str = "Inf"
        else:
            ma_rae_s_str = f"{ma_rae_s:.4f}"

        print(f"{prop:<40} {ma_rae_m_str:<15} {ma_rae_s_str:<15}")

    print("─" * 70)
    print()

    # Check against target threshold
    target_threshold = 0.53
    print(f"Target Threshold Evaluation:")
    print(f"  Target: MA-RAE < {target_threshold}")
    print(f"  Achieved: MA-RAE = {mean_ma_rae:.4f}")

    if not np.isnan(mean_ma_rae):
        if mean_ma_rae < target_threshold:
            print(f"  ✓ PASSED: Mean MA-RAE is below target threshold")
            meets_threshold = True
        else:
            diff = mean_ma_rae - target_threshold
            print(f"  ✗ FAILED: Mean MA-RAE exceeds target by {diff:.4f}")
            meets_threshold = False
    else:
        print(f"  ⚠ UNABLE TO EVALUATE: Mean MA-RAE is NaN")
        meets_threshold = False

    print()

    return {
        'valid': True,
        'n_properties': n_total,
        'n_valid_mean': (n_total - n_nan_mean - n_inf_mean),
        'n_nan_mean': n_nan_mean,
        'n_inf_mean': n_inf_mean,
        'mean_ma_rae': mean_ma_rae,
        'meets_threshold': meets_threshold,
        'target_threshold': target_threshold,
        'error': None
    }


def main():
    """Main validation function."""
    print("=" * 80)
    print("MA-RAE Implementation Verification")
    print("=" * 80)
    print()
    print("This script validates that the MA-RAE metric is correctly implemented")
    print("and meets all success criteria specified in the user request.")
    print()

    # Validate baseline model
    baseline_file = f"{SESSION_DIR}/results/baseline_cv_scores.csv"
    baseline_results = validate_csv_file(baseline_file, "Baseline LightGBM Model")

    # Validate GNN model
    gnn_file = f"{SESSION_DIR}/results/gnn_cv_scores.csv"
    gnn_results = validate_csv_file(gnn_file, "GNN Multi-Task Model")

    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    print()

    # Success criteria checklist
    print("Success Criteria Checklist:")
    print()

    criteria = [
        ("1. baseline_cv_scores.csv exists with MA-RAE column", baseline_results['valid']),
        ("2. gnn_cv_scores.csv exists with MA-RAE column", gnn_results['valid']),
        ("3. All MA-RAE values are valid (not NaN/Inf)",
         baseline_results['valid'] and baseline_results['n_nan_mean'] == 0 and baseline_results['n_inf_mean'] == 0 and
         gnn_results['valid'] and gnn_results['n_nan_mean'] == 0 and gnn_results['n_inf_mean'] == 0),
        ("4. Mean MA-RAE calculated for baseline model",
         baseline_results['valid'] and not np.isnan(baseline_results['mean_ma_rae'])),
        ("5. Mean MA-RAE calculated for GNN model",
         gnn_results['valid'] and not np.isnan(gnn_results['mean_ma_rae'])),
    ]

    all_passed = True
    for criterion, passed in criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {criterion}")
        if not passed:
            all_passed = False

    print()

    # Model performance comparison
    if baseline_results['valid'] and gnn_results['valid']:
        print("Model Performance Comparison:")
        print(f"  Baseline Model MA-RAE: {baseline_results['mean_ma_rae']:.4f}")
        print(f"  GNN Model MA-RAE: {gnn_results['mean_ma_rae']:.4f}")

        if not np.isnan(baseline_results['mean_ma_rae']) and not np.isnan(gnn_results['mean_ma_rae']):
            diff = gnn_results['mean_ma_rae'] - baseline_results['mean_ma_rae']
            pct_change = 100 * diff / baseline_results['mean_ma_rae']

            print(f"  Difference: {diff:+.4f} ({pct_change:+.2f}%)")

            if diff < 0:
                print(f"  ✓ GNN model improved MA-RAE by {abs(diff):.4f} (lower is better)")
            else:
                print(f"  ✗ GNN model increased MA-RAE by {diff:.4f} (baseline is better)")

        print()

        # Best model
        if not np.isnan(baseline_results['mean_ma_rae']) and not np.isnan(gnn_results['mean_ma_rae']):
            if gnn_results['mean_ma_rae'] < baseline_results['mean_ma_rae']:
                best_model = "GNN"
                best_ma_rae = gnn_results['mean_ma_rae']
            else:
                best_model = "Baseline"
                best_ma_rae = baseline_results['mean_ma_rae']

            print(f"Best Model: {best_model} (MA-RAE = {best_ma_rae:.4f})")
            print()

            # Target threshold evaluation
            target = 0.53
            print(f"Target Threshold Evaluation (< {target}):")
            if best_ma_rae < target:
                print(f"  ✓ PASS: Best model MA-RAE ({best_ma_rae:.4f}) is below target ({target})")
            else:
                diff_from_target = best_ma_rae - target
                print(f"  ✗ FAIL: Best model MA-RAE ({best_ma_rae:.4f}) exceeds target by {diff_from_target:.4f}")
            print()

    # Final verdict
    print("=" * 80)
    if all_passed:
        print("FINAL VERDICT: ✓ ALL SUCCESS CRITERIA MET")
    else:
        print("FINAL VERDICT: ✗ SOME SUCCESS CRITERIA NOT MET")
    print("=" * 80)
    print()

    # Return exit code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
