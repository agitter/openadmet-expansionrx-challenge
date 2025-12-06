#!/usr/bin/env python3
"""
Train Baseline LightGBM Model with Scaffold-Based Cross-Validation

This script trains LightGBM models for all 9 ADMET properties using:
1. Molecular features (Morgan fingerprints + RDKit descriptors)
2. Scaffold-based 5-fold cross-validation for robust evaluation
3. Log-transformation for skewed properties
4. Spearman correlation and MA-RAE metrics

Author: K-Dense Coding Agent
Date: 2025-12-05
"""

import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
import sys
import time
import warnings
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Session directory - MUST use absolute paths
SESSION_DIR = "/app/sandbox/session_20251205_152206_4285cc85e60d"

# Properties that require log-transform (identified in EDA)
LOG_TRANSFORM_PROPERTIES = ['HLM CLint', 'MLM CLint', 'Caco-2 Permeability Efflux', 'MBPB']


def generate_scaffold(smiles: str) -> str:
    """
    Generate Murcko scaffold from SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES string

    Returns
    -------
    str
        Scaffold SMILES string, or original SMILES if scaffold generation fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold if scaffold else smiles
    except Exception:
        return smiles


def scaffold_split(smiles_list, n_splits=5, random_seed=42):
    """
    Create scaffold-based splits for cross-validation.

    Molecules with the same scaffold are kept in the same fold.

    Parameters
    ----------
    smiles_list : list
        List of SMILES strings
    n_splits : int
        Number of folds
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    list of tuples
        List of (train_idx, test_idx) for each fold
    """
    print(f"  Generating scaffolds for {len(smiles_list)} molecules...")

    # Generate scaffolds for all molecules
    scaffolds = [generate_scaffold(smiles) for smiles in smiles_list]

    # Group molecule indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, scaffold in enumerate(scaffolds):
        scaffold_to_indices[scaffold].append(idx)

    # Get unique scaffolds and shuffle them
    unique_scaffolds = list(scaffold_to_indices.keys())
    np.random.seed(random_seed)
    np.random.shuffle(unique_scaffolds)

    print(f"  Found {len(unique_scaffolds)} unique scaffolds")

    # Distribute scaffolds to folds trying to balance molecule counts
    folds = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    for scaffold in unique_scaffolds:
        indices = scaffold_to_indices[scaffold]
        # Add to the fold with fewest molecules
        smallest_fold = np.argmin(fold_sizes)
        folds[smallest_fold].extend(indices)
        fold_sizes[smallest_fold] += len(indices)

    print(f"  Fold sizes: {fold_sizes}")

    # Create train/test splits
    splits = []
    for i in range(n_splits):
        test_idx = np.array(folds[i])
        train_idx = np.array([idx for j in range(n_splits) if j != i for idx in folds[j]])
        splits.append((train_idx, test_idx))

    return splits


def calculate_ma_rae(y_true, y_pred):
    """
    Calculate Mean Absolute-Relative Absolute Error (MA-RAE).

    MA-RAE = MAE / MAE_baseline
    where MAE_baseline is the MAE of predicting the mean

    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    float
        MA-RAE score
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mae_baseline = np.mean(np.abs(y_true - np.mean(y_true)))
    return mae / mae_baseline if mae_baseline > 0 else np.nan


def train_and_evaluate_property(X, y, smiles_list, property_name, apply_log_transform=False, n_splits=5):
    """
    Train and evaluate model for a single property using scaffold-based CV.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    smiles_list : list
        List of SMILES strings for scaffold generation
    property_name : str
        Name of the property
    apply_log_transform : bool
        Whether to apply log transform to target
    n_splits : int
        Number of CV folds

    Returns
    -------
    dict
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"Property: {property_name}")
    print(f"{'='*80}")

    # Filter out missing values
    valid_mask = ~np.isnan(y)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    smiles_valid = [smiles for smiles, valid in zip(smiles_list, valid_mask) if valid]

    n_samples = len(y_valid)
    print(f"  Samples with non-missing values: {n_samples} / {len(y)} ({100*n_samples/len(y):.1f}%)")

    if n_samples < 50:
        print(f"  ⚠ Warning: Only {n_samples} samples available - skipping this property")
        return {
            'property': property_name,
            'n_samples': n_samples,
            'spearman_mean': np.nan,
            'spearman_std': np.nan,
            'ma_rae_mean': np.nan,
            'ma_rae_std': np.nan,
            'log_transformed': apply_log_transform,
            'error': 'Insufficient samples'
        }

    # Apply log transform if requested
    if apply_log_transform:
        y_transformed = np.log10(y_valid + 1)
        print(f"  Applied log10(x+1) transformation")
        print(f"    Original range: [{y_valid.min():.3f}, {y_valid.max():.3f}]")
        print(f"    Transformed range: [{y_transformed.min():.3f}, {y_transformed.max():.3f}]")
    else:
        y_transformed = y_valid

    # Generate scaffold splits
    splits = scaffold_split(smiles_valid, n_splits=n_splits)

    # Train and evaluate for each fold
    fold_results = []
    all_y_true = []
    all_y_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"\n  Fold {fold_idx + 1}/{n_splits}")
        print(f"    Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")

        X_train, X_test = X_valid[train_idx], X_valid[test_idx]
        y_train, y_test = y_transformed[train_idx], y_transformed[test_idx]

        # Train LightGBM model
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold_idx,
            verbose=-1,
            n_jobs=1
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predict on test set
        y_pred = model.predict(X_test)

        # Inverse transform if log was applied
        if apply_log_transform:
            y_test_original = 10 ** y_test - 1
            y_pred_original = 10 ** y_pred - 1
            # Clip negative predictions to 0
            y_pred_original = np.maximum(y_pred_original, 0)
        else:
            y_test_original = y_test
            y_pred_original = y_pred

        # Calculate metrics
        spearman_corr, _ = spearmanr(y_test_original, y_pred_original)
        ma_rae = calculate_ma_rae(y_test_original, y_pred_original)

        print(f"    Training time: {train_time:.2f}s")
        print(f"    Spearman correlation: {spearman_corr:.4f}")
        print(f"    MA-RAE: {ma_rae:.4f}")

        fold_results.append({
            'spearman': spearman_corr,
            'ma_rae': ma_rae
        })

        # Store for overall metrics
        all_y_true.extend(y_test_original)
        all_y_pred.extend(y_pred_original)

    # Aggregate results across folds
    spearman_scores = [r['spearman'] for r in fold_results]
    ma_rae_scores = [r['ma_rae'] for r in fold_results]

    spearman_mean = np.mean(spearman_scores)
    spearman_std = np.std(spearman_scores)
    ma_rae_mean = np.mean(ma_rae_scores)
    ma_rae_std = np.std(ma_rae_scores)

    # Overall metrics using all out-of-fold predictions
    overall_spearman, _ = spearmanr(all_y_true, all_y_pred)
    overall_ma_rae = calculate_ma_rae(np.array(all_y_true), np.array(all_y_pred))

    print(f"\n  {'─'*76}")
    print(f"  Cross-Validation Results:")
    print(f"    Spearman: {spearman_mean:.4f} ± {spearman_std:.4f}")
    print(f"    MA-RAE: {ma_rae_mean:.4f} ± {ma_rae_std:.4f}")
    print(f"  Overall (all out-of-fold predictions):")
    print(f"    Spearman: {overall_spearman:.4f}")
    print(f"    MA-RAE: {overall_ma_rae:.4f}")
    print(f"  {'─'*76}")

    return {
        'property': property_name,
        'n_samples': n_samples,
        'spearman_mean': spearman_mean,
        'spearman_std': spearman_std,
        'ma_rae_mean': ma_rae_mean,
        'ma_rae_std': ma_rae_std,
        'overall_spearman': overall_spearman,
        'overall_ma_rae': overall_ma_rae,
        'log_transformed': apply_log_transform,
        'error': None
    }


def main():
    """Main execution function."""
    print("=" * 80)
    print("Baseline LightGBM Model Training with Scaffold-Based Cross-Validation")
    print("=" * 80)
    print()

    # Step 1: Load features
    print("Step 1: Loading features...")
    features_path = f"{SESSION_DIR}/results/baseline_features_train.pkl"

    try:
        with open(features_path, 'rb') as f:
            features_data = pickle.load(f)
        X = features_data['features']
        molecule_ids = features_data['molecule_ids']
        print(f"✓ Loaded features: {X.shape}")
        print(f"  Feature matrix: {X.shape[0]} molecules × {X.shape[1]} features")
    except Exception as e:
        print(f"✗ Error loading features: {e}")
        sys.exit(1)

    print()

    # Step 2: Load target data
    print("Step 2: Loading target data...")
    train_data_path = f"{SESSION_DIR}/results/train_data.pkl"

    try:
        with open(train_data_path, 'rb') as f:
            train_data = pickle.load(f)
        print(f"✓ Loaded training data: {train_data.shape}")
    except Exception as e:
        print(f"✗ Error loading training data: {e}")
        sys.exit(1)

    # Ensure same ordering
    train_data = train_data.set_index('Molecule Name')
    train_data = train_data.loc[molecule_ids]

    # Extract SMILES for scaffold generation
    smiles_list = train_data['SMILES'].tolist()

    # Identify target properties (exclude Molecule Name and SMILES)
    target_properties = [col for col in train_data.columns if col not in ['SMILES']]
    print(f"  Target properties: {len(target_properties)}")
    for prop in target_properties:
        n_available = train_data[prop].notna().sum()
        print(f"    - {prop}: {n_available} samples ({100*n_available/len(train_data):.1f}%)")

    print()

    # Step 3: Train and evaluate each property
    print("Step 3: Training models for all properties...")
    print()

    results = []
    start_time_total = time.time()

    for idx, property_name in enumerate(target_properties, 1):
        print(f"\n[Property {idx}/{len(target_properties)}]")

        # Get target values
        y = train_data[property_name].values

        # Check if log transform should be applied
        apply_log = property_name in LOG_TRANSFORM_PROPERTIES

        # Train and evaluate
        result = train_and_evaluate_property(
            X=X,
            y=y,
            smiles_list=smiles_list,
            property_name=property_name,
            apply_log_transform=apply_log,
            n_splits=5
        )

        results.append(result)

    total_time = time.time() - start_time_total
    print(f"\n{'='*80}")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"{'='*80}")
    print()

    # Step 4: Save results
    print("Step 4: Saving results...")

    results_df = pd.DataFrame(results)

    # Reorder columns
    column_order = [
        'property', 'n_samples', 'spearman_mean', 'spearman_std',
        'overall_spearman', 'ma_rae_mean', 'ma_rae_std', 'overall_ma_rae',
        'log_transformed', 'error'
    ]
    results_df = results_df[column_order]

    output_path = f"{SESSION_DIR}/results/baseline_cv_scores.csv"
    results_df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"✓ Results saved to: {output_path}")
    print()

    # Step 5: Summary statistics
    print("=" * 80)
    print("BASELINE MODEL PERFORMANCE SUMMARY")
    print("=" * 80)
    print()

    # Filter valid results (exclude those with errors)
    valid_results = results_df[results_df['error'].isna()]

    if len(valid_results) > 0:
        print("Cross-Validation Metrics (mean ± std):")
        print()
        print(f"{'Property':<35} {'N':<7} {'Spearman':<20} {'MA-RAE':<20} {'Log'}")
        print("─" * 90)

        for _, row in valid_results.iterrows():
            prop_name = row['property']
            n_samples = int(row['n_samples'])
            spearman_str = f"{row['spearman_mean']:.3f} ± {row['spearman_std']:.3f}"
            ma_rae_str = f"{row['ma_rae_mean']:.3f} ± {row['ma_rae_std']:.3f}"
            log_str = "Yes" if row['log_transformed'] else "No"

            print(f"{prop_name:<35} {n_samples:<7} {spearman_str:<20} {ma_rae_str:<20} {log_str}")

        print("─" * 90)

        # Overall statistics
        mean_spearman = valid_results['spearman_mean'].mean()
        mean_ma_rae = valid_results['ma_rae_mean'].mean()

        print()
        print(f"Overall Mean Spearman: {mean_spearman:.4f}")
        print(f"Overall Mean MA-RAE: {mean_ma_rae:.4f}")
        print()

        # Compare to literature benchmark
        literature_benchmark = 0.8087
        print(f"Literature Benchmark (Kosmos AI): {literature_benchmark:.4f}")
        diff = mean_spearman - literature_benchmark
        diff_pct = 100 * diff / literature_benchmark
        print(f"Difference: {diff:+.4f} ({diff_pct:+.2f}%)")
        print()

        # Performance by tier
        tier1 = valid_results[valid_results['property'].isin(['LogD', 'KSOL'])]['spearman_mean'].mean()
        tier2 = valid_results[valid_results['property'].isin(['HLM CLint', 'MLM CLint'])]['spearman_mean'].mean()
        tier3 = valid_results[valid_results['property'].isin(['Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux'])]['spearman_mean'].mean()
        tier4 = valid_results[valid_results['property'].isin(['MPPB', 'MBPB', 'MGMB'])]['spearman_mean'].mean()

        print("Performance by Data Availability Tier:")
        print(f"  Tier 1 (>95% complete): {tier1:.4f}")
        print(f"  Tier 2 (70-85% complete): {tier2:.4f}")
        print(f"  Tier 3 (40-60% complete): {tier3:.4f}")
        print(f"  Tier 4 (<25% complete): {tier4:.4f}")
        print()

    else:
        print("⚠ No valid results to summarize")
        print()

    # Report any errors
    error_results = results_df[results_df['error'].notna()]
    if len(error_results) > 0:
        print("Properties with errors:")
        for _, row in error_results.iterrows():
            print(f"  - {row['property']}: {row['error']}")
        print()

    print("=" * 80)
    print("Training complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review results in: results/baseline_cv_scores.csv")
    print("  2. Implement advanced GNN model (Track 2)")
    print("  3. Develop ensemble combining baseline + GNN (Track 3)")
    print()


if __name__ == "__main__":
    main()
