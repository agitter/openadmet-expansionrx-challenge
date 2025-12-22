#!/usr/bin/env python3
"""
Feature Importance Analysis for ADMET Property Prediction Models

This script analyzes feature importance across all 9 trained XGBoost models
to identify key molecular drivers of each ADMET property.

Author: K-Dense System
Date: 2025-12-22
"""

import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
BASE_DIR = Path("/app/sandbox/session_20251217_085238_bf1de403d101")
MODELS_DIR = BASE_DIR / "results" / "models"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Target properties
TARGETS = [
    'LogD', 'KSol', 'HLM', 'MLM', 'Papp', 'Peff', 'MPPB', 'MBPB', 'MGMB'
]

# Property full names for better reporting
PROPERTY_NAMES = {
    'LogD': 'Lipophilicity (LogD)',
    'KSol': 'Kinetic Solubility',
    'HLM': 'Human Liver Microsome Clearance',
    'MLM': 'Mouse Liver Microsome Clearance',
    'Papp': 'Caco-2 Permeability (Papp A>B)',
    'Peff': 'Caco-2 Efflux Ratio',
    'MPPB': 'Mouse Plasma Protein Binding',
    'MBPB': 'Mouse Brain Protein Binding',
    'MGMB': 'Mouse Gastric Mucosa Binding'
}


def load_feature_names(train_file):
    """Load feature names from the featurized training data."""
    print(f"Loading feature names from {train_file}...")
    df = pd.read_csv(train_file, nrows=1)

    # Extract feature columns (fp_* and desc_*)
    feature_cols = [col for col in df.columns if col.startswith('fp_') or col.startswith('desc_')]
    print(f"Found {len(feature_cols)} features ({sum(1 for c in feature_cols if c.startswith('fp_'))} fingerprints, {sum(1 for c in feature_cols if c.startswith('desc_'))} descriptors)")

    return feature_cols


def load_model(model_path):
    """Load an XGBoost model from JSON file."""
    model = xgb.Booster()
    model.load_model(str(model_path))
    return model


def extract_feature_importance(model, feature_names, importance_type='gain'):
    """
    Extract feature importance from XGBoost model.

    Parameters:
    -----------
    model : xgb.Booster
        Trained XGBoost model
    feature_names : list
        List of feature names
    importance_type : str
        Type of importance metric ('gain', 'weight', 'cover')

    Returns:
    --------
    pd.DataFrame
        DataFrame with features and their importance scores, sorted by importance
    """
    # Get importance scores as dictionary
    importance_dict = model.get_score(importance_type=importance_type)

    # The importance dict keys might be actual feature names or indices (f0, f1, etc.)
    importance_mapped = {}
    for feat_id, score in importance_dict.items():
        if feat_id in feature_names:
            # Direct feature name match
            importance_mapped[feat_id] = score
        elif feat_id.startswith('f') and feat_id[1:].isdigit():
            # XGBoost uses f0, f1, f2... as feature identifiers
            idx = int(feat_id[1:])
            if idx < len(feature_names):
                importance_mapped[feature_names[idx]] = score
        else:
            # Skip unknown features
            pass

    # Create DataFrame and sort
    importance_df = pd.DataFrame([
        {'feature': feat, 'importance': score}
        for feat, score in importance_mapped.items()
    ]).sort_values('importance', ascending=False)

    return importance_df


def categorize_feature(feature_name):
    """Categorize feature as fingerprint or descriptor."""
    if feature_name.startswith('fp_'):
        return 'Fingerprint'
    elif feature_name.startswith('desc_'):
        return 'Descriptor'
    else:
        return 'Unknown'


def analyze_all_models():
    """Analyze feature importance for all trained models."""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80 + "\n")

    # Load feature names
    train_file = DATA_DIR / "train_featurized.csv"
    feature_names = load_feature_names(train_file)

    # Store results for all models
    all_results = {}

    # Analyze each target
    for i, target in enumerate(TARGETS, 1):
        print(f"\n[{i}/{len(TARGETS)}] Analyzing {target} ({PROPERTY_NAMES[target]})...")

        model_path = MODELS_DIR / f"model_{target}.json"

        if not model_path.exists():
            print(f"  WARNING: Model file not found: {model_path}")
            continue

        # Load model
        model = load_model(model_path)

        # Extract feature importance
        importance_df = extract_feature_importance(model, feature_names, importance_type='gain')

        # Get top 10 features
        top_features = importance_df.head(10).copy()
        top_features['category'] = top_features['feature'].apply(categorize_feature)

        # Store results
        all_results[target] = {
            'full_name': PROPERTY_NAMES[target],
            'top_features': top_features,
            'total_features': len(importance_df),
            'total_importance': importance_df['importance'].sum()
        }

        # Print summary
        print(f"  Total features with importance: {len(importance_df)}")
        print(f"  Top 10 features account for {100 * top_features['importance'].sum() / importance_df['importance'].sum():.1f}% of total importance")
        print(f"  Top feature: {top_features.iloc[0]['feature']} (importance: {top_features.iloc[0]['importance']:.2f})")

    return all_results


def save_feature_importance_report(results, output_file):
    """Save comprehensive feature importance report to text file."""
    print(f"\nSaving feature importance report to {output_file}...")

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FEATURE IMPORTANCE ANALYSIS REPORT\n")
        f.write("ADMET Property Prediction Models\n")
        f.write("="*80 + "\n\n")

        f.write("Analysis Method: XGBoost Feature Importance (Gain Metric)\n")
        f.write("The 'gain' metric measures the average improvement in accuracy brought by a feature.\n")
        f.write("Higher gain indicates more important features for model predictions.\n\n")

        # Summary for each target
        for target in TARGETS:
            if target not in results:
                continue

            result = results[target]
            f.write("\n" + "-"*80 + "\n")
            f.write(f"{target}: {result['full_name']}\n")
            f.write("-"*80 + "\n\n")

            top_features = result['top_features']

            f.write(f"Total features with importance: {result['total_features']}\n")
            f.write(f"Top 10 features account for {100 * top_features['importance'].sum() / result['total_importance']:.1f}% of total importance\n\n")

            f.write("TOP 10 FEATURES:\n")
            f.write(f"{'Rank':<6} {'Feature':<40} {'Category':<15} {'Importance':<12}\n")
            f.write("-" * 75 + "\n")

            for idx, row in top_features.iterrows():
                rank = top_features.index.get_loc(idx) + 1
                f.write(f"{rank:<6} {row['feature']:<40} {row['category']:<15} {row['importance']:<12.2f}\n")

            # Count feature types in top 10
            fp_count = sum(top_features['category'] == 'Fingerprint')
            desc_count = sum(top_features['category'] == 'Descriptor')

            f.write(f"\nFeature Type Distribution in Top 10:\n")
            f.write(f"  - Morgan Fingerprints: {fp_count}\n")
            f.write(f"  - RDKit Descriptors: {desc_count}\n")

        # Cross-property analysis
        f.write("\n\n" + "="*80 + "\n")
        f.write("CROSS-PROPERTY FEATURE ANALYSIS\n")
        f.write("="*80 + "\n\n")

        # Find features that appear in top 10 across multiple properties
        feature_counts = {}
        for target, result in results.items():
            for feat in result['top_features']['feature']:
                if feat not in feature_counts:
                    feature_counts[feat] = []
                feature_counts[feat].append(target)

        # Features appearing in 3+ properties
        common_features = {feat: props for feat, props in feature_counts.items() if len(props) >= 3}

        if common_features:
            f.write("Features appearing in Top 10 for 3+ properties:\n\n")
            for feat, props in sorted(common_features.items(), key=lambda x: len(x[1]), reverse=True):
                f.write(f"  {feat}:\n")
                f.write(f"    Appears in {len(props)} properties: {', '.join(props)}\n")
        else:
            f.write("No features appear in Top 10 for 3 or more properties.\n")
            f.write("This suggests each ADMET property is driven by distinct molecular features.\n")

    print("Feature importance report saved successfully!")


def main():
    """Main execution function."""
    print("Starting Feature Importance Analysis...")
    print(f"Working directory: {BASE_DIR}")

    # Analyze all models
    results = analyze_all_models()

    if not results:
        print("\nERROR: No results generated. Check model files.")
        return 1

    # Save report
    output_file = RESULTS_DIR / "feature_importance_summary.txt"
    save_feature_importance_report(results, output_file)

    print("\n" + "="*80)
    print("Feature importance analysis completed successfully!")
    print(f"Results saved to: {output_file}")
    print("="*80 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
