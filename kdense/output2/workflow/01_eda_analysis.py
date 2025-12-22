#!/usr/bin/env python3
"""
Exploratory Data Analysis for OpenADMET ExpansionRx Challenge
Step 1: Context & Data Analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
BASE_DIR = Path('/app/sandbox/session_20251217_085238_bf1de403d101')
USER_DATA_DIR = BASE_DIR / 'user_data'
RESULTS_DIR = BASE_DIR / 'results'

# Create results directory if needed
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("OpenADMET ExpansionRx Challenge - Exploratory Data Analysis")
print("=" * 80)

# Load training data
print("\n[1/4] Loading training data...")
train_file = USER_DATA_DIR / 'expansion_data_train.csv'
train_df = pd.read_csv(train_file)
print(f"✓ Training data loaded: {train_df.shape[0]} rows × {train_df.shape[1]} columns")

# Load test data
print("\n[2/4] Loading test data (blinded)...")
test_file = USER_DATA_DIR / 'expansion_data_test_blinded.csv'
test_df = pd.read_csv(test_file)
print(f"✓ Test data loaded: {test_df.shape[0]} rows × {test_df.shape[1]} columns")

# Identify columns
print("\n[3/4] Analyzing data structure...")
print(f"\nTraining columns: {list(train_df.columns)}")
print(f"\nTest columns: {list(test_df.columns)}")

# Identify SMILES column and target properties
# Based on documentation: Molecule Name, Smiles, and 9 ADMET properties
smiles_col = 'Smiles'
identifier_cols = ['Molecule Name', 'Smiles']

# The 9 target properties from documentation
target_properties = [
    'LogD',
    'KSol',
    'MLM CLint',
    'HLM CLint',
    'Caco-2 Permeability Efflux',
    'Caco-2 Permeability Papp A>B',
    'MPPB',
    'MBPB',
    'MGMB'
]

print(f"\n✓ SMILES column: {smiles_col}")
print(f"✓ Identifier columns: {identifier_cols}")
print(f"✓ Target properties ({len(target_properties)}):")
for i, prop in enumerate(target_properties, 1):
    print(f"   {i}. {prop}")

# Analyze missing values
print("\n[4/4] Computing missing value statistics...")
missing_stats = {}
for prop in target_properties:
    if prop in train_df.columns:
        total = len(train_df)
        missing = train_df[prop].isna().sum()
        pct_missing = (missing / total) * 100
        missing_stats[prop] = {
            'total': total,
            'missing': missing,
            'present': total - missing,
            'pct_missing': pct_missing,
            'pct_present': 100 - pct_missing
        }
        print(f"  {prop}: {missing}/{total} missing ({pct_missing:.2f}%)")

# Compute basic statistics for each target
print("\n" + "=" * 80)
print("BASIC STATISTICS FOR TARGET PROPERTIES")
print("=" * 80)

stats_summary = {}
for prop in target_properties:
    if prop in train_df.columns:
        data = train_df[prop].dropna()
        stats_summary[prop] = {
            'count': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            '25%': data.quantile(0.25),
            'median': data.median(),
            '75%': data.quantile(0.75),
            'max': data.max()
        }

        print(f"\n{prop}:")
        print(f"  Count:  {stats_summary[prop]['count']}")
        print(f"  Mean:   {stats_summary[prop]['mean']:.4f}")
        print(f"  Std:    {stats_summary[prop]['std']:.4f}")
        print(f"  Min:    {stats_summary[prop]['min']:.4f}")
        print(f"  25%:    {stats_summary[prop]['25%']:.4f}")
        print(f"  Median: {stats_summary[prop]['median']:.4f}")
        print(f"  75%:    {stats_summary[prop]['75%']:.4f}")
        print(f"  Max:    {stats_summary[prop]['max']:.4f}")

# Save comprehensive summary to file
print("\n" + "=" * 80)
print("SAVING SUMMARY TO FILE")
print("=" * 80)

summary_file = RESULTS_DIR / 'eda_summary.txt'
with open(summary_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("OpenADMET ExpansionRx Challenge - EDA Summary\n")
    f.write("=" * 80 + "\n\n")

    # Documentation findings
    f.write("1. CHALLENGE REQUIREMENTS (from documentation)\n")
    f.write("-" * 80 + "\n")
    f.write("Evaluation Metrics:\n")
    f.write("  - Macro-Averaged Relative Absolute Error (MA-RAE)\n")
    f.write("  - Individual metrics available in challenge code repository\n")
    f.write("  - Metrics computed per endpoint and macro-averaged\n\n")

    f.write("Submission Format:\n")
    f.write("  - CSV file with same column names as test set\n")
    f.write("  - Must include all 9 target property columns\n")
    f.write("  - Can submit zeros for properties not predicted\n")
    f.write("  - Column names must match exactly\n\n")

    f.write("Data Handling Notes:\n")
    f.write("  - Log transforms: add 1 to values before log transform (to handle zeros)\n")
    f.write("  - External data allowed for training\n\n")

    # Data structure
    f.write("2. DATA STRUCTURE\n")
    f.write("-" * 80 + "\n")
    f.write(f"Training set: {train_df.shape[0]} molecules × {train_df.shape[1]} columns\n")
    f.write(f"Test set:     {test_df.shape[0]} molecules × {test_df.shape[1]} columns\n\n")

    f.write(f"SMILES column: {smiles_col}\n")
    f.write(f"Identifier columns: {', '.join(identifier_cols)}\n\n")

    f.write(f"Target Properties ({len(target_properties)}):\n")
    for i, prop in enumerate(target_properties, 1):
        f.write(f"  {i}. {prop}\n")
    f.write("\n")

    # Missing values
    f.write("3. MISSING VALUE ANALYSIS\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Property':<40} {'Missing':<10} {'Present':<10} {'% Missing':<12}\n")
    f.write("-" * 80 + "\n")
    for prop in target_properties:
        if prop in missing_stats:
            stats = missing_stats[prop]
            f.write(f"{prop:<40} {stats['missing']:<10} {stats['present']:<10} {stats['pct_missing']:<12.2f}\n")
    f.write("\n")

    # Descriptive statistics
    f.write("4. DESCRIPTIVE STATISTICS\n")
    f.write("-" * 80 + "\n")
    for prop in target_properties:
        if prop in stats_summary:
            stats = stats_summary[prop]
            f.write(f"\n{prop}:\n")
            f.write(f"  Count:      {stats['count']}\n")
            f.write(f"  Mean:       {stats['mean']:.4f}\n")
            f.write(f"  Std Dev:    {stats['std']:.4f}\n")
            f.write(f"  Min:        {stats['min']:.4f}\n")
            f.write(f"  25th %ile:  {stats['25%']:.4f}\n")
            f.write(f"  Median:     {stats['median']:.4f}\n")
            f.write(f"  75th %ile:  {stats['75%']:.4f}\n")
            f.write(f"  Max:        {stats['max']:.4f}\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF SUMMARY\n")
    f.write("=" * 80 + "\n")

print(f"\n✓ Summary saved to: {summary_file}")
print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)
