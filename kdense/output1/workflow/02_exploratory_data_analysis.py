#!/usr/bin/env python3
"""
Step 2: Comprehensive Exploratory Data Analysis
Analyzes training data to understand distributions, missing values, and correlations.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Set reproducibility
np.random.seed(42)

# Define paths
SESSION_DIR = Path("/app/sandbox/session_20251205_152206_4285cc85e60d")
RESULTS_DIR = SESSION_DIR / "results"

print("\n" + "="*70)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*70)

# Load data
print("\n[Loading data...]")
df_train = pd.read_pickle(RESULTS_DIR / "train_data.pkl")
with open(RESULTS_DIR / "target_properties.json", 'r') as f:
    target_properties = json.load(f)

print(f"✓ Data loaded: {df_train.shape[0]:,} samples, {len(target_properties)} target properties")

# 1. Dataset Shape
print("\n" + "="*70)
print("1. DATASET SHAPE")
print("="*70)
print(f"Number of rows (samples): {df_train.shape[0]:,}")
print(f"Number of columns: {df_train.shape[1]}")
print(f"Target properties: {len(target_properties)}")

# 2. Missing Value Analysis
print("\n" + "="*70)
print("2. MISSING VALUE ANALYSIS")
print("="*70)

missing_stats = []
for prop in target_properties:
    n_missing = df_train[prop].isna().sum()
    pct_missing = (n_missing / len(df_train)) * 100
    missing_stats.append({
        'Property': prop,
        'Missing_Count': n_missing,
        'Missing_Percentage': pct_missing,
        'Available_Count': len(df_train) - n_missing
    })

df_missing = pd.DataFrame(missing_stats)
print("\nMissing Value Statistics:")
print(df_missing.to_string(index=False))

# Save missing value statistics
df_missing.to_csv(RESULTS_DIR / "missing_value_statistics.csv", index=False)
print(f"\n✓ Missing value statistics saved to: missing_value_statistics.csv")

# 3. Descriptive Statistics
print("\n" + "="*70)
print("3. DESCRIPTIVE STATISTICS")
print("="*70)

desc_stats = df_train[target_properties].describe()
print("\nBasic descriptive statistics (count, mean, std, min, max):")
print(desc_stats)

# Save descriptive statistics
desc_stats.to_csv(RESULTS_DIR / "descriptive_statistics.csv")
print(f"\n✓ Descriptive statistics saved to: descriptive_statistics.csv")

# Additional statistics
print("\n" + "-"*70)
print("Additional Statistics:")
print("-"*70)

additional_stats = []
for prop in target_properties:
    data = df_train[prop].dropna()
    if len(data) > 0:
        additional_stats.append({
            'Property': prop,
            'Median': data.median(),
            'Q1': data.quantile(0.25),
            'Q3': data.quantile(0.75),
            'IQR': data.quantile(0.75) - data.quantile(0.25),
            'Skewness': data.skew(),
            'Kurtosis': data.kurtosis()
        })

df_additional = pd.DataFrame(additional_stats)
print(df_additional.to_string(index=False))

df_additional.to_csv(RESULTS_DIR / "additional_statistics.csv", index=False)

# 4. Correlation Matrix
print("\n" + "="*70)
print("4. CORRELATION MATRIX")
print("="*70)

# Calculate correlation matrix (pairwise complete observations)
corr_matrix = df_train[target_properties].corr(method='pearson')
print("\nPearson correlation matrix:")
print(corr_matrix.round(3))

# Save correlation matrix
corr_matrix.to_csv(RESULTS_DIR / "correlation_matrix.csv")
print(f"\n✓ Correlation matrix saved to: correlation_matrix.csv")

# Identify high correlations (|r| > 0.5)
print("\n" + "-"*70)
print("Strong Correlations (|r| > 0.5):")
print("-"*70)

strong_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.5:
            strong_corr.append({
                'Property_1': corr_matrix.columns[i],
                'Property_2': corr_matrix.columns[j],
                'Correlation': corr_val
            })

if strong_corr:
    df_strong_corr = pd.DataFrame(strong_corr)
    df_strong_corr = df_strong_corr.sort_values('Correlation', key=abs, ascending=False)
    print(df_strong_corr.to_string(index=False))
    df_strong_corr.to_csv(RESULTS_DIR / "strong_correlations.csv", index=False)
else:
    print("No strong correlations (|r| > 0.5) found between properties")

# 5. Data Range and Distribution Summary
print("\n" + "="*70)
print("5. DATA RANGE SUMMARY")
print("="*70)

range_summary = []
for prop in target_properties:
    data = df_train[prop].dropna()
    if len(data) > 0:
        range_summary.append({
            'Property': prop,
            'Min': data.min(),
            'Max': data.max(),
            'Range': data.max() - data.min(),
            'Mean': data.mean(),
            'Std': data.std(),
            'CV': (data.std() / abs(data.mean())) * 100 if data.mean() != 0 else np.nan
        })

df_range = pd.DataFrame(range_summary)
print(df_range.to_string(index=False))

df_range.to_csv(RESULTS_DIR / "data_range_summary.csv", index=False)

# 6. Summary Report
print("\n" + "="*70)
print("EDA SUMMARY")
print("="*70)

print(f"\n✓ Dataset contains {df_train.shape[0]:,} molecules")
print(f"✓ {len(target_properties)} target properties analyzed")
print(f"✓ Missing data ranges from {df_missing['Missing_Percentage'].min():.1f}% to {df_missing['Missing_Percentage'].max():.1f}%")

# Find properties with most/least missing data
most_missing = df_missing.loc[df_missing['Missing_Percentage'].idxmax()]
least_missing = df_missing.loc[df_missing['Missing_Percentage'].idxmin()]

print(f"✓ Most missing: {most_missing['Property']} ({most_missing['Missing_Percentage']:.1f}%)")
print(f"✓ Least missing: {least_missing['Property']} ({least_missing['Missing_Percentage']:.1f}%)")

if strong_corr:
    print(f"✓ Found {len(strong_corr)} strong correlations (|r| > 0.5)")
    strongest = df_strong_corr.iloc[0]
    print(f"✓ Strongest correlation: {strongest['Property_1']} ↔ {strongest['Property_2']} (r = {strongest['Correlation']:.3f})")
else:
    print(f"✓ No strong correlations (|r| > 0.5) found")

print("\n✓ All EDA outputs saved to results directory")
print("\nExploratory Data Analysis completed successfully!")
