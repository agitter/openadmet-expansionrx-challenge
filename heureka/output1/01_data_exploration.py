#!/usr/bin/env python3
"""
OpenADMET ExpansionRx Blind Challenge - Data Exploration
Purpose: Explore training/test data structure, missingness patterns, and target distributions
Date: 2025-12-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Load data
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

train_df = pd.read_csv('expansion_data_train.csv')
test_df = pd.read_csv('expansion_data_test_blinded.csv')

print(f"\nTraining data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Column names
target_cols = ['LogD', 'KSOL', 'HLM CLint', 'MLM CLint', 'Caco-2 Permeability Papp A>B',
               'Caco-2 Permeability Efflux', 'MPPB', 'MBPB', 'MGMB']

print(f"\nTraining columns: {train_df.columns.tolist()}")
print(f"\nTest columns: {test_df.columns.tolist()}")

# Basic statistics
print("\n" + "=" * 80)
print("TARGET STATISTICS")
print("=" * 80)

for col in target_cols:
    n_values = train_df[col].notna().sum()
    pct = n_values / len(train_df) * 100
    if n_values > 0:
        print(f"\n{col}:")
        print(f"  Non-null: {n_values} ({pct:.1f}%)")
        print(f"  Min: {train_df[col].min():.3f}")
        print(f"  Max: {train_df[col].max():.3f}")
        print(f"  Mean: {train_df[col].mean():.3f}")
        print(f"  Std: {train_df[col].std():.3f}")
        print(f"  Median: {train_df[col].median():.3f}")
    else:
        print(f"\n{col}: No data available")

# Missingness pattern analysis
print("\n" + "=" * 80)
print("MISSINGNESS PATTERN ANALYSIS")
print("=" * 80)

# Create missingness matrix
missing_matrix = train_df[target_cols].isna()
missing_counts = missing_matrix.sum()
print("\nMissing values per target:")
for col in target_cols:
    n_missing = missing_counts[col]
    pct_missing = n_missing / len(train_df) * 100
    print(f"  {col}: {n_missing} ({pct_missing:.1f}% missing)")

# Co-missingness analysis - how often are targets missing together
print("\n\nCo-occurrence of non-missing values:")
non_missing_matrix = train_df[target_cols].notna()
cooccurrence = non_missing_matrix.astype(int).T.dot(non_missing_matrix.astype(int))
print(pd.DataFrame(cooccurrence, index=target_cols, columns=target_cols).to_string())

# Hierarchical missingness pattern
print("\n\nUnique missingness patterns (top 20):")
pattern_df = pd.DataFrame({col: train_df[col].notna().astype(int) for col in target_cols})
patterns = pattern_df.groupby(target_cols).size().sort_values(ascending=False)
print(patterns.head(20))

# Target correlations
print("\n" + "=" * 80)
print("TARGET CORRELATIONS")
print("=" * 80)

# Pairwise correlations (only where both values exist)
corr_matrix = train_df[target_cols].corr(method='spearman')
print("\nSpearman correlation matrix:")
print(corr_matrix.round(3).to_string())

# SMILES length analysis
print("\n" + "=" * 80)
print("SMILES LENGTH ANALYSIS (Distribution Shift)")
print("=" * 80)

train_df['smiles_len'] = train_df['SMILES'].str.len()
test_df['smiles_len'] = test_df['SMILES'].str.len()

print(f"\nTraining SMILES length:")
print(f"  Mean: {train_df['smiles_len'].mean():.1f}")
print(f"  Median: {train_df['smiles_len'].median():.1f}")
print(f"  Min: {train_df['smiles_len'].min()}")
print(f"  Max: {train_df['smiles_len'].max()}")

print(f"\nTest SMILES length:")
print(f"  Mean: {test_df['smiles_len'].mean():.1f}")
print(f"  Median: {test_df['smiles_len'].median():.1f}")
print(f"  Min: {test_df['smiles_len'].min()}")
print(f"  Max: {test_df['smiles_len'].max()}")

# KS test for distribution difference
ks_stat, ks_pval = stats.ks_2samp(train_df['smiles_len'], test_df['smiles_len'])
print(f"\nKS test for SMILES length distribution shift:")
print(f"  KS statistic: {ks_stat:.4f}")
print(f"  p-value: {ks_pval:.2e}")

# Create visualizations
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Figure 1: Missingness heatmap
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. Missingness pattern
ax1 = axes[0, 0]
missing_pct = (train_df[target_cols].isna().sum() / len(train_df) * 100).values
ax1.barh(range(len(target_cols)), missing_pct, color='steelblue')
ax1.set_yticks(range(len(target_cols)))
ax1.set_yticklabels(target_cols)
ax1.set_xlabel('Percentage Missing (%)')
ax1.set_title('Missing Data per Target Property')
ax1.invert_yaxis()
for i, v in enumerate(missing_pct):
    ax1.text(v + 1, i, f'{v:.1f}%', va='center')

# 2. Target correlations heatmap
ax2 = axes[0, 1]
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax2, vmin=-1, vmax=1, square=True)
ax2.set_title('Target Property Correlations (Spearman)')

# 3. SMILES length distribution
ax3 = axes[1, 0]
ax3.hist(train_df['smiles_len'], bins=50, alpha=0.7, label='Training', density=True)
ax3.hist(test_df['smiles_len'], bins=50, alpha=0.7, label='Test', density=True)
ax3.set_xlabel('SMILES String Length')
ax3.set_ylabel('Density')
ax3.set_title(f'SMILES Length Distribution Shift (KS p={ks_pval:.2e})')
ax3.legend()
ax3.axvline(train_df['smiles_len'].mean(), color='blue', linestyle='--', alpha=0.5)
ax3.axvline(test_df['smiles_len'].mean(), color='orange', linestyle='--', alpha=0.5)

# 4. Target distributions
ax4 = axes[1, 1]
# Pick a few representative targets
sample_targets = ['HLM CLint', 'MLM CLint', 'LogD', 'KSOL']
colors = plt.cm.tab10.colors
for i, target in enumerate(sample_targets):
    data = train_df[target].dropna()
    if len(data) > 0:
        ax4.hist(data, bins=30, alpha=0.5, label=f'{target} (n={len(data)})', color=colors[i])
ax4.set_xlabel('Value')
ax4.set_ylabel('Count')
ax4.set_title('Target Value Distributions (Selected)')
ax4.legend()
ax4.set_yscale('log')

plt.tight_layout()
plt.savefig('data_exploration_plots.png', dpi=150, bbox_inches='tight')
print("Saved: data_exploration_plots.png")
plt.close()

# Figure 2: More detailed target distributions
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(target_cols):
    ax = axes[i]
    data = train_df[col].dropna()
    if len(data) > 10:
        ax.hist(data, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
        ax.set_title(f'{col}\n(n={len(data)}, mean={data.mean():.2f})')
        ax.axvline(data.mean(), color='red', linestyle='--', alpha=0.7)
        ax.axvline(data.median(), color='green', linestyle='--', alpha=0.7)
    else:
        ax.text(0.5, 0.5, f'Insufficient data\nn={len(data)}',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(col)

plt.tight_layout()
plt.savefig('target_distributions.png', dpi=150, bbox_inches='tight')
print("Saved: target_distributions.png")
plt.close()

# Analyze log-transformation needs
print("\n" + "=" * 80)
print("LOG-TRANSFORMATION ANALYSIS")
print("=" * 80)

for col in target_cols:
    data = train_df[col].dropna()
    if len(data) > 30:
        # Check skewness
        skewness = stats.skew(data)
        # Check if all positive (can log-transform)
        is_positive = (data > 0).all()

        print(f"\n{col}:")
        print(f"  Skewness: {skewness:.3f}")
        print(f"  All positive: {is_positive}")

        if is_positive and abs(skewness) > 1:
            log_data = np.log1p(data)
            log_skewness = stats.skew(log_data)
            print(f"  Log-transformed skewness: {log_skewness:.3f}")
            print(f"  Recommendation: Consider log-transformation")
        elif abs(skewness) <= 1:
            print(f"  Recommendation: Distribution is reasonably symmetric")

# Summary statistics for validation split planning
print("\n" + "=" * 80)
print("VALIDATION SPLIT PLANNING")
print("=" * 80)

# Identify compounds with most measured properties (for stratification)
n_measured = train_df[target_cols].notna().sum(axis=1)
print(f"\nNumber of measured properties per compound:")
print(n_measured.value_counts().sort_index())

# Compounds with all properties vs sparse
print(f"\nCompounds with all 9 properties: {(n_measured == 9).sum()}")
print(f"Compounds with >= 5 properties: {(n_measured >= 5).sum()}")
print(f"Compounds with >= 3 properties: {(n_measured >= 3).sum()}")
print(f"Compounds with < 3 properties: {(n_measured < 3).sum()}")

# Distribution shift - longer SMILES in test
print(f"\nFor distribution shift validation:")
print(f"Training molecules with SMILES len >= 50: {(train_df['smiles_len'] >= 50).sum()}")
print(f"Training molecules with SMILES len >= 60: {(train_df['smiles_len'] >= 60).sum()}")
print(f"Training molecules with SMILES len >= 70: {(train_df['smiles_len'] >= 70).sum()}")

print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)

# Save summary statistics
summary_stats = pd.DataFrame({
    'property': target_cols,
    'n_samples': [train_df[col].notna().sum() for col in target_cols],
    'pct_available': [train_df[col].notna().sum() / len(train_df) * 100 for col in target_cols],
    'mean': [train_df[col].mean() for col in target_cols],
    'std': [train_df[col].std() for col in target_cols],
    'min': [train_df[col].min() for col in target_cols],
    'max': [train_df[col].max() for col in target_cols],
    'median': [train_df[col].median() for col in target_cols],
    'skewness': [stats.skew(train_df[col].dropna()) if train_df[col].notna().sum() > 10 else np.nan for col in target_cols]
})
summary_stats.to_csv('target_summary_stats.csv', index=False)
print("\nSaved: target_summary_stats.csv")
