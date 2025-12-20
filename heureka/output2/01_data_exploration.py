"""
Data Exploration for OpenADMET ExpansionRx Blind Challenge
============================================================
Analyzes training data structure, missing values, distributions, and correlations
for 9 ADMET properties to be predicted from molecular SMILES strings.

Date: 2024-12-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("OpenADMET ExpansionRx Blind Challenge - Data Exploration")
print("="*80)

# Load training data
print("\n[1] Loading training data...")
train_df = pd.read_csv('expansion_data_train.csv')
print(f"Training set shape: {train_df.shape}")
print(f"Columns: {list(train_df.columns)}")

# Load test data
print("\n[2] Loading test data...")
test_df = pd.read_csv('expansion_data_test_blinded.csv')
print(f"Test set shape: {test_df.shape}")
print(f"Columns: {list(test_df.columns)}")

# Define ADMET properties
admet_properties = [
    'LogD',
    'KSOL',
    'HLM CLint',
    'MLM CLint',
    'Caco-2 Permeability Papp A>B',
    'Caco-2 Permeability Efflux',
    'MPPB',
    'MBPB',
    'MGMB'
]

print(f"\n[3] ADMET Properties to Predict ({len(admet_properties)}):")
for i, prop in enumerate(admet_properties, 1):
    print(f"  {i}. {prop}")

# Data quality assessment
print("\n[4] Data Quality Assessment")
print("-" * 80)

missing_summary = []
for prop in admet_properties:
    total = len(train_df)
    missing = train_df[prop].isna().sum()
    present = total - missing
    missing_pct = (missing / total) * 100
    missing_summary.append({
        'Property': prop,
        'Total': total,
        'Present': present,
        'Missing': missing,
        'Missing %': missing_pct
    })
    print(f"{prop:40s}: {present:4d} / {total:4d} ({missing_pct:5.1f}% missing)")

missing_df = pd.DataFrame(missing_summary)

# Statistical summary for each property
print("\n[5] Statistical Summary of ADMET Properties")
print("-" * 80)
stats_summary = train_df[admet_properties].describe()
print(stats_summary)

# Check for outliers and distribution characteristics
print("\n[6] Distribution Characteristics")
print("-" * 80)
for prop in admet_properties:
    data = train_df[prop].dropna()
    if len(data) > 0:
        skewness = data.skew()
        kurtosis = data.kurtosis()
        print(f"{prop:40s}: Skew={skewness:6.2f}, Kurtosis={kurtosis:6.2f}")

# Correlation analysis (only on complete cases)
print("\n[7] Correlation Analysis")
print("-" * 80)
# Calculate correlation matrix on available data
corr_matrix = train_df[admet_properties].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Find highly correlated pairs
print("\nHighly Correlated Property Pairs (|r| > 0.6):")
high_corr_pairs = []
for i in range(len(admet_properties)):
    for j in range(i+1, len(admet_properties)):
        corr_val = corr_matrix.iloc[i, j]
        if not np.isnan(corr_val) and abs(corr_val) > 0.6:
            high_corr_pairs.append((admet_properties[i], admet_properties[j], corr_val))
            print(f"  {admet_properties[i]} <-> {admet_properties[j]}: r = {corr_val:.3f}")

if not high_corr_pairs:
    print("  No property pairs with |r| > 0.6")

# SMILES validation
print("\n[8] SMILES String Validation")
print("-" * 80)
print(f"Training set - unique SMILES: {train_df['SMILES'].nunique()} / {len(train_df)}")
print(f"Test set - unique SMILES: {test_df['SMILES'].nunique()} / {len(test_df)}")

# Check for duplicates
train_duplicates = train_df['SMILES'].duplicated().sum()
test_duplicates = test_df['SMILES'].duplicated().sum()
print(f"Training set - duplicate SMILES: {train_duplicates}")
print(f"Test set - duplicate SMILES: {test_duplicates}")

# SMILES length statistics
train_df['SMILES_length'] = train_df['SMILES'].str.len()
test_df['SMILES_length'] = test_df['SMILES'].str.len()
print(f"\nTraining SMILES length - Mean: {train_df['SMILES_length'].mean():.1f}, "
      f"Range: [{train_df['SMILES_length'].min()}, {train_df['SMILES_length'].max()}]")
print(f"Test SMILES length - Mean: {test_df['SMILES_length'].mean():.1f}, "
      f"Range: [{test_df['SMILES_length'].min()}, {test_df['SMILES_length'].max()}]")

# Save summary to file
print("\n[9] Saving summary statistics...")
with open('data_exploration_summary.txt', 'w') as f:
    f.write("OpenADMET ExpansionRx Blind Challenge - Data Summary\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Training set: {train_df.shape[0]} compounds, {train_df.shape[1]} columns\n")
    f.write(f"Test set: {test_df.shape[0]} compounds\n")
    f.write(f"Properties to predict: {len(admet_properties)}\n\n")
    f.write("Missing Data Summary:\n")
    f.write("-" * 80 + "\n")
    f.write(missing_df.to_string(index=False))
    f.write("\n\n")
    f.write("Statistical Summary:\n")
    f.write("-" * 80 + "\n")
    f.write(stats_summary.to_string())
    f.write("\n\n")
    f.write("Correlation Matrix:\n")
    f.write("-" * 80 + "\n")
    f.write(corr_matrix.to_string())

print("Summary saved to: data_exploration_summary.txt")

# Create comprehensive visualization
print("\n[10] Creating visualizations...")
fig = plt.figure(figsize=(20, 16))

# 1. Missing data heatmap
ax1 = plt.subplot(3, 3, 1)
missing_matrix = train_df[admet_properties].isna().astype(int)
sns.heatmap(missing_matrix.T, cbar=True, cmap='RdYlGn_r', ax=ax1,
            xticklabels=False, yticklabels=admet_properties)
ax1.set_title('Missing Data Pattern (White=Present, Dark=Missing)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Compound Index')

# 2. Missing data bar chart
ax2 = plt.subplot(3, 3, 2)
missing_df.plot(x='Property', y='Missing %', kind='barh', ax=ax2, legend=False, color='coral')
ax2.set_xlabel('Missing Data (%)', fontsize=10)
ax2.set_ylabel('')
ax2.set_title('Missing Data Percentage by Property', fontsize=12, fontweight='bold')
ax2.set_xlim([0, 100])

# 3. Data availability
ax3 = plt.subplot(3, 3, 3)
data_availability = missing_df[['Property', 'Present', 'Missing']].set_index('Property')
data_availability.plot(kind='barh', stacked=True, ax=ax3, color=['lightgreen', 'lightcoral'])
ax3.set_xlabel('Number of Compounds', fontsize=10)
ax3.set_title('Data Availability by Property', fontsize=12, fontweight='bold')
ax3.legend(['Present', 'Missing'], loc='lower right')

# 4-9. Distribution plots for each property (using 6 slots)
plot_positions = [(3, 3, i) for i in range(4, 10)]
for idx, (prop, pos) in enumerate(zip(admet_properties[:6], plot_positions)):
    ax = plt.subplot(*pos)
    data = train_df[prop].dropna()
    if len(data) > 0:
        ax.hist(data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(data.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {data.median():.2f}')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(f'{prop}\n(n={len(data)}, skew={data.skew():.2f})', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_distributions_part1.png', dpi=300, bbox_inches='tight')
print("Saved: eda_distributions_part1.png")

# Create second figure for remaining properties and correlation
fig2 = plt.figure(figsize=(20, 12))

# Distribution plots for remaining properties
plot_positions2 = [(2, 3, i) for i in range(1, 4)]
for idx, (prop, pos) in enumerate(zip(admet_properties[6:], plot_positions2)):
    ax = plt.subplot(*pos)
    data = train_df[prop].dropna()
    if len(data) > 0:
        ax.hist(data, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.axvline(data.median(), color='darkred', linestyle='--', linewidth=2, label=f'Median: {data.median():.2f}')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(f'{prop}\n(n={len(data)}, skew={data.skew():.2f})', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

# Correlation heatmap
ax_corr = plt.subplot(2, 3, (4, 6))
mask = np.isnan(corr_matrix)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax_corr,
            mask=mask, vmin=-1, vmax=1)
ax_corr.set_title('ADMET Property Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.setp(ax_corr.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(ax_corr.get_yticklabels(), rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig('eda_distributions_part2.png', dpi=300, bbox_inches='tight')
print("Saved: eda_distributions_part2.png")

# SMILES length comparison
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train_df['SMILES_length'], bins=50, color='steelblue', edgecolor='black', alpha=0.7, label='Train')
axes[0].hist(test_df['SMILES_length'], bins=50, color='orange', edgecolor='black', alpha=0.5, label='Test')
axes[0].set_xlabel('SMILES String Length', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('SMILES Length Distribution', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Box plot comparison
axes[1].boxplot([train_df['SMILES_length'], test_df['SMILES_length']],
                labels=['Train', 'Test'], patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
axes[1].set_ylabel('SMILES String Length', fontsize=12)
axes[1].set_title('SMILES Length Comparison', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('smiles_length_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: smiles_length_analysis.png")

print("\n" + "="*80)
print("Data exploration complete!")
print("="*80)
print(f"\nKey Findings:")
print(f"1. Training set: {len(train_df)} compounds")
print(f"2. Test set: {len(test_df)} compounds")
print(f"3. Properties with sparse data: {len([p for p in missing_df.itertuples() if p._4 > 50])} properties with >50% missing")
print(f"4. Most complete property: {missing_df.iloc[missing_df['Present'].argmax()]['Property']} ({missing_df['Present'].max()} samples)")
print(f"5. Least complete property: {missing_df.iloc[missing_df['Present'].argmin()]['Property']} ({missing_df['Present'].min()} samples)")
