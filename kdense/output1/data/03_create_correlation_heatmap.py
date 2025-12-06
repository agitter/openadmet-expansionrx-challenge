#!/usr/bin/env python3
"""
Step 3: Create Correlation Heatmap Visualization
Generates publication-quality heatmap of target property correlations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set reproducibility
np.random.seed(42)

# Configure matplotlib for publication-quality figures
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['figure.dpi'] = 300

# Define paths
SESSION_DIR = Path("/app/sandbox/session_20251205_152206_4285cc85e60d")
RESULTS_DIR = SESSION_DIR / "results"

print("\n" + "="*70)
print("STEP 3: CORRELATION HEATMAP VISUALIZATION")
print("="*70)

# Load correlation matrix
print("\n[Loading correlation matrix...]")
corr_matrix = pd.read_csv(RESULTS_DIR / "correlation_matrix.csv", index_col=0)
print(f"✓ Correlation matrix loaded: {corr_matrix.shape}")

# Create figure
print("\n[Creating heatmap visualization...]")
fig, ax = plt.subplots(figsize=(12, 10))

# Create heatmap with seaborn
sns.heatmap(
    corr_matrix,
    annot=True,  # Show correlation coefficients
    fmt='.2f',   # Format to 2 decimal places
    cmap='RdBu_r',  # Red-Blue diverging colormap (reversed)
    center=0,    # Center colormap at 0
    vmin=-1,     # Min correlation
    vmax=1,      # Max correlation
    square=True, # Square cells
    linewidths=0.5,  # Grid lines
    cbar_kws={
        'label': 'Pearson Correlation Coefficient',
        'shrink': 0.8
    },
    ax=ax
)

# Customize plot
ax.set_title(
    'Correlation Matrix of Target Properties\n' +
    'Molecular Property Prediction Dataset',
    fontsize=14,
    fontweight='bold',
    pad=20
)

# Rotate labels for better readability
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)

# Tight layout to prevent label cutoff
plt.tight_layout()

# Save figure
output_file = RESULTS_DIR / "property_correlation_heatmap.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Heatmap saved to: {output_file}")

# Also save as PDF for vector graphics
output_pdf = RESULTS_DIR / "property_correlation_heatmap.pdf"
plt.savefig(output_pdf, bbox_inches='tight')
print(f"✓ Vector version saved to: {output_pdf}")

plt.close()

# Generate summary statistics about correlations
print("\n" + "="*70)
print("CORRELATION SUMMARY")
print("="*70)

# Extract upper triangle (avoid duplicates)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
upper_tri = corr_matrix.where(mask)
correlations = upper_tri.stack()

print(f"\nTotal unique correlations: {len(correlations)}")
print(f"Mean absolute correlation: {correlations.abs().mean():.3f}")
print(f"Median correlation: {correlations.median():.3f}")
print(f"Max positive correlation: {correlations.max():.3f}")
print(f"Max negative correlation: {correlations.min():.3f}")

# Count by strength
very_strong = (correlations.abs() > 0.7).sum()
strong = ((correlations.abs() > 0.5) & (correlations.abs() <= 0.7)).sum()
moderate = ((correlations.abs() > 0.3) & (correlations.abs() <= 0.5)).sum()
weak = (correlations.abs() <= 0.3).sum()

print(f"\nCorrelation strength distribution:")
print(f"  Very strong (|r| > 0.7): {very_strong} ({very_strong/len(correlations)*100:.1f}%)")
print(f"  Strong (0.5 < |r| ≤ 0.7): {strong} ({strong/len(correlations)*100:.1f}%)")
print(f"  Moderate (0.3 < |r| ≤ 0.5): {moderate} ({moderate/len(correlations)*100:.1f}%)")
print(f"  Weak (|r| ≤ 0.3): {weak} ({weak/len(correlations)*100:.1f}%)")

# Top positive and negative correlations
print(f"\nTop 3 positive correlations:")
top_pos = correlations.nlargest(3)
for (var1, var2), corr in top_pos.items():
    print(f"  {var1} ↔ {var2}: r = {corr:.3f}")

print(f"\nTop 3 negative correlations:")
top_neg = correlations.nsmallest(3)
for (var1, var2), corr in top_neg.items():
    print(f"  {var1} ↔ {var2}: r = {corr:.3f}")

print("\n✓ Correlation heatmap visualization completed successfully!")
print(f"✓ Output file: {output_file}")
