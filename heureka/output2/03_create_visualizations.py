"""
Create Comprehensive Visualizations for Model Performance
==========================================================
Generates publication-quality figures showing model performance,
feature importance, and prediction distributions.

Date: 2024-12-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("Creating Performance Visualizations")
print("="*80)

# Load data
train_df = pd.read_csv('expansion_data_train.csv')
pred_df = pd.read_csv('admet_predictions_test.csv')

# Load models
with open('trained_models.pkl', 'rb') as f:
    models = pickle.load(f)

ADMET_PROPERTIES = [
    'LogD', 'KSOL', 'HLM CLint', 'MLM CLint',
    'Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux',
    'MPPB', 'MBPB', 'MGMB'
]

# ============================================================================
# Figure 1: Model Performance Summary
# ============================================================================

print("\n[1] Creating model performance summary...")

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

for idx, prop in enumerate(ADMET_PROPERTIES):
    ax = axes[idx]

    # Training data distribution
    train_data = train_df[prop].dropna()
    pred_data = pred_df[prop]

    # Create overlapping histograms
    ax.hist(train_data, bins=40, alpha=0.6, color='steelblue',
            edgecolor='black', label=f'Training (n={len(train_data)})')
    ax.hist(pred_data, bins=40, alpha=0.5, color='coral',
            edgecolor='black', label=f'Predictions (n={len(pred_data)})')

    # Add median lines
    ax.axvline(train_data.median(), color='darkblue', linestyle='--',
               linewidth=2, label=f'Train Median: {train_data.median():.2f}')
    ax.axvline(pred_data.median(), color='darkred', linestyle='--',
               linewidth=2, label=f'Pred Median: {pred_data.median():.2f}')

    ax.set_xlabel('Value', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title(f'{prop}', fontsize=11, fontweight='bold', pad=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

plt.suptitle('ADMET Property Distributions: Training vs Predictions',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('prediction_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: prediction_distributions.png")

# ============================================================================
# Figure 2: Cross-Validation Performance
# ============================================================================

print("\n[2] Creating CV performance visualization...")

# Read performance metrics from file
with open('model_performance_report.txt', 'r') as f:
    report_text = f.read()

# Extract CV R² values
cv_r2_values = []
train_r2_values = []
n_samples = []

for prop in ADMET_PROPERTIES:
    # Find the section for this property
    start_idx = report_text.find(f"{prop}:")
    if start_idx != -1:
        section = report_text[start_idx:start_idx+500]

        # Extract CV R²
        if "Cross-validation R²:" in section:
            cv_line = [l for l in section.split('\n') if 'Cross-validation R²:' in l][0]
            cv_val = float(cv_line.split(':')[1].split('±')[0].strip())
            cv_r2_values.append(cv_val)

            # Extract training R²
            train_line = [l for l in section.split('\n') if 'Training R²:' in l][0]
            train_val = float(train_line.split(':')[1].strip())
            train_r2_values.append(train_val)

            # Extract sample count
            sample_line = [l for l in section.split('\n') if 'Training samples:' in l][0]
            n_val = int(sample_line.split(':')[1].strip())
            n_samples.append(n_val)
        else:
            cv_r2_values.append(np.nan)
            train_r2_values.append(np.nan)
            n_samples.append(0)

# Create performance dataframe
perf_df = pd.DataFrame({
    'Property': ADMET_PROPERTIES,
    'CV R²': cv_r2_values,
    'Training R²': train_r2_values,
    'Samples': n_samples
})

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: R² comparison
ax1 = axes[0]
x = np.arange(len(ADMET_PROPERTIES))
width = 0.35

bars1 = ax1.bar(x - width/2, perf_df['CV R²'], width,
                label='Cross-Validation R²', color='lightcoral',
                edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, perf_df['Training R²'], width,
                label='Training R²', color='lightblue',
                edgecolor='black', linewidth=1.5)

ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax1.axhline(y=0.7, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xlabel('ADMET Property', fontsize=12, fontweight='bold')
ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance: CV vs Training', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([p.replace(' ', '\n') for p in ADMET_PROPERTIES],
                     rotation=45, ha='right', fontsize=9)
ax1.legend(fontsize=10, loc='lower right')
ax1.set_ylim([0, 1.0])
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=7)

# Plot 2: Sample size vs performance
ax2 = axes[1]
valid_mask = ~np.isnan(perf_df['CV R²'])
scatter = ax2.scatter(perf_df.loc[valid_mask, 'Samples'],
                     perf_df.loc[valid_mask, 'CV R²'],
                     s=200, c=perf_df.loc[valid_mask, 'CV R²'],
                     cmap='RdYlGn', edgecolor='black', linewidth=2,
                     vmin=0, vmax=1)

# Add property labels
for idx, row in perf_df[valid_mask].iterrows():
    ax2.annotate(row['Property'], (row['Samples'], row['CV R²']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.7)

ax2.set_xlabel('Number of Training Samples', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cross-Validation R²', fontsize=12, fontweight='bold')
ax2.set_title('Performance vs Data Availability', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.0])

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('CV R² Score', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('model_performance_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_performance_summary.png")

# ============================================================================
# Figure 3: Prediction Statistics
# ============================================================================

print("\n[3] Creating prediction statistics...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Prediction ranges
ax1 = axes[0, 0]
pred_stats = []
for prop in ADMET_PROPERTIES:
    train_vals = train_df[prop].dropna()
    pred_vals = pred_df[prop]
    pred_stats.append({
        'Property': prop,
        'Train_Min': train_vals.min(),
        'Train_Max': train_vals.max(),
        'Train_Mean': train_vals.mean(),
        'Pred_Min': pred_vals.min(),
        'Pred_Max': pred_vals.max(),
        'Pred_Mean': pred_vals.mean()
    })

stats_df = pd.DataFrame(pred_stats)

x_pos = np.arange(len(ADMET_PROPERTIES))
train_ranges = stats_df['Train_Max'] - stats_df['Train_Min']
pred_ranges = stats_df['Pred_Max'] - stats_df['Pred_Min']

ax1.barh(x_pos - 0.2, train_ranges, 0.4, label='Training Range',
         color='steelblue', edgecolor='black')
ax1.barh(x_pos + 0.2, pred_ranges, 0.4, label='Prediction Range',
         color='coral', edgecolor='black')
ax1.set_yticks(x_pos)
ax1.set_yticklabels(ADMET_PROPERTIES, fontsize=9)
ax1.set_xlabel('Value Range', fontsize=11, fontweight='bold')
ax1.set_title('Data Range Comparison', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='x')

# Plot 2: Mean comparison
ax2 = axes[0, 1]
ax2.scatter(stats_df['Train_Mean'], stats_df['Pred_Mean'],
           s=200, c='purple', alpha=0.6, edgecolor='black', linewidth=2)

# Add diagonal line (perfect agreement)
min_val = min(stats_df['Train_Mean'].min(), stats_df['Pred_Mean'].min())
max_val = max(stats_df['Train_Mean'].max(), stats_df['Pred_Mean'].max())
ax2.plot([min_val, max_val], [min_val, max_val],
         'k--', linewidth=2, alpha=0.5, label='Perfect Agreement')

for idx, row in stats_df.iterrows():
    ax2.annotate(row['Property'], (row['Train_Mean'], row['Pred_Mean']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=7, alpha=0.7)

ax2.set_xlabel('Training Mean', fontsize=11, fontweight='bold')
ax2.set_ylabel('Prediction Mean', fontsize=11, fontweight='bold')
ax2.set_title('Mean Value Comparison', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Coefficient of variation
ax3 = axes[1, 0]
train_cv = [train_df[p].dropna().std() / train_df[p].dropna().mean()
            for p in ADMET_PROPERTIES]
pred_cv = [pred_df[p].std() / pred_df[p].mean() for p in ADMET_PROPERTIES]

x_pos = np.arange(len(ADMET_PROPERTIES))
ax3.bar(x_pos - 0.2, train_cv, 0.4, label='Training CV',
        color='lightgreen', edgecolor='black')
ax3.bar(x_pos + 0.2, pred_cv, 0.4, label='Prediction CV',
        color='lightyellow', edgecolor='black')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([p.replace(' ', '\n') for p in ADMET_PROPERTIES],
                     rotation=45, ha='right', fontsize=8)
ax3.set_ylabel('Coefficient of Variation', fontsize=11, fontweight='bold')
ax3.set_title('Variability Comparison', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Summary statistics table
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')

summary_data = []
summary_data.append(['Metric', 'Value'])
summary_data.append(['Total Test Compounds', f'{len(pred_df)}'])
summary_data.append(['Properties Predicted', f'{len(ADMET_PROPERTIES)}'])
summary_data.append(['Mean CV R²', f'{perf_df["CV R²"].mean():.3f}'])
summary_data.append(['Median CV R²', f'{perf_df["CV R²"].median():.3f}'])
summary_data.append(['Best CV R² (Property)', f'{perf_df.loc[perf_df["CV R²"].idxmax(), "Property"]}: {perf_df["CV R²"].max():.3f}'])
summary_data.append(['Total Training Samples', f'{len(train_df)}'])
summary_data.append(['Feature Type', 'Morgan FP (2048 bits)'])
summary_data.append(['Model Type', 'Random Forest'])

table = ax4.table(cellText=summary_data, cellLoc='left',
                 colWidths=[0.6, 0.4], loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(2):
    cell = table[(0, i)]
    cell.set_facecolor('#4CAF50')
    cell.set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(summary_data)):
    for j in range(2):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#f0f0f0')

ax4.set_title('Challenge Summary Statistics', fontsize=12,
             fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('prediction_statistics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: prediction_statistics.png")

print("\n" + "="*80)
print("Visualization Complete!")
print("="*80)
print("\nGenerated figures:")
print("  1. prediction_distributions.png - Training vs prediction distributions")
print("  2. model_performance_summary.png - CV performance and data availability")
print("  3. prediction_statistics.png - Comprehensive prediction statistics")
