#!/usr/bin/env python3
"""
Generate figures for the ADMET Technical Report
Author: K-Dense Web
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150

output_dir = '/app/sandbox/session_20251217_085238_bf1de403d101/writing_outputs/figures'

# Data from model_performance.csv
properties = ['LogD', 'KSol', 'MLM', 'HLM', 'Peff', 'Papp', 'MPPB', 'MBPB', 'MGMB']
property_full_names = [
    'LogD\n(Lipophilicity)',
    'KSol\n(Solubility)',
    'MLM CLint\n(Mouse Liver)',
    'HLM CLint\n(Human Liver)',
    'Peff\n(Efflux Ratio)',
    'Papp A>B\n(Permeability)',
    'MPPB\n(Plasma Binding)',
    'MBPB\n(Brain Binding)',
    'MGMB\n(Muscle Binding)'
]

ma_rae_mean = [0.377, 6.749, 1.342, 1.379, 0.588, 1.334, 1.498, 1.921, 1.616]
ma_rae_std = [0.045, 3.852, 0.267, 0.206, 0.035, 0.228, 0.225, 0.590, 1.673]
training_samples = [5039, 5128, 4522, 3759, 2161, 2157, 1302, 975, 222]

# Missing data percentages
missing_pct = [5.4, 3.7, 15.1, 29.4, 59.4, 59.5, 75.6, 81.7, 95.8]

#===============================================================================
# FIGURE 1: Model Performance Comparison
#===============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(properties))
width = 0.6

# Color by performance
colors = []
for val in ma_rae_mean:
    if val < 1.0:
        colors.append('#2ecc71')  # Green - good
    elif val < 2.0:
        colors.append('#f39c12')  # Orange - moderate
    else:
        colors.append('#e74c3c')  # Red - challenging

bars = ax.bar(x, ma_rae_mean, width, yerr=ma_rae_std, capsize=4, color=colors,
              edgecolor='black', linewidth=0.5, alpha=0.85)

ax.set_ylabel('MA-RAE (Macro-Averaged Relative Absolute Error)', fontsize=12)
ax.set_xlabel('ADMET Property', fontsize=12)
ax.set_title('Cross-Validation Model Performance by ADMET Property', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(property_full_names, fontsize=9, rotation=0)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='MA-RAE = 1.0')

# Add legend
green_patch = mpatches.Patch(color='#2ecc71', label='Excellent (MA-RAE < 1.0)')
orange_patch = mpatches.Patch(color='#f39c12', label='Good (1.0 ≤ MA-RAE < 2.0)')
red_patch = mpatches.Patch(color='#e74c3c', label='Challenging (MA-RAE ≥ 2.0)')
ax.legend(handles=[green_patch, orange_patch, red_patch], loc='upper right', fontsize=10)

# Add value labels on bars
for bar, val, std in zip(bars, ma_rae_mean, ma_rae_std):
    height = bar.get_height()
    ax.annotate(f'{val:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.1),
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylim(0, max(ma_rae_mean) + max(ma_rae_std) + 1.5)
plt.tight_layout()
plt.savefig(f'{output_dir}/figure1_model_performance.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure1_model_performance.pdf', bbox_inches='tight')
plt.close()
print("Figure 1: Model Performance saved")

#===============================================================================
# FIGURE 2: Data Availability vs Model Error
#===============================================================================
fig, ax1 = plt.subplots(figsize=(12, 6))

x = np.arange(len(properties))
width = 0.35

# Bar for training samples
ax1.bar(x - width/2, training_samples, width, color='#3498db', alpha=0.8,
        edgecolor='black', linewidth=0.5, label='Training Samples')
ax1.set_xlabel('ADMET Property', fontsize=12)
ax1.set_ylabel('Number of Training Samples', color='#3498db', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#3498db')
ax1.set_xticks(x)
ax1.set_xticklabels(property_full_names, fontsize=9, rotation=0)

# Create second y-axis for MA-RAE
ax2 = ax1.twinx()
ax2.bar(x + width/2, ma_rae_mean, width, color='#e74c3c', alpha=0.8,
        edgecolor='black', linewidth=0.5, label='MA-RAE')
ax2.set_ylabel('MA-RAE Score', color='#e74c3c', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#e74c3c')

ax1.set_title('Training Data Availability vs. Model Performance', fontsize=14, fontweight='bold')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/figure2_data_vs_performance.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure2_data_vs_performance.pdf', bbox_inches='tight')
plt.close()
print("Figure 2: Data vs Performance saved")

#===============================================================================
# FIGURE 3: Feature Importance Analysis - Top Features by Property
#===============================================================================
# Feature importance data (top 5 for each property for clarity)
feature_data = {
    'LogD': {'features': ['fp_1683', 'fp_197', 'fp_561', 'MolLogP', 'fp_858'],
             'importance': [161.42, 127.80, 84.85, 57.57, 47.12]},
    'KSol': {'features': ['fp_1683', 'fp_1086', 'fp_736', 'fp_1422', 'fp_407'],
             'importance': [405.37, 177.92, 156.55, 139.92, 132.62]},
    'MLM': {'features': ['fp_1402', 'fp_488', 'fp_1231', 'fp_1101', 'fp_634'],
            'importance': [1619.46, 369.88, 235.80, 124.66, 118.70]},
    'HLM': {'features': ['fp_1765', 'fp_1998', 'fp_754', 'fp_1482', 'fp_1757'],
            'importance': [97.57, 93.16, 86.22, 80.74, 74.72]},
    'MPPB': {'features': ['fp_378', 'fp_519', 'fp_1825', 'fp_101', 'fp_1101'],
             'importance': [9838.66, 5651.38, 4622.48, 3877.40, 3759.88]},
}

fig, axes = plt.subplots(2, 3, figsize=(14, 10))
axes = axes.flatten()

colors_map = {'LogD': '#2ecc71', 'KSol': '#e74c3c', 'MLM': '#3498db',
              'HLM': '#9b59b6', 'MPPB': '#f39c12'}

for idx, (prop, data) in enumerate(feature_data.items()):
    ax = axes[idx]
    y_pos = np.arange(len(data['features']))

    # Normalize importance for visualization
    max_imp = max(data['importance'])
    norm_imp = [x/max_imp * 100 for x in data['importance']]

    bars = ax.barh(y_pos, norm_imp, color=colors_map[prop], alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data['features'], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Relative Importance (%)', fontsize=10)
    ax.set_title(f'{prop}', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 110)

    # Add value labels
    for bar, val in zip(bars, data['importance']):
        width_bar = bar.get_width()
        ax.annotate(f'{val:.0f}',
                    xy=(width_bar + 2, bar.get_y() + bar.get_height()/2),
                    ha='left', va='center', fontsize=8)

# Remove empty subplot
axes[5].axis('off')

fig.suptitle('Top 5 Feature Importance by ADMET Property\n(XGBoost Gain Metric)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/figure3_feature_importance.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure3_feature_importance.pdf', bbox_inches='tight')
plt.close()
print("Figure 3: Feature Importance saved")

#===============================================================================
# FIGURE 4: Feature Type Distribution Pie Chart
#===============================================================================
fig, ax = plt.subplots(figsize=(8, 8))

# Aggregate data: 89 fingerprints, 1 descriptor (MolLogP for LogD) in top-10 across all
fingerprints = 89
descriptors = 1

sizes = [fingerprints, descriptors]
labels = [f'Morgan Fingerprints\n({fingerprints}/90 = 98.9%)',
          f'RDKit Descriptors\n({descriptors}/90 = 1.1%)']
colors_pie = ['#3498db', '#e74c3c']
explode = (0.02, 0.1)

wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                   autopct='%1.1f%%', shadow=False, startangle=90,
                                   textprops={'fontsize': 12})
ax.set_title('Distribution of Feature Types in Top-10 Features\nAcross All 9 ADMET Models',
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/figure4_feature_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure4_feature_distribution.pdf', bbox_inches='tight')
plt.close()
print("Figure 4: Feature Distribution saved")

#===============================================================================
# FIGURE 5: Prediction Distribution Violin Plots
#===============================================================================
# Prediction statistics from the report
pred_stats = {
    'LogD': {'min': -1.16, 'q1': 1.64, 'median': 2.11, 'mean': 2.10, 'q3': 2.66, 'max': 4.30},
    'KSol': {'min': 0.00, 'q1': 55.51, 'median': 91.67, 'mean': 100.63, 'q3': 144.30, 'max': 338.39},
    'MLM': {'min': 0.77, 'q1': 71.34, 'median': 167.79, 'mean': 197.34, 'q3': 282.99, 'max': 1048.48},
    'HLM': {'min': 0.87, 'q1': 7.17, 'median': 14.64, 'mean': 20.25, 'q3': 26.97, 'max': 126.16},
    'MPPB': {'min': 0.00, 'q1': 8.79, 'median': 13.74, 'mean': 15.99, 'q3': 21.11, 'max': 70.27},
    'MBPB': {'min': 0.00, 'q1': 2.46, 'median': 5.36, 'mean': 6.60, 'q3': 8.79, 'max': 53.33},
    'MGMB': {'min': 0.00, 'q1': 3.87, 'median': 6.66, 'mean': 9.24, 'q3': 10.37, 'max': 55.55},
}

# Create box plot-like visualization from statistics
fig, axes = plt.subplots(2, 4, figsize=(14, 8))
axes = axes.flatten()

prop_colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c', '#e67e22']

for idx, (prop, stats) in enumerate(pred_stats.items()):
    ax = axes[idx]

    # Create synthetic box plot data
    box_data = [stats['min'], stats['q1'], stats['median'], stats['q3'], stats['max']]

    bp = ax.boxplot([box_data], positions=[1], widths=0.5, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor(prop_colors[idx])
    bp['boxes'][0].set_alpha(0.7)

    ax.axhline(y=stats['mean'], color='red', linestyle='--', linewidth=1.5, label=f"Mean={stats['mean']:.1f}")
    ax.set_title(prop, fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Value', fontsize=10)
    ax.set_xticks([])
    ax.legend(loc='upper right', fontsize=8)

# Remove empty subplot
axes[7].axis('off')

fig.suptitle('Test Set Prediction Distributions by ADMET Property\n(2,282 molecules)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/figure5_prediction_distributions.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure5_prediction_distributions.pdf', bbox_inches='tight')
plt.close()
print("Figure 5: Prediction Distributions saved")

#===============================================================================
# FIGURE 6: Missing Data Heatmap
#===============================================================================
fig, ax = plt.subplots(figsize=(10, 4))

# Create matrix for heatmap (1 row with missing percentages)
missing_matrix = np.array([missing_pct]).reshape(1, -1)

# Custom colormap: green (low missing) to red (high missing)
cmap = plt.cm.RdYlGn_r

im = ax.imshow(missing_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)

# Add text annotations
for j, val in enumerate(missing_pct):
    color = 'white' if val > 50 else 'black'
    ax.text(j, 0, f'{val:.1f}%', ha='center', va='center',
            color=color, fontsize=11, fontweight='bold')

ax.set_xticks(np.arange(len(properties)))
ax.set_xticklabels(property_full_names, fontsize=9, rotation=0)
ax.set_yticks([])
ax.set_title('Training Data Missing Value Percentages by ADMET Property',
             fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
cbar.set_label('% Missing Data', fontsize=11)

plt.tight_layout()
plt.savefig(f'{output_dir}/figure6_missing_data.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure6_missing_data.pdf', bbox_inches='tight')
plt.close()
print("Figure 6: Missing Data Heatmap saved")

print("\n=== All figures generated successfully ===")
print(f"Output directory: {output_dir}")
