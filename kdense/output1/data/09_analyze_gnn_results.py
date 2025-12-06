#!/usr/bin/env python3
"""
GNN Results Analysis and Comparison with Baseline
==================================================

Compare the GNN model performance against the LightGBM baseline.
Generate summary statistics and performance comparison report.

Author: K-Dense Coding Agent
Date: 2025-12-05
"""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("GNN Results Analysis and Comparison")
print("=" * 80)

# Paths
BASE_DIR = Path("/app/sandbox/session_20251205_152206_4285cc85e60d")
BASELINE_PATH = BASE_DIR / "results" / "baseline_cv_scores.csv"
GNN_PATH = BASE_DIR / "results" / "gnn_cv_scores.csv"
OUTPUT_PATH = BASE_DIR / "results" / "gnn_performance_comparison.txt"

# Load results
print("\n1. Loading results...")
baseline_df = pd.read_csv(BASELINE_PATH)
gnn_df = pd.read_csv(GNN_PATH)

print(f"   ✓ Baseline results: {len(baseline_df)} properties")
print(f"   ✓ GNN results: {len(gnn_df)} properties")

# Property name mapping (baseline uses full names, GNN uses simplified)
property_mapping = {
    'LogD': 'LogD',
    'KSOL': 'KSOL',
    'HLM_CLint': 'HLM CLint',
    'MLM_CLint': 'MLM CLint',
    'Caco-2_Permeability_Papp_AtoB': 'Caco-2 Permeability Papp A>B',
    'Caco-2_Permeability_Efflux': 'Caco-2 Permeability Efflux',
    'MPPB': 'MPPB',
    'MBPB': 'MBPB',
    'MGMB': 'MGMB'
}

# Map GNN property names to baseline names
gnn_df['property_full'] = gnn_df['property'].map(property_mapping)

# Merge dataframes
print("\n2. Comparing performance...")
comparison = pd.merge(
    baseline_df[['property', 'n_samples', 'spearman_mean', 'spearman_std', 'ma_rae_mean', 'ma_rae_std']],
    gnn_df[['property_full', 'spearman_mean', 'spearman_std', 'ma_rae_mean', 'ma_rae_std']],
    left_on='property',
    right_on='property_full',
    suffixes=('_baseline', '_gnn')
)

# Calculate differences
comparison['spearman_diff'] = comparison['spearman_mean_gnn'] - comparison['spearman_mean_baseline']
comparison['spearman_relative_improvement'] = 100 * comparison['spearman_diff'] / comparison['spearman_mean_baseline']
comparison['ma_rae_diff'] = comparison['ma_rae_mean_gnn'] - comparison['ma_rae_mean_baseline']
comparison['ma_rae_relative_improvement'] = 100 * comparison['ma_rae_diff'] / comparison['ma_rae_mean_baseline']

# Overall statistics
baseline_mean_spearman = baseline_df['spearman_mean'].mean()
gnn_mean_spearman = gnn_df['spearman_mean'].mean()
overall_spearman_diff = gnn_mean_spearman - baseline_mean_spearman
overall_spearman_relative = 100 * overall_spearman_diff / baseline_mean_spearman

baseline_mean_ma_rae = baseline_df['ma_rae_mean'].mean()
gnn_mean_ma_rae = gnn_df['ma_rae_mean'].mean()
overall_ma_rae_diff = gnn_mean_ma_rae - baseline_mean_ma_rae
overall_ma_rae_relative = 100 * overall_ma_rae_diff / baseline_mean_ma_rae

# Generate report
print("\n3. Generating comparison report...")

report = []
report.append("=" * 80)
report.append("GNN vs Baseline Performance Comparison")
report.append("=" * 80)
report.append("")
report.append("OVERALL PERFORMANCE")
report.append("-" * 80)
report.append(f"Metric              Baseline LightGBM    GNN Multi-Task       Difference")
report.append("-" * 80)
report.append(f"Mean Spearman:      {baseline_mean_spearman:8.4f}             {gnn_mean_spearman:8.4f}             {overall_spearman_diff:+.4f} ({overall_spearman_relative:+.2f}%)")
report.append(f"Mean MA-RAE:        {baseline_mean_ma_rae:8.4f}             {gnn_mean_ma_rae:8.4f}             {overall_ma_rae_diff:+.4f} ({overall_ma_rae_relative:+.2f}%)")
report.append("")

if overall_spearman_diff > 0:
    report.append(f"✓ GNN model shows {abs(overall_spearman_diff):.4f} Spearman improvement over baseline")
else:
    report.append(f"⚠ GNN model shows {abs(overall_spearman_diff):.4f} Spearman decrease vs baseline")

report.append("")
report.append("SPEARMAN CORRELATION - PROPERTY-LEVEL COMPARISON")
report.append("-" * 80)
report.append(f"{'Property':<40s} {'Baseline':<15s} {'GNN':<15s} {'Diff':<12s} {'Rel %':<10s}")
report.append("-" * 80)

for _, row in comparison.iterrows():
    prop = row['property']
    baseline_val = row['spearman_mean_baseline']
    baseline_std = row['spearman_std_baseline']
    gnn_val = row['spearman_mean_gnn']
    gnn_std = row['spearman_std_gnn']
    diff = row['spearman_diff']
    rel_pct = row['spearman_relative_improvement']

    baseline_str = f"{baseline_val:.4f}±{baseline_std:.3f}"
    gnn_str = f"{gnn_val:.4f}±{gnn_std:.3f}"
    diff_str = f"{diff:+.4f}"
    rel_str = f"{rel_pct:+.2f}%"

    report.append(f"{prop:<40s} {baseline_str:<15s} {gnn_str:<15s} {diff_str:<12s} {rel_str:<10s}")

report.append("-" * 80)
report.append("")

report.append("MA-RAE - PROPERTY-LEVEL COMPARISON")
report.append("-" * 80)
report.append(f"{'Property':<40s} {'Baseline':<15s} {'GNN':<15s} {'Diff':<12s} {'Rel %':<10s}")
report.append("-" * 80)

for _, row in comparison.iterrows():
    prop = row['property']
    baseline_val = row['ma_rae_mean_baseline']
    baseline_std = row['ma_rae_std_baseline']
    gnn_val = row['ma_rae_mean_gnn']
    gnn_std = row['ma_rae_std_gnn']
    diff = row['ma_rae_diff']
    rel_pct = row['ma_rae_relative_improvement']

    baseline_str = f"{baseline_val:.4f}±{baseline_std:.3f}"
    gnn_str = f"{gnn_val:.4f}±{gnn_std:.3f}"
    diff_str = f"{diff:+.4f}"
    rel_str = f"{rel_pct:+.2f}%"

    report.append(f"{prop:<40s} {baseline_str:<15s} {gnn_str:<15s} {diff_str:<12s} {rel_str:<10s}")

report.append("-" * 80)
report.append("")

# Identify properties with improvement
improved_spearman = comparison[comparison['spearman_diff'] > 0]['property'].tolist()
declined_spearman = comparison[comparison['spearman_diff'] < 0]['property'].tolist()
improved_ma_rae = comparison[comparison['ma_rae_diff'] < 0]['property'].tolist()  # Lower MA-RAE is better
declined_ma_rae = comparison[comparison['ma_rae_diff'] > 0]['property'].tolist()

report.append("ANALYSIS (Spearman Correlation)")
report.append("-" * 80)
report.append(f"Properties improved ({len(improved_spearman)}):")
for prop in improved_spearman:
    row = comparison[comparison['property'] == prop].iloc[0]
    report.append(f"   • {prop:<40s}: {row['spearman_diff']:+.4f} ({row['spearman_relative_improvement']:+.2f}%)")

report.append("")
report.append(f"Properties declined ({len(declined_spearman)}):")
for prop in declined_spearman:
    row = comparison[comparison['property'] == prop].iloc[0]
    report.append(f"   • {prop:<40s}: {row['spearman_diff']:+.4f} ({row['spearman_relative_improvement']:+.2f}%)")

report.append("")
report.append("KEY FINDINGS")
report.append("-" * 80)

# Find best improvements
top_improved_spearman = comparison.nlargest(3, 'spearman_diff')['property'].tolist()
report.append(f"Top 3 Spearman improvements: {', '.join(top_improved_spearman)}")

# Note: For MA-RAE, lower is better, so we look for negative diffs
top_improved_ma_rae = comparison.nsmallest(3, 'ma_rae_diff')['property'].tolist()
report.append(f"Top 3 MA-RAE improvements: {', '.join(top_improved_ma_rae)}")

# Find worst declines
if len(declined_spearman) > 0:
    worst_declined = comparison.nsmallest(3, 'spearman_diff')['property'].tolist()
    report.append(f"Largest Spearman declines: {', '.join(worst_declined)}")

# Tier analysis
report.append("")
report.append("TIER ANALYSIS (by data availability)")
report.append("-" * 80)

tiers = {
    'Tier 1 (>95% complete)': ['LogD', 'KSOL'],
    'Tier 2 (70-85% complete)': ['HLM CLint', 'MLM CLint'],
    'Tier 3 (40-60% complete)': ['Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux'],
    'Tier 4 (<25% complete)': ['MPPB', 'MBPB', 'MGMB']
}

for tier_name, tier_props in tiers.items():
    tier_baseline = baseline_df[baseline_df['property'].isin(tier_props)]['spearman_mean'].mean()
    tier_gnn_props = [property_mapping.get(p, p) for p in tier_props]
    tier_gnn = gnn_df[gnn_df['property'].isin(tier_gnn_props)]['spearman_mean'].mean()
    tier_diff = tier_gnn - tier_baseline
    tier_rel = 100 * tier_diff / tier_baseline

    report.append(f"{tier_name}:")
    report.append(f"   Baseline: {tier_baseline:.4f}, GNN: {tier_gnn:.4f}, "
                  f"Diff: {tier_diff:+.4f} ({tier_rel:+.2f}%)")

report.append("")
report.append("INTERPRETATION")
report.append("-" * 80)

if overall_spearman_diff > 0.01:
    report.append("✓ GNN model shows meaningful Spearman improvement over baseline (+1.0% or more)")
    report.append("  The multi-task learning and Z-score normalization strategy was effective.")
elif overall_spearman_diff > 0:
    report.append("✓ GNN model shows modest Spearman improvement over baseline")
    report.append("  Results are comparable, suggesting both approaches are viable.")
else:
    report.append("⚠ GNN model did not outperform baseline on Spearman")
    report.append("  This is acceptable - baseline LightGBM is very strong for this task.")
    report.append("  GNN could benefit from: more epochs, hyperparameter tuning, or true graph")
    report.append("  neural network architecture (vs Morgan fingerprints used here).")

report.append("")
if overall_ma_rae_diff < 0:
    report.append(f"✓ GNN model shows MA-RAE improvement (lower is better): {abs(overall_ma_rae_diff):.4f} reduction")
else:
    report.append(f"⚠ GNN model shows MA-RAE increase: {abs(overall_ma_rae_diff):.4f}")

report.append("")
report.append("TECHNICAL NOTES")
report.append("-" * 80)
report.append("• Baseline: LightGBM with Morgan FP (2048-bit) + RDKit descriptors (217)")
report.append("• GNN: Multi-task neural network with Morgan FP (2048-bit) as input")
report.append("• Both use scaffold-based 5-fold cross-validation")
report.append("• GNN uses Z-score normalized targets (critical for multi-task learning)")
report.append("• Metrics: Spearman correlation (higher is better), MA-RAE (lower is better)")
report.append("• MA-RAE = MAE / MAE_baseline (baseline = mean prediction)")
report.append("• Note: This GNN uses Morgan fingerprints, not true graph convolutions")
report.append("  True GNN (with message passing) could show larger improvements")

report.append("")
report.append("=" * 80)

report_text = "\n".join(report)

# Print report
print("\n" + report_text)

# Save report
with open(OUTPUT_PATH, 'w') as f:
    f.write(report_text)

print(f"\n4. Report saved to: {OUTPUT_PATH}")
print("=" * 80)
