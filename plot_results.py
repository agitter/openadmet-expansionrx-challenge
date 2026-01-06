"""
This script answers three analysis questions using plots generated from
a Markdown table of benchmark results.

Questions addressed:

Q1. What is the relative performance of the four models that only have
    "Dataset and descriptions" as context?

Q2. For each of the four models, how does performance change when using
    "Dataset and descriptions" versus all other context types?

Q3. How does Kosmos model performance change across each type of context
    it uses? (Submission 11 is ignored as a duplicate.)

For each question, results are plotted with four columns corresponding to:
    - MA-RAE
    - R2
    - Spearman R
    - Kendall's Tau
"""

# Written by ChatGPT

import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------
def parse_markdown_table(md_text):
    lines = [l for l in md_text.splitlines() if "|" in l]
    header = [h.strip() for h in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        values = [v.strip() for v in line.strip("|").split("|")]
        if len(values) == len(header):
            rows.append(values)
    return pd.DataFrame(rows, columns=header)

def extract_mean_and_err(x):
    """
    Extract mean and uncertainty from strings like:
    '0.73 +/- 0.03'
    """
    if pd.isna(x):
        return np.nan, np.nan
    m = re.match(r"([-+]?[0-9]*\.?[0-9]+)\s*\+/-\s*([0-9]*\.?[0-9]+)", str(x))
    if m:
        return float(m.group(1)), float(m.group(2))
    return np.nan, np.nan

def aggregate_mean_and_err(df, metric):
    """
    Aggregate means by averaging.
    Aggregate uncertainties using RMS.
    """
    mean = df[metric].mean()
    err = np.sqrt(np.mean(df[f"{metric}_err"] ** 2))
    return mean, err

def plot_metric_columns(df, group_col, title, filename):
    metrics = ["MA-RAE", "R2", "Spearman R", "Kendall's Tau"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    grouped = df.groupby(group_col)

    for ax, metric in zip(axes, metrics):
        means = []
        errs = []
        labels = []

        for name, g in grouped:
            m, e = aggregate_mean_and_err(g, metric)
            means.append(m)
            errs.append(e)
            labels.append(name)

        ax.bar(range(len(labels)), means, yerr=errs, capsize=4)
        ax.set_title(metric)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
if len(sys.argv) != 2:
    print("Usage: python plot_results.py <markdown_file>")
    sys.exit(1)

md_file = sys.argv[1]

with open(md_file, "r", encoding="utf-8") as f:
    md_text = f.read()

df = parse_markdown_table(md_text)

# Convert numeric columns
df["Submission"] = pd.to_numeric(df["Submission"])
df["rank"] = pd.to_numeric(df["rank"])

metrics = ["MA-RAE", "R2", "Spearman R", "Kendall's Tau"]
for m in metrics:
    df[m], df[f"{m}_err"] = zip(*df[m].apply(extract_mean_and_err))

# Ignore duplicate submission 11
df = df[df["Submission"] != 11]

# ============================================================
# Q1. Relative performance with "Dataset and descriptions" only
# ============================================================
df_dataset_only = df[df["context"] == "Dataset and descriptions"]

plot_metric_columns(
    df_dataset_only,
    "Model",
    "Relative performance (Dataset and descriptions only)",
    "results/dataset_only_relative_performance.png"
)

# ============================================================
# Q2. Dataset vs other contexts for each model
# ============================================================
df["ContextType"] = np.where(
    df["context"] == "Dataset and descriptions",
    "Dataset and descriptions",
    "Other contexts"
)

plot_metric_columns(
    df,
    ["Model", "ContextType"],
    "Performance change: Dataset vs other contexts",
    "results/context_comparison_per_model.png"
)

# ============================================================
# Q3. Kosmos performance across context types
# ============================================================
df_kosmos = df[df["Model"] == "Kosmos"]

plot_metric_columns(
    df_kosmos,
    "context",
    "Kosmos performance across context types",
    "results/kosmos_context_effect.png"
)
