#!/usr/bin/env python3
"""
GNN Training with Chemprop - Multi-Task ADMET Prediction
=========================================================

Train a multi-task Graph Neural Network using Chemprop for predicting 9 ADMET properties.
Uses scaffold-based cross-validation for fair comparison with LightGBM baseline.

Configuration:
- Task type: Multi-task regression (9 properties)
- Split type: scaffold_balanced (molecules with same Murcko scaffold in same fold)
- Evaluation metric: Spearman correlation
- Cross-validation: 5-fold

Author: K-Dense Coding Agent
Date: 2025-12-05
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("=" * 80)
print("GNN Training with Chemprop - Multi-Task ADMET Prediction")
print("=" * 80)

# Define paths
BASE_DIR = Path("/app/sandbox/session_20251205_152206_4285cc85e60d")
DATA_PATH = BASE_DIR / "workflow" / "gnn_train_data_normalized.csv"
OUTPUT_DIR = BASE_DIR / "results" / "chemprop_gnn_model"
CONFIG_PATH = OUTPUT_DIR / "training_config.json"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n1. Setup")
print(f"   Data: {DATA_PATH}")
print(f"   Output directory: {OUTPUT_DIR}")

# Verify data file exists
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

# Load data to verify and get target columns
print(f"\n2. Loading data to verify...")
data = pd.read_csv(DATA_PATH)
print(f"   ✓ Loaded: {data.shape[0]:,} molecules")
print(f"   ✓ Columns: {list(data.columns)}")

# Get target columns (all except 'smiles')
target_cols = [col for col in data.columns if col != 'smiles']
print(f"   ✓ Target properties: {len(target_cols)}")
for col in target_cols:
    n_available = data[col].notna().sum()
    pct_available = 100 * n_available / len(data)
    print(f"      {col:40s}: {n_available:5d} ({pct_available:5.1f}%)")

# Training configuration
config = {
    "data_path": str(DATA_PATH),
    "dataset_type": "regression",
    "split_type": "scaffold_balanced",
    "num_folds": 5,
    "metric": "mae",  # Use MAE as primary metric, will calculate Spearman separately
    "extra_metrics": ["rmse", "r2"],
    "save_dir": str(OUTPUT_DIR),
    "save_preds": True,
    "epochs": 30,
    "batch_size": 50,
    "hidden_size": 300,
    "depth": 3,
    "dropout": 0.1,
    "ffn_num_layers": 2,
    "target_columns": target_cols,
    "smiles_column": "smiles",
    "seed": 42,
    "quiet": False
}

# Save configuration
print(f"\n3. Saving training configuration...")
with open(CONFIG_PATH, 'w') as f:
    json.dump(config, f, indent=2)
print(f"   ✓ Config saved to: {CONFIG_PATH}")

# Import chemprop (just check version, we'll use CLI)
print(f"\n4. Verifying Chemprop installation...")
try:
    import chemprop
    print(f"   ✓ Chemprop v{chemprop.__version__} available")
    print(f"   ✓ Will use Chemprop CLI for training (more stable for CV)")
except ImportError as e:
    print(f"   ✗ Failed to import chemprop: {e}")
    raise

# Using Chemprop CLI via subprocess (more stable than Python API for complex configs)
print(f"\n5. Training GNN model with Chemprop...")
print(f"   Configuration:")
print(f"      - Split type: {config['split_type']}")
print(f"      - CV folds: {config['num_folds']}")
print(f"      - Epochs: {config['epochs']}")
print(f"      - Batch size: {config['batch_size']}")
print(f"      - Hidden size: {config['hidden_size']}")
print(f"      - Depth: {config['depth']}")
print(f"      - Dropout: {config['dropout']}")
print(f"   Starting training...")
print(f"   NOTE: This may take 15-30 minutes. Progress updates will print periodically.")
print()

import subprocess
import sys

# Construct chemprop_train command
# Using CLI is more stable for cross-validation
cmd = [
    "uv", "run", "python", "-m", "chemprop", "train",
    "--data-path", str(DATA_PATH),
    "--task-type", "regression",
    "--split-type", "scaffold_balanced",
    "--num-folds", str(config['num_folds']),
    "--epochs", str(config['epochs']),
    "--batch-size", str(config['batch_size']),
    "--hidden-size", str(config['hidden_size']),
    "--depth", str(config['depth']),
    "--dropout", str(config['dropout']),
    "--ffn-num-layers", str(config['ffn_num_layers']),
    "--save-dir", str(OUTPUT_DIR),
    "--save-preds",
    "--smiles-columns", "smiles",
    "--target-columns"] + target_cols + [
    "--seed", str(config['seed'])
]

print(f"   Command: {' '.join(cmd[:10])} ...")
print()

# Run training
start_time = time.time()

try:
    # Run with real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Stream output with periodic updates
    line_count = 0
    for line in process.stdout:
        line_count += 1
        line = line.rstrip()

        # Print all lines for visibility
        # Key progress indicators: epochs, folds, loss values
        if any(keyword in line.lower() for keyword in
               ['epoch', 'fold', 'loss', 'train', 'val', 'test', 'error', 'warning', 'best']):
            print(f"   {line}")
        elif line_count % 50 == 0:  # Print every 50th line to show activity
            print(f"   [Progress] Processing... ({line_count} lines)")

    process.wait()
    return_code = process.returncode

    elapsed_time = time.time() - start_time
    elapsed_mins = elapsed_time / 60

    print()
    print(f"   Training completed in {elapsed_mins:.2f} minutes")

    if return_code != 0:
        print(f"   ⚠ WARNING: Process returned code {return_code}")
        print(f"   Check output directory for partial results: {OUTPUT_DIR}")
    else:
        print(f"   ✓ Training successful")

except Exception as e:
    print(f"   ✗ Error during training: {e}")
    raise

# Check outputs
print(f"\n6. Checking outputs...")

output_files = list(OUTPUT_DIR.glob("**/*"))
print(f"   ✓ Found {len(output_files)} files in output directory")

# Look for key output files
key_files = {
    "predictions": "test_preds.csv",
    "scores": "test_scores.json",
    "fold_0": "fold_0",
    "config": "training_config.json"
}

found_files = {}
for key, pattern in key_files.items():
    matching = [f for f in output_files if pattern in str(f)]
    if matching:
        found_files[key] = matching[0]
        print(f"   ✓ Found {key}: {matching[0].name}")
    else:
        print(f"   ⚠ Not found: {pattern}")

# List all output files for reference
print(f"\n   All output files:")
for f in sorted(output_files):
    if f.is_file():
        size_kb = f.stat().st_size / 1024
        print(f"      {f.relative_to(OUTPUT_DIR):<50s} ({size_kb:>8.1f} KB)")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✓ Data loaded: {data.shape[0]:,} molecules, {len(target_cols)} targets")
print(f"✓ Training configuration saved: {CONFIG_PATH.name}")
print(f"✓ Chemprop training completed in {elapsed_mins:.2f} minutes")
print(f"✓ Output directory: {OUTPUT_DIR}")
print(f"✓ Found {len(found_files)}/{len(key_files)} expected output types")
print("\nNext step: Analyze results and calculate performance metrics")
print("=" * 80)
