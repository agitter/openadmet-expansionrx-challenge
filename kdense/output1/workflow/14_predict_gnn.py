#!/usr/bin/env python3
"""
Generate GNN Model Predictions for Test Set

This script loads the trained multi-task neural network and generates predictions
for all test molecules. Applies inverse Z-score transform to get original scale.

Author: K-Dense Coding Agent
Date: 2025-12-05
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

# Suppress warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# Set seed
np.random.seed(42)
torch.manual_seed(42)

# Paths
BASE_DIR = Path("/app/sandbox/session_20251205_152206_4285cc85e60d")

print("=" * 80)
print("Generate GNN Predictions for Test Set")
print("=" * 80)

# Load test data
print("\n1. Loading test data...")
test_df = pd.read_csv(BASE_DIR / "user_data" / "expansion_data_test_blinded.csv")
print(f"   ✓ Test molecules: {len(test_df):,}")

test_smiles = test_df['SMILES'].values
test_mol_names = test_df['Molecule Name'].values

# Generate molecular features (same as training)
print("\n2. Generating molecular features...")
print("   Using Morgan fingerprints (2048-bit, radius=2)...")

def smiles_to_fingerprint(smi):
    """Convert SMILES to Morgan fingerprint."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return np.array(fp)
    except:
        return None

features = []
failed_indices = []

for i, smi in enumerate(test_smiles):
    fp = smiles_to_fingerprint(smi)
    if fp is not None:
        features.append(fp)
    else:
        # Use zero vector for failed molecules
        features.append(np.zeros(2048))
        failed_indices.append(i)

    if (i + 1) % 250 == 0:
        print(f"   Progress: {i+1}/{len(test_smiles)} molecules...")

features = np.array(features, dtype=np.float32)
print(f"   ✓ Features shape: {features.shape}")
if failed_indices:
    print(f"   ⚠ Failed featurizations: {len(failed_indices)}")

# Load GNN model
print("\n3. Loading trained GNN model...")

# Define model architecture (must match training)
class MultiTaskNN(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=300, n_tasks=9):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_tasks)])

    def forward(self, x):
        h = self.encoder(x)
        outputs = [head(h) for head in self.heads]
        return torch.cat(outputs, dim=1)

# Load model
model = MultiTaskNN(input_dim=2048, hidden_dim=300, n_tasks=9)
checkpoint = torch.load(BASE_DIR / "results" / "gnn_model_full.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"   ✓ Model loaded successfully")

# Generate predictions (normalized scale)
print("\n4. Generating predictions...")
with torch.no_grad():
    X_tensor = torch.FloatTensor(features)
    predictions_normalized = model(X_tensor).numpy()

print(f"   ✓ Predictions shape: {predictions_normalized.shape}")

# Load Z-score scaler to inverse transform
print("\n5. Loading Z-score scaler for inverse transform...")
with open(BASE_DIR / "results" / "target_scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# Get property names from training data
with open(BASE_DIR / "results" / "train_data.pkl", 'rb') as f:
    train_df = pickle.load(f)

target_properties = [col for col in train_df.columns if col not in ['Molecule Name', 'SMILES']]
print(f"   ✓ Target properties: {target_properties}")

# Apply inverse Z-score transform
print("\n6. Applying inverse Z-score transform...")
predictions_original = np.zeros_like(predictions_normalized)

for i, prop in enumerate(target_properties):
    mean = scaler[prop]['mean']
    std = scaler[prop]['std']
    predictions_original[:, i] = predictions_normalized[:, i] * std + mean
    print(f"   [{i+1}/9] {prop:30s}: mean={mean:8.2f}, std={std:8.2f}")

# Create predictions dataframe
print("\n7. Creating predictions dataframe...")
pred_df = pd.DataFrame({
    'Molecule Name': test_mol_names,
    'SMILES': test_smiles
})

for i, prop in enumerate(target_properties):
    pred_df[prop] = predictions_original[:, i]

print(f"   ✓ Predictions shape: {pred_df.shape}")

# Save predictions
output_path = BASE_DIR / "results" / "gnn_test_predictions.csv"
pred_df.to_csv(output_path, index=False)

print(f"\n8. Predictions saved:")
print(f"   ✓ File: {output_path}")
print(f"   ✓ Shape: {pred_df.shape}")

# Summary statistics
print("\n9. Prediction summary statistics:")
for prop in target_properties:
    vals = pred_df[prop].values
    print(f"   {prop:30s}: mean={vals.mean():8.2f}, std={vals.std():8.2f}, "
          f"min={vals.min():8.2f}, max={vals.max():8.2f}")

print("\n" + "=" * 80)
print("✓ GNN predictions complete!")
print("=" * 80)
