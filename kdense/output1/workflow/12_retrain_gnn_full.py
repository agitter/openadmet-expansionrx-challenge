#!/usr/bin/env python3
"""
Retrain GNN Model on Full Training Set

This script trains a multi-task neural network on the COMPLETE training dataset.
Uses same architecture and approach as cross-validation:
- Z-score normalized targets
- Morgan fingerprints as input (2048-bit)
- Multi-task architecture with shared encoder
- 10% validation split for early stopping

Author: K-Dense Coding Agent
Date: 2025-12-05
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

# Suppress warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

print("=" * 80)
print("Retrain GNN Model on Full Training Set")
print("=" * 80)

# Paths
BASE_DIR = Path("/app/sandbox/session_20251205_152206_4285cc85e60d")
DATA_PATH = BASE_DIR / "workflow" / "gnn_train_data_normalized.csv"
OUTPUT_DIR = BASE_DIR / "results"

# Load data
print("\n1. Loading normalized training data...")
data = pd.read_csv(DATA_PATH)
print(f"   ✓ Loaded: {data.shape[0]:,} molecules × {data.shape[1]} columns")

# Extract SMILES and targets
smiles = data['smiles'].values
target_cols = [col for col in data.columns if col != 'smiles']
targets = data[target_cols].values
print(f"   ✓ Target properties: {len(target_cols)}")

# Generate molecular features
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
valid_indices = []

for i, smi in enumerate(smiles):
    fp = smiles_to_fingerprint(smi)
    if fp is not None:
        features.append(fp)
        valid_indices.append(i)
    if (i + 1) % 500 == 0:
        print(f"   Progress: {i+1}/{len(smiles)} molecules...")

features = np.array(features, dtype=np.float32)
targets = targets[valid_indices]
smiles = smiles[valid_indices]

print(f"   ✓ Features shape: {features.shape}")
print(f"   ✓ Valid molecules: {len(valid_indices):,}")

# Split into train/val (90/10)
print("\n3. Creating train/validation split (90/10)...")
n_total = len(features)
n_val = int(0.1 * n_total)
n_train = n_total - n_val

# Shuffle indices
indices = np.arange(n_total)
np.random.shuffle(indices)

train_idx = indices[:n_train]
val_idx = indices[n_train:]

X_train, y_train = features[train_idx], targets[train_idx]
X_val, y_val = features[val_idx], targets[val_idx]

print(f"   ✓ Training set: {len(train_idx):,} molecules")
print(f"   ✓ Validation set: {len(val_idx):,} molecules")

# Define PyTorch Dataset
class MoleculeDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Define Multi-Task Neural Network
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
        # Task-specific heads
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_tasks)])

    def forward(self, x):
        h = self.encoder(x)
        outputs = [head(h) for head in self.heads]
        return torch.cat(outputs, dim=1)

# Masked MSE Loss
def masked_mse_loss(pred, target):
    """MSE loss that ignores NaN values."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return ((pred[mask] - target[mask]) ** 2).mean()

# Create datasets and loaders
train_dataset = MoleculeDataset(X_train, y_train)
val_dataset = MoleculeDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)

# Initialize model
print("\n4. Initializing multi-task neural network...")
model = MultiTaskNN(input_dim=2048, hidden_dim=300, n_tasks=9)
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_params = sum(p.numel() for p in model.parameters())
print(f"   ✓ Model parameters: {n_params:,}")
print(f"   ✓ Architecture: 2048 → 300 → 300 → 9 tasks")

# Training loop
print("\n5. Training model...")
print("   Max epochs: 30")
print("   Early stopping patience: 5 epochs")
print("   Batch size: 50")

best_val_loss = float('inf')
patience_counter = 0
patience = 5

start_time = time.time()

for epoch in range(30):
    # Training
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = masked_mse_loss(pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            pred = model(X_batch)
            loss = masked_mse_loss(pred, y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"   Epoch {epoch+1:2d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"   Early stopping triggered at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(best_model_state)

elapsed = time.time() - start_time
print(f"\n   ✓ Training complete in {elapsed/60:.2f} minutes")
print(f"   ✓ Best validation loss: {best_val_loss:.4f}")

# Save model
print("\n6. Saving trained model...")
model_path = OUTPUT_DIR / "gnn_model_full.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'input_dim': 2048,
        'hidden_dim': 300,
        'n_tasks': 9
    },
    'training_info': {
        'best_val_loss': best_val_loss,
        'n_train_samples': n_train,
        'n_val_samples': n_val,
        'training_time_minutes': elapsed / 60
    }
}, model_path)

print(f"   ✓ Model saved: {model_path}")

print("\n" + "=" * 80)
print("✓ GNN model retraining complete!")
print("=" * 80)
