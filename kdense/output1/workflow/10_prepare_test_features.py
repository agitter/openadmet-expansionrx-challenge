#!/usr/bin/env python3
"""
Prepare Test Set Molecular Features

This script generates molecular features for the blind test set:
1. Load test data (SMILES only, no target values)
2. Generate Morgan fingerprints (2048-bit, radius=2)
3. Calculate RDKit 2D descriptors (217 features)
4. Save features for baseline model predictions

Author: K-Dense Coding Agent
Date: 2025-12-05
"""

import pickle
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
import warnings
import time

# Suppress warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# Set random seed
np.random.seed(42)

# Session directory
SESSION_DIR = "/app/sandbox/session_20251205_152206_4285cc85e60d"

print("=" * 80)
print("Test Set Feature Generation")
print("=" * 80)

# Load test data
print("\n1. Loading test data...")
test_df = pd.read_csv(f"{SESSION_DIR}/user_data/expansion_data_test_blinded.csv")
print(f"   ✓ Loaded: {len(test_df):,} test molecules")
print(f"   ✓ Columns: {list(test_df.columns)}")

# Verify expected columns
if 'SMILES' not in test_df.columns or 'Molecule Name' not in test_df.columns:
    raise ValueError(f"Expected 'SMILES' and 'Molecule Name' columns, got: {list(test_df.columns)}")

smiles_list = test_df['SMILES'].values
mol_names = test_df['Molecule Name'].values

print(f"\n2. Generating molecular features for {len(smiles_list):,} molecules...")
print("   This matches the baseline training approach:")
print("   - Morgan fingerprints: 2048-bit, radius=2")
print("   - RDKit 2D descriptors: 217 features")
print("   - Total: 2,265 features per molecule")

# Get descriptor names
descriptor_names = [desc[0] for desc in Descriptors._descList]
print(f"   ✓ RDKit descriptors available: {len(descriptor_names)}")

# Feature generation
morgan_fps = []
rdkit_descs = []
failed_indices = []

start_time = time.time()
for i, smiles in enumerate(smiles_list):
    if (i + 1) % 250 == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        eta = (len(smiles_list) - i - 1) / rate if rate > 0 else 0
        print(f"   Progress: {i+1:,}/{len(smiles_list):,} ({100*(i+1)/len(smiles_list):.1f}%) "
              f"- Rate: {rate:.0f} mol/s - ETA: {eta:.1f}s")

    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")

        # Morgan fingerprint (2048-bit, radius 2)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        morgan_fp = np.array(fp, dtype=np.float32)

        # RDKit descriptors
        desc_values = []
        for desc_name in descriptor_names:
            try:
                desc_func = getattr(Descriptors, desc_name)
                value = desc_func(mol)
                # Handle inf/nan values
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                desc_values.append(value)
            except:
                desc_values.append(0.0)

        rdkit_desc = np.array(desc_values, dtype=np.float32)

        # Append features
        morgan_fps.append(morgan_fp)
        rdkit_descs.append(rdkit_desc)

    except Exception as e:
        print(f"   ⚠ Failed to featurize molecule {i} ({mol_names[i]}): {e}")
        failed_indices.append(i)
        # Use zero features as fallback
        morgan_fps.append(np.zeros(2048, dtype=np.float32))
        rdkit_descs.append(np.zeros(len(descriptor_names), dtype=np.float32))

elapsed_time = time.time() - start_time
print(f"\n   ✓ Featurization complete in {elapsed_time:.1f} seconds")
print(f"   ✓ Success rate: {100*(len(smiles_list)-len(failed_indices))/len(smiles_list):.2f}%")
if failed_indices:
    print(f"   ⚠ Failed molecules: {len(failed_indices)}")

# Convert to arrays
morgan_fps = np.array(morgan_fps, dtype=np.float32)
rdkit_descs = np.array(rdkit_descs, dtype=np.float32)

# Combine features
features = np.hstack([morgan_fps, rdkit_descs])

print(f"\n3. Feature matrix created:")
print(f"   ✓ Shape: {features.shape}")
print(f"   ✓ Morgan fingerprints: {morgan_fps.shape[1]} features")
print(f"   ✓ RDKit descriptors: {rdkit_descs.shape[1]} features")
print(f"   ✓ Total features: {features.shape[1]}")

# Create feature names
fp_names = [f"Morgan_{i}" for i in range(2048)]
desc_names = descriptor_names
feature_names = fp_names + desc_names

# Save features
print("\n4. Saving test features...")
test_features = {
    'features': features,
    'molecule_names': mol_names,
    'smiles': smiles_list,
    'feature_names': feature_names,
    'failed_indices': failed_indices,
    'metadata': {
        'n_molecules': len(smiles_list),
        'n_features': features.shape[1],
        'morgan_radius': 2,
        'morgan_bits': 2048,
        'rdkit_descriptors': len(descriptor_names),
        'featurization_date': '2025-12-05',
        'success_rate': (len(smiles_list) - len(failed_indices)) / len(smiles_list)
    }
}

output_path = f"{SESSION_DIR}/results/baseline_features_test.pkl"
with open(output_path, 'wb') as f:
    pickle.dump(test_features, f, protocol=pickle.HIGHEST_PROTOCOL)

file_size_mb = len(pickle.dumps(test_features)) / (1024 * 1024)
print(f"   ✓ Saved to: {output_path}")
print(f"   ✓ File size: {file_size_mb:.1f} MB")

print("\n" + "=" * 80)
print("✓ Test feature generation complete!")
print("=" * 80)
