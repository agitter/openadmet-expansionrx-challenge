#!/usr/bin/env python3
"""
Generate Baseline Features for ADMET Property Prediction

This script generates molecular features from SMILES strings using RDKit:
1. Morgan fingerprints (2048-bit, radius 2)
2. RDKit 2D descriptors (~200 features)

The combined feature matrix is saved for downstream modeling.

Author: K-Dense Coding Agent
Date: 2025-12-05
"""

import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import Tuple, List
import sys
import time

# Set random seed for reproducibility
np.random.seed(42)

# Session directory - MUST use absolute paths
SESSION_DIR = "/app/sandbox/session_20251205_152206_4285cc85e60d"


def generate_morgan_fingerprint(mol, radius=2, n_bits=2048):
    """
    Generate Morgan fingerprint for a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object
    radius : int
        Fingerprint radius (default: 2)
    n_bits : int
        Number of bits in fingerprint (default: 2048)

    Returns
    -------
    np.ndarray
        Binary fingerprint as numpy array
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)


def calculate_rdkit_descriptors(mol):
    """
    Calculate all available RDKit 2D descriptors for a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object

    Returns
    -------
    np.ndarray
        Array of descriptor values
    """
    descriptors = []
    for desc_name, desc_func in Descriptors.descList:
        try:
            value = desc_func(mol)
            # Handle NaN or inf values
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            descriptors.append(value)
        except Exception as e:
            # If descriptor calculation fails, use 0
            descriptors.append(0.0)

    return np.array(descriptors)


def featurize_molecule(smiles: str, radius=2, n_bits=2048) -> Tuple[np.ndarray, bool, str]:
    """
    Generate complete feature vector for a molecule from SMILES.

    Parameters
    ----------
    smiles : str
        SMILES string representation of molecule
    radius : int
        Morgan fingerprint radius
    n_bits : int
        Number of bits for Morgan fingerprint

    Returns
    -------
    features : np.ndarray or None
        Combined feature vector (Morgan + RDKit descriptors)
    success : bool
        Whether featurization was successful
    error_msg : str
        Error message if featurization failed
    """
    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, False, "Invalid SMILES - failed to parse"

        # Generate Morgan fingerprint
        morgan_fp = generate_morgan_fingerprint(mol, radius=radius, n_bits=n_bits)

        # Calculate RDKit descriptors
        rdkit_desc = calculate_rdkit_descriptors(mol)

        # Concatenate features
        features = np.concatenate([morgan_fp, rdkit_desc])

        return features, True, ""

    except Exception as e:
        return None, False, str(e)


def main():
    """Main execution function."""
    print("=" * 80)
    print("Baseline Feature Generation for ADMET Property Prediction")
    print("=" * 80)
    print()

    # Load training data
    print("Step 1: Loading training data...")
    train_data_path = f"{SESSION_DIR}/results/train_data.pkl"

    try:
        with open(train_data_path, 'rb') as f:
            train_data = pickle.load(f)
        print(f"✓ Loaded training data: {train_data.shape}")
        print(f"  Columns: {list(train_data.columns)}")
    except Exception as e:
        print(f"✗ Error loading training data: {e}")
        sys.exit(1)

    print()

    # Extract SMILES and molecule IDs
    print("Step 2: Extracting SMILES strings...")
    smiles_col = 'SMILES'
    id_col = 'Molecule Name'

    if smiles_col not in train_data.columns:
        print(f"✗ Error: '{smiles_col}' column not found in data")
        sys.exit(1)

    smiles_list = train_data[smiles_col].tolist()
    molecule_ids = train_data[id_col].tolist()
    n_molecules = len(smiles_list)

    print(f"✓ Extracted {n_molecules} SMILES strings")
    print()

    # Featurization parameters
    morgan_radius = 2
    morgan_bits = 2048

    # Get descriptor count
    n_descriptors = len(Descriptors.descList)
    n_features_total = morgan_bits + n_descriptors

    print("Step 3: Generating molecular features...")
    print(f"  Morgan fingerprint: radius={morgan_radius}, bits={morgan_bits}")
    print(f"  RDKit descriptors: {n_descriptors} features")
    print(f"  Total features: {n_features_total}")
    print()

    # Initialize feature matrix
    feature_matrix = np.zeros((n_molecules, n_features_total), dtype=np.float32)
    valid_mask = np.ones(n_molecules, dtype=bool)
    failed_molecules = []

    # Featurize each molecule
    start_time = time.time()
    for i, (mol_id, smiles) in enumerate(zip(molecule_ids, smiles_list)):
        # Progress logging every 100 molecules or every 10 seconds
        if i > 0 and (i % 100 == 0 or (time.time() - start_time) > 10):
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n_molecules - i) / rate if rate > 0 else 0
            print(f"  Progress: {i}/{n_molecules} ({100*i/n_molecules:.1f}%) - "
                  f"Rate: {rate:.1f} mol/s - ETA: {eta:.1f}s")
            start_time = time.time()  # Reset for next interval

        features, success, error_msg = featurize_molecule(smiles, radius=morgan_radius, n_bits=morgan_bits)

        if success:
            feature_matrix[i, :] = features
        else:
            valid_mask[i] = False
            failed_molecules.append({
                'molecule_id': mol_id,
                'smiles': smiles,
                'error': error_msg
            })

    print(f"✓ Featurization complete!")
    print()

    # Report results
    n_valid = valid_mask.sum()
    n_failed = len(failed_molecules)
    failure_rate = 100 * n_failed / n_molecules

    print("Step 4: Featurization Summary")
    print(f"  Total molecules: {n_molecules}")
    print(f"  Successfully featurized: {n_valid} ({100*n_valid/n_molecules:.2f}%)")
    print(f"  Failed: {n_failed} ({failure_rate:.2f}%)")
    print()

    if failed_molecules:
        print("  Failed molecules:")
        for fail_info in failed_molecules[:10]:  # Show first 10
            print(f"    - {fail_info['molecule_id']}: {fail_info['error']}")
        if len(failed_molecules) > 10:
            print(f"    ... and {len(failed_molecules) - 10} more")
        print()

    # Filter to valid molecules
    feature_matrix_valid = feature_matrix[valid_mask, :]
    molecule_ids_valid = [mol_id for mol_id, valid in zip(molecule_ids, valid_mask) if valid]

    # Create feature names
    morgan_names = [f"Morgan_{i}" for i in range(morgan_bits)]
    rdkit_names = [desc_name for desc_name, _ in Descriptors.descList]
    feature_names = morgan_names + rdkit_names

    print("Step 5: Saving features...")
    print(f"  Feature matrix shape: {feature_matrix_valid.shape}")
    print(f"  Feature matrix dtype: {feature_matrix_valid.dtype}")
    print(f"  Memory usage: {feature_matrix_valid.nbytes / 1024 / 1024:.2f} MB")
    print()

    # Save features
    output_data = {
        'features': feature_matrix_valid,
        'molecule_ids': molecule_ids_valid,
        'feature_names': feature_names,
        'valid_mask': valid_mask,
        'failed_molecules': failed_molecules,
        'featurization_params': {
            'morgan_radius': morgan_radius,
            'morgan_bits': morgan_bits,
            'n_rdkit_descriptors': n_descriptors,
            'n_total_features': n_features_total
        }
    }

    output_path = f"{SESSION_DIR}/results/baseline_features_train.pkl"
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f, protocol=4)
        print(f"✓ Features saved to: {output_path}")
        print(f"  File size: {pd.Series([output_path]).apply(lambda x: pd.io.common.get_filepath_or_buffer(x)[0]).values[0]}")
    except Exception as e:
        print(f"✗ Error saving features: {e}")
        sys.exit(1)

    print()
    print("=" * 80)
    print("Feature generation complete!")
    print("=" * 80)
    print()
    print("Next step: Train baseline LightGBM model using workflow/06_train_baseline_model.py")
    print()


if __name__ == "__main__":
    main()
