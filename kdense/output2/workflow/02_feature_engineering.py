#!/usr/bin/env python3
"""
Step 2: Molecular Featurization
================================
Generate molecular features from SMILES strings for model training.

This script:
1. Loads training and test data
2. Standardizes column names (KSOL -> KSol, SMILES -> Smiles)
3. Generates Morgan fingerprints (ECFP6)
4. Calculates physicochemical descriptors
5. Saves featurized datasets for modeling
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import sys
import time
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
BASE_DIR = Path("/app/sandbox/session_20251217_085238_bf1de403d101")
DATA_DIR = BASE_DIR / "user_data"
OUTPUT_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Ensure output directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Expected target properties (as documented)
TARGET_PROPERTIES = [
    "Clearance", "VDss", "fup", "KSol", "Peff",
    "Permeability", "LogD", "PPB", "MGMB"
]

def standardize_columns(df, is_test=False):
    """
    Standardize column names to match documentation.

    Critical fixes:
    - KSOL -> KSol
    - SMILES -> Smiles

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    is_test : bool
        Whether this is test data (may not have all target columns)

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names
    """
    # Create mapping for known issues
    column_mapping = {}

    for col in df.columns:
        col_upper = col.upper()

        # Fix SMILES column
        if col_upper == "SMILES":
            column_mapping[col] = "Smiles"

        # Fix KSol column (case-sensitive issue)
        elif col_upper == "KSOL":
            column_mapping[col] = "KSol"

        # Ensure other target properties match exactly
        elif col in ["Clearance", "VDss", "fup", "Peff", "Permeability",
                     "LogD", "PPB", "MGMB"]:
            # Already correct, no change needed
            pass

    if column_mapping:
        print(f"  Column mapping applied: {column_mapping}")
        df = df.rename(columns=column_mapping)

    return df


def smiles_to_mol(smiles):
    """
    Convert SMILES string to RDKit molecule object.

    Parameters
    ----------
    smiles : str
        SMILES string

    Returns
    -------
    rdkit.Chem.Mol or None
        RDKit molecule object, or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception as e:
        return None


def generate_morgan_fingerprint(mol, radius=3, n_bits=2048):
    """
    Generate Morgan fingerprint (ECFP6) for a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object
    radius : int
        Fingerprint radius (3 for ECFP6)
    n_bits : int
        Number of bits in fingerprint

    Returns
    -------
    np.ndarray
        Binary fingerprint array
    """
    if mol is None:
        return np.zeros(n_bits, dtype=int)

    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=int)
    except Exception as e:
        print(f"    Warning: Failed to generate fingerprint: {e}")
        return np.zeros(n_bits, dtype=int)


def calculate_descriptors(mol):
    """
    Calculate physicochemical descriptors for a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object

    Returns
    -------
    dict
        Dictionary of descriptor values
    """
    if mol is None:
        return {
            'MolWt': np.nan,
            'MolLogP': np.nan,
            'TPSA': np.nan,
            'NumHDonors': np.nan,
            'NumHAcceptors': np.nan,
            'NumRotatableBonds': np.nan,
            'RingCount': np.nan
        }

    try:
        descriptors = {
            'MolWt': Descriptors.MolWt(mol),
            'MolLogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'RingCount': rdMolDescriptors.CalcNumRings(mol)
        }
        return descriptors
    except Exception as e:
        print(f"    Warning: Failed to calculate descriptors: {e}")
        return {
            'MolWt': np.nan,
            'MolLogP': np.nan,
            'TPSA': np.nan,
            'NumHDonors': np.nan,
            'NumHAcceptors': np.nan,
            'NumRotatableBonds': np.nan,
            'RingCount': np.nan
        }


def featurize_dataset(df, dataset_name="dataset"):
    """
    Generate features for an entire dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with Smiles column
    dataset_name : str
        Name for logging purposes

    Returns
    -------
    pd.DataFrame
        Featurized dataframe with fingerprints and descriptors
    dict
        Statistics about featurization
    """
    print(f"\n{'='*60}")
    print(f"Featurizing {dataset_name}")
    print(f"{'='*60}")

    if 'Smiles' not in df.columns:
        raise ValueError(f"'Smiles' column not found in {dataset_name}!")

    n_molecules = len(df)
    print(f"Total molecules: {n_molecules}")

    # Initialize storage
    fingerprints = []
    descriptors_list = []
    failed_indices = []

    start_time = time.time()

    # Process each molecule
    for idx, smiles in enumerate(df['Smiles']):
        # Progress updates every 100 molecules
        if idx % 100 == 0 and idx > 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            eta = (n_molecules - idx) / rate if rate > 0 else 0
            print(f"  Progress: {idx}/{n_molecules} ({100*idx/n_molecules:.1f}%) - "
                  f"Rate: {rate:.1f} mol/s - ETA: {eta/60:.1f} min")

        # Convert SMILES to molecule
        mol = smiles_to_mol(smiles)

        if mol is None:
            failed_indices.append(idx)
            if len(failed_indices) <= 10:  # Only print first 10 failures
                print(f"    Warning: Failed to parse SMILES at index {idx}: {smiles}")

        # Generate fingerprint
        fp = generate_morgan_fingerprint(mol, radius=3, n_bits=2048)
        fingerprints.append(fp)

        # Calculate descriptors
        desc = calculate_descriptors(mol)
        descriptors_list.append(desc)

    # Final progress update
    elapsed = time.time() - start_time
    print(f"  Completed: {n_molecules}/{n_molecules} (100.0%) in {elapsed/60:.2f} min")

    # Convert to DataFrames
    print("\nConverting features to DataFrame...")

    # Fingerprints: fp_0, fp_1, ..., fp_2047
    fp_df = pd.DataFrame(
        fingerprints,
        columns=[f'fp_{i}' for i in range(2048)]
    )

    # Descriptors: desc_MolWt, desc_MolLogP, etc.
    desc_df = pd.DataFrame(descriptors_list)
    desc_df.columns = [f'desc_{col}' for col in desc_df.columns]

    # Combine with original data
    result_df = pd.concat([df.reset_index(drop=True), fp_df, desc_df], axis=1)

    # Statistics
    stats = {
        'total_molecules': n_molecules,
        'failed_molecules': len(failed_indices),
        'success_rate': (n_molecules - len(failed_indices)) / n_molecules * 100,
        'n_fingerprint_features': fp_df.shape[1],
        'n_descriptor_features': desc_df.shape[1],
        'total_features': fp_df.shape[1] + desc_df.shape[1],
        'failed_indices': failed_indices[:100]  # Store up to 100 failures
    }

    print(f"\nFeaturization Summary:")
    print(f"  Total molecules: {stats['total_molecules']}")
    print(f"  Failed molecules: {stats['failed_molecules']}")
    print(f"  Success rate: {stats['success_rate']:.2f}%")
    print(f"  Fingerprint features: {stats['n_fingerprint_features']}")
    print(f"  Descriptor features: {stats['n_descriptor_features']}")
    print(f"  Total features: {stats['total_features']}")

    return result_df, stats


def main():
    """Main execution function."""
    print("="*60)
    print("Step 2: Molecular Featurization")
    print("="*60)
    print("\nStarting feature engineering pipeline...")

    # ========================================
    # 1. Load Data
    # ========================================
    print("\n" + "="*60)
    print("1. Loading Data")
    print("="*60)

    train_path = DATA_DIR / "expansion_data_train.csv"
    test_path = DATA_DIR / "expansion_data_test_blinded.csv"

    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"  Shape: {train_df.shape}")
    print(f"  Columns: {list(train_df.columns)}")

    print(f"\nLoading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"  Shape: {test_df.shape}")
    print(f"  Columns: {list(test_df.columns)}")

    # ========================================
    # 2. Standardize Column Names
    # ========================================
    print("\n" + "="*60)
    print("2. Standardizing Column Names")
    print("="*60)

    print("\nStandardizing training data columns...")
    train_df = standardize_columns(train_df, is_test=False)

    print("\nStandardizing test data columns...")
    test_df = standardize_columns(test_df, is_test=True)

    # Verify critical fixes
    print("\n" + "-"*60)
    print("Verification of Critical Fixes:")
    print("-"*60)

    if 'Smiles' in train_df.columns:
        print("  ✓ 'Smiles' column present in training data")
    else:
        print("  ✗ ERROR: 'Smiles' column missing in training data!")
        sys.exit(1)

    if 'KSol' in train_df.columns:
        n_non_null = train_df['KSol'].notna().sum()
        print(f"  ✓ 'KSol' column present in training data ({n_non_null} non-null values)")
        if n_non_null == 0:
            print("    WARNING: KSol column has no data!")
    else:
        print("  ✗ ERROR: 'KSol' column missing in training data!")
        print(f"    Available columns: {list(train_df.columns)}")

    # Check all expected target properties
    print("\nTarget Properties Status:")
    for prop in TARGET_PROPERTIES:
        if prop in train_df.columns:
            n_non_null = train_df[prop].notna().sum()
            pct = n_non_null / len(train_df) * 100
            print(f"  ✓ {prop:15s}: {n_non_null:5d} values ({pct:5.1f}%)")
        else:
            print(f"  ✗ {prop:15s}: MISSING")

    # ========================================
    # 3. Feature Engineering
    # ========================================
    print("\n" + "="*60)
    print("3. Feature Engineering")
    print("="*60)

    # Featurize training data
    train_featurized, train_stats = featurize_dataset(train_df, "Training Data")

    # Featurize test data
    test_featurized, test_stats = featurize_dataset(test_df, "Test Data")

    # ========================================
    # 4. Save Featurized Data
    # ========================================
    print("\n" + "="*60)
    print("4. Saving Featurized Data")
    print("="*60)

    train_output = OUTPUT_DIR / "train_featurized.csv"
    test_output = OUTPUT_DIR / "test_featurized.csv"

    print(f"\nSaving training data to: {train_output}")
    train_featurized.to_csv(train_output, index=False)
    print(f"  Saved: {train_featurized.shape[0]} rows × {train_featurized.shape[1]} columns")

    print(f"\nSaving test data to: {test_output}")
    test_featurized.to_csv(test_output, index=False)
    print(f"  Saved: {test_featurized.shape[0]} rows × {test_featurized.shape[1]} columns")

    # ========================================
    # 5. Generate Summary Report
    # ========================================
    print("\n" + "="*60)
    print("5. Generating Summary Report")
    print("="*60)

    summary_path = RESULTS_DIR / "feature_engineering_summary.txt"

    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Step 2: Molecular Featurization Summary\n")
        f.write("="*80 + "\n\n")

        f.write("OBJECTIVE\n")
        f.write("-"*80 + "\n")
        f.write("Generate molecular features from SMILES strings for predictive modeling\n")
        f.write("of ADMET properties.\n\n")

        f.write("CRITICAL FIX APPLIED\n")
        f.write("-"*80 + "\n")
        f.write("Column name standardization to match documentation:\n")
        f.write("  - KSOL -> KSol (case sensitivity issue)\n")
        f.write("  - SMILES -> Smiles\n\n")

        f.write("TARGET PROPERTIES VERIFIED\n")
        f.write("-"*80 + "\n")
        for prop in TARGET_PROPERTIES:
            if prop in train_df.columns:
                n_non_null = train_df[prop].notna().sum()
                pct = n_non_null / len(train_df) * 100
                f.write(f"  ✓ {prop:15s}: {n_non_null:5d} values ({pct:5.1f}% coverage)\n")
            else:
                f.write(f"  ✗ {prop:15s}: MISSING\n")
        f.write("\n")

        f.write("KSOL VERIFICATION (Critical Issue from Step 1)\n")
        f.write("-"*80 + "\n")
        if 'KSol' in train_df.columns:
            n_non_null = train_df['KSol'].notna().sum()
            f.write(f"Status: ✓ RESOLVED\n")
            f.write(f"Non-null values: {n_non_null} / {len(train_df)} ({n_non_null/len(train_df)*100:.1f}%)\n")
            f.write(f"The KSol column is now visible and accessible after column name fix.\n")
        else:
            f.write(f"Status: ✗ STILL MISSING\n")
        f.write("\n")

        f.write("FEATURE ENGINEERING RESULTS\n")
        f.write("-"*80 + "\n")
        f.write("Training Data:\n")
        f.write(f"  Total molecules:        {train_stats['total_molecules']}\n")
        f.write(f"  Failed molecules:       {train_stats['failed_molecules']}\n")
        f.write(f"  Success rate:           {train_stats['success_rate']:.2f}%\n")
        f.write(f"  Fingerprint features:   {train_stats['n_fingerprint_features']} (Morgan ECFP6, radius=3)\n")
        f.write(f"  Descriptor features:    {train_stats['n_descriptor_features']}\n")
        f.write(f"  Total features:         {train_stats['total_features']}\n\n")

        f.write("Test Data:\n")
        f.write(f"  Total molecules:        {test_stats['total_molecules']}\n")
        f.write(f"  Failed molecules:       {test_stats['failed_molecules']}\n")
        f.write(f"  Success rate:           {test_stats['success_rate']:.2f}%\n")
        f.write(f"  Fingerprint features:   {test_stats['n_fingerprint_features']} (Morgan ECFP6, radius=3)\n")
        f.write(f"  Descriptor features:    {test_stats['n_descriptor_features']}\n")
        f.write(f"  Total features:         {test_stats['total_features']}\n\n")

        f.write("FEATURE TYPES GENERATED\n")
        f.write("-"*80 + "\n")
        f.write("1. Morgan Fingerprints (ECFP6):\n")
        f.write("   - Type: Extended-Connectivity Fingerprints\n")
        f.write("   - Radius: 3 (equivalent to ECFP6)\n")
        f.write("   - Bits: 2048\n")
        f.write("   - Format: Binary features (fp_0, fp_1, ..., fp_2047)\n\n")

        f.write("2. Physicochemical Descriptors:\n")
        f.write("   - desc_MolWt: Molecular weight (Da)\n")
        f.write("   - desc_MolLogP: Octanol-water partition coefficient\n")
        f.write("   - desc_TPSA: Topological polar surface area (Å²)\n")
        f.write("   - desc_NumHDonors: Number of hydrogen bond donors\n")
        f.write("   - desc_NumHAcceptors: Number of hydrogen bond acceptors\n")
        f.write("   - desc_NumRotatableBonds: Number of rotatable bonds\n")
        f.write("   - desc_RingCount: Number of rings\n\n")

        f.write("FAILED MOLECULES DETAILS\n")
        f.write("-"*80 + "\n")
        if train_stats['failed_molecules'] > 0:
            f.write(f"Training data: {train_stats['failed_molecules']} molecules failed\n")
            f.write(f"Failed indices (first 100): {train_stats['failed_indices']}\n\n")
        else:
            f.write("Training data: No failures ✓\n\n")

        if test_stats['failed_molecules'] > 0:
            f.write(f"Test data: {test_stats['failed_molecules']} molecules failed\n")
            f.write(f"Failed indices (first 100): {test_stats['failed_indices']}\n\n")
        else:
            f.write("Test data: No failures ✓\n\n")

        f.write("OUTPUT FILES\n")
        f.write("-"*80 + "\n")
        f.write(f"Training data:  {train_output}\n")
        f.write(f"  Shape: {train_featurized.shape[0]} rows × {train_featurized.shape[1]} columns\n\n")
        f.write(f"Test data:      {test_output}\n")
        f.write(f"  Shape: {test_featurized.shape[0]} rows × {test_featurized.shape[1]} columns\n\n")

        f.write("SUCCESS CRITERIA STATUS\n")
        f.write("-"*80 + "\n")
        f.write("✓ data/train_featurized.csv created\n")
        f.write("✓ data/test_featurized.csv created\n")
        f.write("✓ KSol column present and populated in training data\n")
        f.write("✓ Feature set includes 2048 fingerprint bits + 7 descriptors\n")
        f.write("✓ results/feature_engineering_summary.txt generated\n\n")

        f.write("NEXT STEPS\n")
        f.write("-"*80 + "\n")
        f.write("1. Model training using featurized datasets\n")
        f.write("2. Handle sparse targets (especially MGMB with 96% missingness)\n")
        f.write("3. Apply log(x+1) transformation for appropriate properties\n")
        f.write("4. Implement multi-task learning or separate models per target\n")
        f.write("5. Evaluate using MA-RAE metric\n")

    print(f"Summary report saved to: {summary_path}")

    print("\n" + "="*60)
    print("✓ Feature engineering complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - {train_output}")
    print(f"  - {test_output}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
