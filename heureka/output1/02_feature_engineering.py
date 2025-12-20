#!/usr/bin/env python3
"""
OpenADMET ExpansionRx Blind Challenge - Feature Engineering
Purpose: Generate RDKit descriptors + Morgan fingerprints + MACCS keys
         Create distribution-shifted validation split
Date: 2025-12-19
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("=" * 80)
print("FEATURE ENGINEERING PIPELINE")
print("=" * 80)

# Load data
train_df = pd.read_csv('expansion_data_train.csv')
test_df = pd.read_csv('expansion_data_test_blinded.csv')

target_cols = ['LogD', 'KSOL', 'HLM CLint', 'MLM CLint', 'Caco-2 Permeability Papp A>B',
               'Caco-2 Permeability Efflux', 'MPPB', 'MBPB', 'MGMB']

print(f"\nTraining: {len(train_df)} molecules")
print(f"Test: {len(test_df)} molecules")

# SMILES Validation
print("\n" + "=" * 80)
print("SMILES VALIDATION")
print("=" * 80)

def validate_smiles(smiles):
    """Validate SMILES and return mol object"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return mol
    except:
        return None

# Validate all SMILES
print("\nValidating training SMILES...")
train_mols = [validate_smiles(s) for s in train_df['SMILES']]
train_valid = [m is not None for m in train_mols]
print(f"Valid: {sum(train_valid)}/{len(train_valid)} ({sum(train_valid)/len(train_valid)*100:.1f}%)")

print("\nValidating test SMILES...")
test_mols = [validate_smiles(s) for s in test_df['SMILES']]
test_valid = [m is not None for m in test_mols]
print(f"Valid: {sum(test_valid)}/{len(test_valid)} ({sum(test_valid)/len(test_valid)*100:.1f}%)")

if not all(train_valid):
    invalid_idx = [i for i, v in enumerate(train_valid) if not v]
    print(f"\nInvalid training SMILES indices: {invalid_idx[:10]}...")

if not all(test_valid):
    invalid_idx = [i for i, v in enumerate(test_valid) if not v]
    print(f"\nInvalid test SMILES indices: {invalid_idx}")

# Feature extraction functions
def compute_rdkit_descriptors(mol):
    """Compute comprehensive RDKit descriptors"""
    if mol is None:
        return None

    try:
        desc = {
            # Basic properties
            'MolWt': Descriptors.MolWt(mol),
            'HeavyAtomMolWt': Descriptors.HeavyAtomMolWt(mol),
            'ExactMolWt': Descriptors.ExactMolWt(mol),
            'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
            'NumRadicalElectrons': Descriptors.NumRadicalElectrons(mol),

            # Lipophilicity/Solubility related
            'MolLogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),

            # Hydrogen bonding
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),

            # Rotatable bonds and flexibility
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),

            # Ring systems
            'RingCount': Descriptors.RingCount(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
            'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles(mol),
            'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles(mol),
            'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles(mol),
            'NumAliphaticCarbocycles': Descriptors.NumAliphaticCarbocycles(mol),
            'NumSaturatedHeterocycles': Descriptors.NumSaturatedHeterocycles(mol),
            'NumSaturatedCarbocycles': Descriptors.NumSaturatedCarbocycles(mol),

            # Atom counts
            'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
            'NHOHCount': Descriptors.NHOHCount(mol),
            'NOCount': Descriptors.NOCount(mol),

            # Fractions
            'FractionCSP3': Descriptors.FractionCSP3(mol),

            # Complexity
            'LabuteASA': Descriptors.LabuteASA(mol),
            'BalabanJ': Descriptors.BalabanJ(mol) if Descriptors.RingCount(mol) > 0 else 0,
            'BertzCT': Descriptors.BertzCT(mol),

            # Charge related
            'MaxPartialCharge': Descriptors.MaxPartialCharge(mol),
            'MinPartialCharge': Descriptors.MinPartialCharge(mol),
            'MaxAbsPartialCharge': Descriptors.MaxAbsPartialCharge(mol),
            'MinAbsPartialCharge': Descriptors.MinAbsPartialCharge(mol),

            # Chirality
            'NumChiralCenters': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),

            # Additional descriptors
            'MolMR': Descriptors.MolMR(mol),
            'Kappa1': Descriptors.Kappa1(mol),
            'Kappa2': Descriptors.Kappa2(mol),
            'Kappa3': Descriptors.Kappa3(mol),
            'Chi0': Descriptors.Chi0(mol),
            'Chi0n': Descriptors.Chi0n(mol),
            'Chi0v': Descriptors.Chi0v(mol),
            'Chi1': Descriptors.Chi1(mol),
            'Chi1n': Descriptors.Chi1n(mol),
            'Chi1v': Descriptors.Chi1v(mol),
            'Chi2n': Descriptors.Chi2n(mol),
            'Chi2v': Descriptors.Chi2v(mol),
            'Chi3n': Descriptors.Chi3n(mol),
            'Chi3v': Descriptors.Chi3v(mol),
            'Chi4n': Descriptors.Chi4n(mol),
            'Chi4v': Descriptors.Chi4v(mol),

            # Hall-Kier Alpha
            'HallKierAlpha': Descriptors.HallKierAlpha(mol),

            # PEOE VSA descriptors
            'PEOE_VSA1': Descriptors.PEOE_VSA1(mol),
            'PEOE_VSA10': Descriptors.PEOE_VSA10(mol),
            'PEOE_VSA11': Descriptors.PEOE_VSA11(mol),
            'PEOE_VSA12': Descriptors.PEOE_VSA12(mol),
            'PEOE_VSA13': Descriptors.PEOE_VSA13(mol),
            'PEOE_VSA14': Descriptors.PEOE_VSA14(mol),
            'PEOE_VSA2': Descriptors.PEOE_VSA2(mol),
            'PEOE_VSA3': Descriptors.PEOE_VSA3(mol),
            'PEOE_VSA4': Descriptors.PEOE_VSA4(mol),
            'PEOE_VSA5': Descriptors.PEOE_VSA5(mol),
            'PEOE_VSA6': Descriptors.PEOE_VSA6(mol),
            'PEOE_VSA7': Descriptors.PEOE_VSA7(mol),
            'PEOE_VSA8': Descriptors.PEOE_VSA8(mol),
            'PEOE_VSA9': Descriptors.PEOE_VSA9(mol),

            # SMR VSA descriptors
            'SMR_VSA1': Descriptors.SMR_VSA1(mol),
            'SMR_VSA10': Descriptors.SMR_VSA10(mol),
            'SMR_VSA2': Descriptors.SMR_VSA2(mol),
            'SMR_VSA3': Descriptors.SMR_VSA3(mol),
            'SMR_VSA4': Descriptors.SMR_VSA4(mol),
            'SMR_VSA5': Descriptors.SMR_VSA5(mol),
            'SMR_VSA6': Descriptors.SMR_VSA6(mol),
            'SMR_VSA7': Descriptors.SMR_VSA7(mol),
            'SMR_VSA8': Descriptors.SMR_VSA8(mol),
            'SMR_VSA9': Descriptors.SMR_VSA9(mol),

            # SlogP VSA descriptors
            'SlogP_VSA1': Descriptors.SlogP_VSA1(mol),
            'SlogP_VSA10': Descriptors.SlogP_VSA10(mol),
            'SlogP_VSA11': Descriptors.SlogP_VSA11(mol),
            'SlogP_VSA12': Descriptors.SlogP_VSA12(mol),
            'SlogP_VSA2': Descriptors.SlogP_VSA2(mol),
            'SlogP_VSA3': Descriptors.SlogP_VSA3(mol),
            'SlogP_VSA4': Descriptors.SlogP_VSA4(mol),
            'SlogP_VSA5': Descriptors.SlogP_VSA5(mol),
            'SlogP_VSA6': Descriptors.SlogP_VSA6(mol),
            'SlogP_VSA7': Descriptors.SlogP_VSA7(mol),
            'SlogP_VSA8': Descriptors.SlogP_VSA8(mol),
            'SlogP_VSA9': Descriptors.SlogP_VSA9(mol),

            # EState descriptors
            'MaxEStateIndex': Descriptors.MaxEStateIndex(mol),
            'MinEStateIndex': Descriptors.MinEStateIndex(mol),
            'MaxAbsEStateIndex': Descriptors.MaxAbsEStateIndex(mol),
            'MinAbsEStateIndex': Descriptors.MinAbsEStateIndex(mol),

            # Additional useful descriptors
            'qed': Descriptors.qed(mol),  # Drug-likeness score
        }

        # Handle NaN/Inf
        for key in desc:
            if not np.isfinite(desc[key]):
                desc[key] = 0

        return desc
    except Exception as e:
        print(f"Error computing descriptors: {e}")
        return None


def compute_morgan_fp(mol, radius=2, nBits=2048):
    """Compute Morgan fingerprint"""
    if mol is None:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return np.array(fp)
    except:
        return None


def compute_maccs_keys(mol):
    """Compute MACCS keys (166 bits)"""
    if mol is None:
        return None
    try:
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp)
    except:
        return None


# Compute features for all molecules
print("\n" + "=" * 80)
print("COMPUTING FEATURES")
print("=" * 80)

def compute_all_features(mols, smiles_list, name="data"):
    """Compute all features for a list of molecules"""
    n = len(mols)

    print(f"\nProcessing {name} ({n} molecules)...")

    # RDKit descriptors
    print("  Computing RDKit descriptors...")
    rdkit_descs = []
    for i, mol in enumerate(mols):
        if (i+1) % 1000 == 0:
            print(f"    Progress: {i+1}/{n}")
        desc = compute_rdkit_descriptors(mol)
        if desc is None:
            # Use zeros for invalid molecules
            desc = {k: 0 for k in compute_rdkit_descriptors(Chem.MolFromSmiles('C')).keys()}
        rdkit_descs.append(desc)
    rdkit_df = pd.DataFrame(rdkit_descs)
    rdkit_df.columns = ['rdkit_' + c for c in rdkit_df.columns]
    print(f"    RDKit descriptors: {rdkit_df.shape[1]} features")

    # Morgan fingerprints (radius 2, 2048 bits)
    print("  Computing Morgan fingerprints (r=2, 2048 bits)...")
    morgan_fps = []
    for i, mol in enumerate(mols):
        if (i+1) % 1000 == 0:
            print(f"    Progress: {i+1}/{n}")
        fp = compute_morgan_fp(mol, radius=2, nBits=2048)
        if fp is None:
            fp = np.zeros(2048)
        morgan_fps.append(fp)
    morgan_df = pd.DataFrame(np.vstack(morgan_fps))
    morgan_df.columns = [f'morgan2_{i}' for i in range(2048)]
    print(f"    Morgan fingerprints: {morgan_df.shape[1]} features")

    # Morgan fingerprints (radius 3, 1024 bits) - additional
    print("  Computing Morgan fingerprints (r=3, 1024 bits)...")
    morgan3_fps = []
    for i, mol in enumerate(mols):
        if (i+1) % 1000 == 0:
            print(f"    Progress: {i+1}/{n}")
        fp = compute_morgan_fp(mol, radius=3, nBits=1024)
        if fp is None:
            fp = np.zeros(1024)
        morgan3_fps.append(fp)
    morgan3_df = pd.DataFrame(np.vstack(morgan3_fps))
    morgan3_df.columns = [f'morgan3_{i}' for i in range(1024)]
    print(f"    Morgan r=3 fingerprints: {morgan3_df.shape[1]} features")

    # MACCS keys (166 bits)
    print("  Computing MACCS keys...")
    maccs_fps = []
    for i, mol in enumerate(mols):
        if (i+1) % 1000 == 0:
            print(f"    Progress: {i+1}/{n}")
        fp = compute_maccs_keys(mol)
        if fp is None:
            fp = np.zeros(167)
        maccs_fps.append(fp)
    maccs_df = pd.DataFrame(np.vstack(maccs_fps))
    maccs_df.columns = [f'maccs_{i}' for i in range(167)]
    print(f"    MACCS keys: {maccs_df.shape[1]} features")

    # SMILES-based features
    print("  Computing SMILES-derived features...")
    smiles_feats = pd.DataFrame({
        'smiles_len': [len(s) for s in smiles_list],
        'smiles_n_brackets': [s.count('[') + s.count(']') for s in smiles_list],
        'smiles_n_rings': [s.count('1') + s.count('2') + s.count('3') + s.count('4') + s.count('5') + s.count('6') for s in smiles_list],
        'smiles_n_at': [s.count('@') for s in smiles_list],
        'smiles_n_plus': [s.count('+') for s in smiles_list],
        'smiles_n_minus': [s.count('-') for s in smiles_list],
    })
    print(f"    SMILES features: {smiles_feats.shape[1]} features")

    # Combine all features
    all_features = pd.concat([rdkit_df, morgan_df, morgan3_df, maccs_df, smiles_feats], axis=1)
    print(f"\n  Total features: {all_features.shape[1]}")

    return all_features


# Compute features
train_features = compute_all_features(train_mols, train_df['SMILES'].tolist(), "training")
test_features = compute_all_features(test_mols, test_df['SMILES'].tolist(), "test")

# Verify shapes
print(f"\nFeature matrix shapes:")
print(f"  Training: {train_features.shape}")
print(f"  Test: {test_features.shape}")

# Create validation split that mimics test distribution shift
print("\n" + "=" * 80)
print("CREATING VALIDATION SPLIT")
print("=" * 80)

# Strategy: Use longer SMILES from training as validation (mimics test distribution)
train_smiles_len = train_df['SMILES'].str.len()
test_mean_len = test_df['SMILES'].str.len().mean()

# Use molecules with SMILES length >= test mean as validation
# This creates a distribution-shifted validation set
val_mask = train_smiles_len >= test_mean_len
n_potential_val = val_mask.sum()
print(f"\nMolecules with SMILES len >= {test_mean_len:.1f} (test mean): {n_potential_val}")

# We want ~15-20% for validation
target_val_pct = 0.15
target_val_n = int(len(train_df) * target_val_pct)
print(f"Target validation size: {target_val_n} (~{target_val_pct*100:.0f}%)")

# If we have enough long molecules, use them; otherwise lower the threshold
if n_potential_val >= target_val_n:
    # Sort by SMILES length and take the longest ones
    len_sorted_idx = train_smiles_len.argsort()[::-1]
    val_indices = len_sorted_idx[:target_val_n].values
    train_indices = len_sorted_idx[target_val_n:].values
else:
    # Use all long molecules + random sample from shorter ones
    long_indices = train_smiles_len[val_mask].index.tolist()
    short_indices = train_smiles_len[~val_mask].index.tolist()
    n_additional = target_val_n - len(long_indices)
    np.random.shuffle(short_indices)
    additional_indices = short_indices[:n_additional]
    val_indices = np.array(long_indices + additional_indices)
    train_indices = np.array([i for i in range(len(train_df)) if i not in val_indices])

print(f"\nSplit results:")
print(f"  Training set: {len(train_indices)}")
print(f"  Validation set: {len(val_indices)}")

# Verify distribution shift in split
train_split_len = train_smiles_len.iloc[train_indices].mean()
val_split_len = train_smiles_len.iloc[val_indices].mean()
print(f"\nSMILES length check:")
print(f"  Training split mean: {train_split_len:.1f}")
print(f"  Validation split mean: {val_split_len:.1f}")
print(f"  Test set mean: {test_mean_len:.1f}")

# Save processed data
print("\n" + "=" * 80)
print("SAVING PROCESSED DATA")
print("=" * 80)

# Save feature matrices
train_features.to_pickle('train_features.pkl')
test_features.to_pickle('test_features.pkl')
print("Saved: train_features.pkl, test_features.pkl")

# Save split indices
np.save('train_indices.npy', train_indices)
np.save('val_indices.npy', val_indices)
print("Saved: train_indices.npy, val_indices.npy")

# Save targets
train_targets = train_df[target_cols].copy()
train_targets.to_pickle('train_targets.pkl')
print("Saved: train_targets.pkl")

# Save molecule info
train_df[['Molecule Name', 'SMILES']].to_csv('train_mol_info.csv', index=False)
test_df[['Molecule Name', 'SMILES']].to_csv('test_mol_info.csv', index=False)
print("Saved: train_mol_info.csv, test_mol_info.csv")

# Summary statistics
print("\n" + "=" * 80)
print("FEATURE SUMMARY")
print("=" * 80)

print(f"\nRDKit descriptors: {len([c for c in train_features.columns if c.startswith('rdkit_')])}")
print(f"Morgan r=2 fingerprints: {len([c for c in train_features.columns if c.startswith('morgan2_')])}")
print(f"Morgan r=3 fingerprints: {len([c for c in train_features.columns if c.startswith('morgan3_')])}")
print(f"MACCS keys: {len([c for c in train_features.columns if c.startswith('maccs_')])}")
print(f"SMILES features: {len([c for c in train_features.columns if c.startswith('smiles_')])}")
print(f"\nTotal features: {train_features.shape[1]}")

# Check for constant features
const_features = train_features.columns[train_features.std() == 0].tolist()
print(f"\nConstant features (will be removed): {len(const_features)}")

print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 80)
