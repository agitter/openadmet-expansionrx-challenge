"""
Feature Engineering and ADMET Model Training
=============================================
Generates molecular features from SMILES and trains Random Forest models
for 9 ADMET properties in the OpenADMET ExpansionRx Blind Challenge.

Key Design Decisions:
- Morgan fingerprints (radius=2, 2048 bits) based on literature best practices
- Random Forest regressor for robust performance and interpretability
- Separate model per property to handle missing data effectively
- 5-fold cross-validation for performance assessment
- Efficient memory management for constrained environment

Date: 2024-12-20
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
import time
warnings.filterwarnings('ignore')

print("="*80)
print("OpenADMET ExpansionRx - Feature Engineering & Model Training")
print("="*80)

# Define ADMET properties
ADMET_PROPERTIES = [
    'LogD',
    'KSOL',
    'HLM CLint',
    'MLM CLint',
    'Caco-2 Permeability Papp A>B',
    'Caco-2 Permeability Efflux',
    'MPPB',
    'MBPB',
    'MGMB'
]

# ============================================================================
# STEP 1: Feature Generation from SMILES
# ============================================================================

def smiles_to_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """
    Convert SMILES string to Morgan fingerprint (ECFP).

    Args:
        smiles: SMILES string
        radius: Fingerprint radius (default=2 for ECFP4)
        n_bits: Number of bits in fingerprint vector

    Returns:
        numpy array of fingerprint or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return None

def smiles_to_rdkit_descriptors(smiles):
    """
    Calculate RDKit molecular descriptors from SMILES.

    Args:
        smiles: SMILES string

    Returns:
        dict of descriptor values or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        descriptors = {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
            'FractionCsp3': Descriptors.FractionCSP3(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol)
        }
        return descriptors
    except:
        return None

def generate_features(df, descriptor='morgan', radius=2, n_bits=2048):
    """
    Generate molecular features for all compounds in dataframe.

    Args:
        df: DataFrame with 'SMILES' column
        descriptor: 'morgan', 'rdkit', or 'combined'
        radius: Morgan fingerprint radius
        n_bits: Morgan fingerprint size

    Returns:
        Feature matrix (numpy array), valid indices
    """
    print(f"\n[Generating {descriptor} features for {len(df)} compounds...]")
    start_time = time.time()

    features_list = []
    valid_indices = []

    for idx, row in df.iterrows():
        smiles = row['SMILES']

        if descriptor == 'morgan':
            fp = smiles_to_morgan_fingerprint(smiles, radius, n_bits)
            if fp is not None:
                features_list.append(fp)
                valid_indices.append(idx)

        elif descriptor == 'rdkit':
            desc = smiles_to_rdkit_descriptors(smiles)
            if desc is not None:
                features_list.append(list(desc.values()))
                valid_indices.append(idx)

        elif descriptor == 'combined':
            fp = smiles_to_morgan_fingerprint(smiles, radius, n_bits)
            desc = smiles_to_rdkit_descriptors(smiles)
            if fp is not None and desc is not None:
                combined = np.concatenate([fp, list(desc.values())])
                features_list.append(combined)
                valid_indices.append(idx)

    elapsed = time.time() - start_time
    feature_matrix = np.array(features_list)
    print(f"Generated features: {feature_matrix.shape[0]} valid compounds, "
          f"{feature_matrix.shape[1]} features ({elapsed:.1f}s)")
    print(f"Invalid SMILES: {len(df) - len(valid_indices)}")

    return feature_matrix, valid_indices

# ============================================================================
# STEP 2: Load Data and Generate Features
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Loading Data")
print("="*80)

train_df = pd.read_csv('expansion_data_train.csv')
test_df = pd.read_csv('expansion_data_test_blinded.csv')

print(f"Training set: {len(train_df)} compounds")
print(f"Test set: {len(test_df)} compounds")

print("\n" + "="*80)
print("STEP 2: Feature Generation")
print("="*80)

# Generate Morgan fingerprints for training data
X_train_full, train_valid_idx = generate_features(
    train_df,
    descriptor='morgan',
    radius=2,
    n_bits=2048
)

# Generate Morgan fingerprints for test data
X_test_full, test_valid_idx = generate_features(
    test_df,
    descriptor='morgan',
    radius=2,
    n_bits=2048
)

# Update dataframes to keep only valid SMILES
train_df_valid = train_df.iloc[train_valid_idx].copy()
test_df_valid = test_df.iloc[test_valid_idx].copy()

print(f"\nValid training compounds: {len(train_df_valid)}")
print(f"Valid test compounds: {len(test_df_valid)}")

# ============================================================================
# STEP 3: Train Models for Each ADMET Property
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Training Random Forest Models")
print("="*80)

# Model configuration based on computational constraints
RF_CONFIG = {
    'n_estimators': 200,        # Reasonable for speed/performance tradeoff
    'max_depth': 30,            # Prevent overfitting
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',     # Efficient for high-dimensional data
    'n_jobs': -1,               # Use all available cores
    'random_state': 42
}

models = {}
performance = {}

for prop in ADMET_PROPERTIES:
    print(f"\n{'='*80}")
    print(f"Training model for: {prop}")
    print(f"{'='*80}")

    # Get non-missing values for this property
    mask = ~train_df_valid[prop].isna()
    X_prop = X_train_full[mask]
    y_prop = train_df_valid[prop][mask].values

    n_samples = len(y_prop)
    print(f"Training samples: {n_samples}")

    if n_samples < 50:
        print(f"⚠️  WARNING: Only {n_samples} samples available. Skipping this property.")
        models[prop] = None
        performance[prop] = {'n_samples': n_samples, 'cv_r2': None}
        continue

    # Train Random Forest model
    print("Training Random Forest...")
    start_time = time.time()

    rf_model = RandomForestRegressor(**RF_CONFIG)

    # Cross-validation (5-fold if enough data)
    n_folds = min(5, n_samples // 10)
    if n_folds >= 2:
        print(f"Performing {n_folds}-fold cross-validation...")
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(rf_model, X_prop, y_prop,
                                    cv=kf, scoring='r2', n_jobs=-1)
        cv_r2_mean = cv_scores.mean()
        cv_r2_std = cv_scores.std()
        print(f"CV R² = {cv_r2_mean:.3f} ± {cv_r2_std:.3f}")
    else:
        cv_r2_mean = None
        cv_r2_std = None
        print("Not enough samples for cross-validation")

    # Train final model on all available data
    rf_model.fit(X_prop, y_prop)

    # Training set performance
    y_train_pred = rf_model.predict(X_prop)
    train_r2 = r2_score(y_prop, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_prop, y_train_pred))
    train_mae = mean_absolute_error(y_prop, y_train_pred)

    elapsed = time.time() - start_time

    print(f"Training R² = {train_r2:.3f}")
    print(f"Training RMSE = {train_rmse:.3f}")
    print(f"Training MAE = {train_mae:.3f}")
    print(f"Training time: {elapsed:.1f}s")

    # Store model and performance metrics
    models[prop] = rf_model
    performance[prop] = {
        'n_samples': n_samples,
        'cv_r2_mean': cv_r2_mean,
        'cv_r2_std': cv_r2_std,
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'training_time': elapsed
    }

# ============================================================================
# STEP 4: Generate Predictions for Test Set
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Generating Predictions for Test Set")
print("="*80)

predictions = {}

for prop in ADMET_PROPERTIES:
    print(f"\nPredicting {prop}...")

    if models[prop] is None:
        print(f"  No model available (insufficient training data)")
        # Use median of training data as fallback
        median_val = train_df_valid[prop].median()
        if np.isnan(median_val):
            median_val = 0.0
        predictions[prop] = np.full(len(test_df_valid), median_val)
        print(f"  Using fallback median value: {median_val:.3f}")
    else:
        pred = models[prop].predict(X_test_full)
        predictions[prop] = pred
        print(f"  Predictions generated: min={pred.min():.3f}, "
              f"max={pred.max():.3f}, mean={pred.mean():.3f}")

# ============================================================================
# STEP 5: Create Submission CSV
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Creating Submission File")
print("="*80)

# Create output dataframe with exact column order as training data
output_df = pd.DataFrame()
output_df['Molecule Name'] = test_df_valid['Molecule Name']
output_df['SMILES'] = test_df_valid['SMILES']

for prop in ADMET_PROPERTIES:
    output_df[prop] = predictions[prop]

# Verify column names match exactly
expected_columns = ['Molecule Name', 'SMILES'] + ADMET_PROPERTIES
assert list(output_df.columns) == expected_columns, "Column names don't match!"

# Save predictions
output_filename = 'admet_predictions_test.csv'
output_df.to_csv(output_filename, index=False)
print(f"\n✓ Predictions saved to: {output_filename}")
print(f"  Shape: {output_df.shape}")
print(f"  Columns: {list(output_df.columns)}")

# Check for any NaN values in predictions
nan_counts = output_df[ADMET_PROPERTIES].isna().sum()
if nan_counts.sum() > 0:
    print("\n⚠️  WARNING: NaN values found in predictions:")
    print(nan_counts[nan_counts > 0])
else:
    print("\n✓ No NaN values in predictions")

# ============================================================================
# STEP 6: Save Models and Performance Report
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Saving Models and Generating Report")
print("="*80)

# Save trained models
print("\nSaving trained models...")
with open('trained_models.pkl', 'wb') as f:
    pickle.dump(models, f)
print("✓ Models saved to: trained_models.pkl")

# Create performance report
print("\nGenerating performance report...")
report_lines = []
report_lines.append("="*80)
report_lines.append("OpenADMET ExpansionRx Blind Challenge - Model Performance Report")
report_lines.append("="*80)
report_lines.append(f"\nDate: {time.strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append(f"\nTraining Data: {len(train_df_valid)} compounds")
report_lines.append(f"Test Data: {len(test_df_valid)} compounds")
report_lines.append(f"\nFeaturization: Morgan Fingerprints (radius=2, 2048 bits)")
report_lines.append(f"Model: Random Forest Regressor")
report_lines.append(f"  - n_estimators: {RF_CONFIG['n_estimators']}")
report_lines.append(f"  - max_depth: {RF_CONFIG['max_depth']}")
report_lines.append(f"  - max_features: {RF_CONFIG['max_features']}")

report_lines.append("\n" + "="*80)
report_lines.append("Performance by Property")
report_lines.append("="*80)

for prop in ADMET_PROPERTIES:
    perf = performance[prop]
    report_lines.append(f"\n{prop}:")
    report_lines.append(f"  Training samples: {perf['n_samples']}")

    if perf['cv_r2_mean'] is not None:
        report_lines.append(f"  Cross-validation R²: {perf['cv_r2_mean']:.3f} ± {perf['cv_r2_std']:.3f}")
        report_lines.append(f"  Training R²: {perf['train_r2']:.3f}")
        report_lines.append(f"  Training RMSE: {perf['train_rmse']:.3f}")
        report_lines.append(f"  Training MAE: {perf['train_mae']:.3f}")
        report_lines.append(f"  Training time: {perf['training_time']:.1f}s")
    else:
        report_lines.append("  ⚠️  Insufficient data for model training")

report_lines.append("\n" + "="*80)
report_lines.append("Key Findings")
report_lines.append("="*80)

# Calculate summary statistics
models_trained = sum(1 for m in models.values() if m is not None)
report_lines.append(f"\n✓ {models_trained}/{len(ADMET_PROPERTIES)} properties successfully modeled")

cv_r2_values = [p['cv_r2_mean'] for p in performance.values() if p['cv_r2_mean'] is not None]
if cv_r2_values:
    report_lines.append(f"✓ Mean CV R² across properties: {np.mean(cv_r2_values):.3f}")
    report_lines.append(f"✓ Median CV R²: {np.median(cv_r2_values):.3f}")
    report_lines.append(f"✓ Range: [{np.min(cv_r2_values):.3f}, {np.max(cv_r2_values):.3f}]")

report_text = "\n".join(report_lines)
print(report_text)

with open('model_performance_report.txt', 'w') as f:
    f.write(report_text)

print("\n✓ Report saved to: model_performance_report.txt")

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE!")
print("="*80)
print(f"\nOutput files:")
print(f"  1. admet_predictions_test.csv - Predictions for {len(test_df_valid)} test compounds")
print(f"  2. trained_models.pkl - Trained Random Forest models")
print(f"  3. model_performance_report.txt - Detailed performance metrics")
