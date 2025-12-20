# OpenADMET ExpansionRx Blind Challenge - Model Report

## Executive Summary

This report documents the development of an ensemble machine learning model for predicting 9 ADMET properties from molecular SMILES strings. The model achieves **Spearman correlation of 0.815** and **MA-RAE of 0.508** on cross-validation, meeting the target performance of ≥0.81 Spearman and <0.53 MA-RAE.

## Dataset Overview

### Training Data
- **Compounds**: 5,326 molecules with SMILES
- **Properties**: 9 ADMET endpoints (sparse, hierarchical data)
- **Availability**: 4.2% (MGMB) to 96.3% (KSOL)

### Test Data
- **Compounds**: 2,282 molecules (blinded)
- **Distribution shift**: Test molecules are ~20% longer (57.8 vs 48.0 SMILES characters)

### Target Properties

| Property | Available Samples | % Available |
|----------|------------------|-------------|
| LogD | 5,039 | 94.6% |
| KSOL | 5,128 | 96.3% |
| HLM CLint | 3,759 | 70.6% |
| MLM CLint | 4,522 | 84.9% |
| Caco-2 Papp A>B | 2,157 | 40.5% |
| Caco-2 Efflux | 2,161 | 40.6% |
| MPPB | 1,302 | 24.4% |
| MBPB | 975 | 18.3% |
| MGMB | 222 | 4.2% |

## Model Architecture

### Feature Engineering

**Total Features: 3,263**

1. **RDKit Descriptors (88 features)**
   - Molecular properties: MolWt, LogP, TPSA, HBD/HBA
   - Ring descriptors: aromatic, aliphatic, heterocyclic counts
   - Complexity metrics: BertzCT, HallKierAlpha, Kappa indices
   - VSA descriptors: PEOE, SMR, SlogP surface area bins

2. **Morgan Fingerprints r=2 (2,009 features)**
   - 2048-bit ECFP-like circular fingerprints
   - Radius 2 captures local atom environments

3. **Morgan Fingerprints r=3 (1,023 features)**
   - 1024-bit extended circular fingerprints
   - Captures larger structural motifs

4. **MACCS Keys (137 features)**
   - 166-bit structural key fingerprints
   - Interpretable substructure patterns

5. **SMILES-derived (6 features)**
   - String length, bracket counts, ring indicators
   - Stereochemistry markers

### Model Configuration

**Algorithm**: LightGBM (Gradient Boosted Decision Trees)

**Key Strategies**:
1. **Property-specific configurations**: Each target has optimized hyperparameters
2. **Log-transformation**: Applied to highly skewed properties (HLM CLint, MLM CLint, Caco-2 Efflux)
3. **Z-score normalization**: Targets normalized before training
4. **5-fold cross-validation**: Robust performance estimation
5. **Ensemble of 3 configurations per property**: Model diversity

**Hyperparameter Ranges**:
- `num_leaves`: 7-127 (property-dependent)
- `max_depth`: 4-12
- `learning_rate`: 0.02-0.08
- `feature_fraction`: 0.5-0.9
- `min_child_samples`: 5-50

### Training Procedure

```
For each property:
  1. Z-score normalize targets (optionally log-transform)
  2. For each of 3 LightGBM configurations:
     a. 5-fold cross-validation
     b. Early stopping on validation MAE (50 rounds patience)
     c. Max 500 boosting rounds
  3. Weighted ensemble based on CV Spearman correlation
  4. Denormalize predictions
```

## Results

### Cross-Validation Performance

| Property | n_samples | Spearman | MA-RAE |
|----------|-----------|----------|--------|
| LogD | 5,039 | **0.939** | 0.308 |
| KSOL | 5,128 | 0.771 | 0.472 |
| HLM CLint | 3,759 | 0.789 | 0.559 |
| MLM CLint | 4,522 | **0.837** | 0.555 |
| Caco-2 Papp A>B | 2,157 | 0.778 | 0.563 |
| Caco-2 Efflux | 2,161 | 0.749 | 0.538 |
| MPPB | 1,302 | **0.839** | 0.471 |
| MBPB | 975 | 0.815 | 0.543 |
| MGMB | 222 | 0.817 | 0.561 |

**Overall Performance**:
- **Simple Average Spearman**: 0.815 ✓ (target: ≥0.81)
- **Simple Average MA-RAE**: 0.508 ✓ (target: <0.53)
- **Weighted Average Spearman**: 0.823
- **Weighted Average MA-RAE**: 0.484

### Key Observations

1. **Best performing properties**: LogD (0.94), MPPB (0.84), MLM CLint (0.84)
2. **Most challenging**: Caco-2 Efflux (0.75), KSOL (0.77)
3. **Sparse property success**: MGMB achieved 0.82 Spearman with only 222 samples
4. **Log-transform impact**: Significantly improved HLM CLint (from 0.15 to 0.79)

## Files Generated

### Predictions
- `admet_predictions_final.csv` - **Final submission file** (2,282 compounds × 9 properties)

### Models
- `cv_ensemble_models.pkl` - Trained model weights and scalers
- `ensemble_models.pkl` - Intermediate ensemble models
- `baseline_models.pkl` - Single-task baseline models

### Results
- `final_results.csv` - Per-property CV performance metrics
- `cv_ensemble_results.csv` - Detailed CV results
- `baseline_results.csv` - Baseline model metrics

### Features
- `train_features.pkl` - Training feature matrix
- `test_features.pkl` - Test feature matrix

### Visualizations
- `data_exploration_plots.png` - Missingness and correlation analysis
- `target_distributions.png` - Target value distributions

## Reproducibility

All scripts are provided for full reproducibility:

1. `01_data_exploration.py` - Data analysis and visualization
2. `02_feature_engineering.py` - Feature extraction from SMILES
3. `03_baseline_lightgbm.py` - Baseline single-task models
4. `04_advanced_ensemble.py` - Multi-configuration ensemble
5. `05_cv_ensemble.py` - Cross-validation ensemble
6. `06_final_model.py` - Final optimized model
7. `07_finalize_predictions.py` - Value clipping and validation

**Random seed**: 42 (set in all scripts)

## Technical Considerations

### Distribution Shift Handling
- Test molecules are 20% longer than training (57.8 vs 48.0 chars)
- Validation split created using longer training molecules to mimic shift
- Model generalizes well despite shift

### Missing Data Strategy
- No imputation of target values
- Models trained only on available data per property
- Cross-property correlations not explicitly exploited (potential improvement)

### Prediction Post-processing
- Negative values clipped to 0 for non-negative properties
- No extreme extrapolation clipping required (predictions within 50% margin of training range)

## Potential Improvements

1. **Graph Neural Networks**: Chemprop or GNN encoders could capture molecular topology better
2. **Multi-task learning**: Explicit modeling of property correlations
3. **Transfer learning**: Pre-trained molecular embeddings (ChemBERTa, MolBERT)
4. **Data augmentation**: SMILES enumeration for training augmentation
5. **Uncertainty quantification**: Conformal prediction for confidence intervals

## Conclusion

The developed LightGBM ensemble model successfully predicts 9 ADMET properties with Spearman correlation of 0.815 and MA-RAE of 0.508, meeting competition targets. Key success factors:

1. Comprehensive feature engineering (RDKit + fingerprints)
2. Property-specific hyperparameter tuning
3. Log-transformation for skewed targets
4. Robust cross-validation ensemble strategy

The model is ready for submission to the OpenADMET ExpansionRx Blind Challenge.
