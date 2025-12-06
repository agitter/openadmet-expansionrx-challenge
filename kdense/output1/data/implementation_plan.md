# Implementation Plan: Baseline LightGBM Model Development

**Date**: 2025-12-05
**Task**: Develop and evaluate a baseline LightGBM model for ADMET property prediction
**Session**: session_20251205_152206_4285cc85e60d

## Task Overview
Implement a featurization and training pipeline for a baseline LightGBM model using traditional molecular descriptors (Morgan fingerprints + RDKit 2D descriptors). This model will serve as a performance benchmark for subsequent advanced models.

## Prerequisites

**Required Input Files**:
- `results/train_data.pkl` - Processed training data (5,326 molecules × 11 columns)
- `results/01_initial_review.md` - Data analysis identifying skewed properties

**Required Libraries**:
- `rdkit-pypi` - Chemical featurization
- `lightgbm` - Gradient boosting model

**Literature Guidance**:
- Kosmos AI achieved Spearman 0.8087 with Morgan (2048-bit, radius 2) + RDKit descriptors
- Log transformation required for: HLM CLint, MLM CLint, Caco-2 Efflux, MBPB
- Scaffold-based splitting critical for robust evaluation
- 5-fold cross-validation for stable estimates

## Implementation Steps

### Step 1: Environment Setup
**Objective**: Install required cheminformatics libraries

**Actions**:
1. Verify UV installation (already done in previous work)
2. Add `rdkit-pypi` package
3. Add `lightgbm` package
4. Run `uv sync`
5. Verify imports work
6. Document versions

**Expected Duration**: 2-3 minutes

### Step 2: Create Featurization Script
**Objective**: Generate molecular features from SMILES

**Script**: `workflow/05_generate_baseline_features.py`

**Algorithm**:
1. Load training data from `results/train_data.pkl`
2. For each SMILES:
   - Parse with RDKit
   - Generate Morgan fingerprints (2048-bit, radius 2)
   - Calculate RDKit 2D descriptors (~200 features)
   - Concatenate into feature vector
3. Handle errors gracefully
4. Save to `results/baseline_features_train.pkl`

**Expected Duration**: 5-10 minutes

### Step 3: Create Training and Evaluation Script
**Objective**: Train LightGBM with scaffold-based CV

**Script**: `workflow/06_train_baseline_model.py`

**Algorithm**:
1. Load features and targets
2. For each of 9 target properties:
   - Filter molecules with non-missing target
   - Apply log-transform if skewed (HLM, MLM, Caco-2 Efflux, MBPB)
   - Generate Murcko scaffolds for splitting
   - Implement 5-fold scaffold-based CV
   - Train LGBMRegressor per fold
   - Calculate Spearman and MA-RAE
3. Aggregate results across all tasks
4. Save to `results/baseline_cv_scores.csv`

**Expected Duration**: 15-30 minutes

### Step 4: Validation and Quality Checks
**Objective**: Verify outputs and assess performance

**Actions**:
1. Load and inspect CV scores
2. Calculate average metrics
3. Compare to literature benchmarks
4. Identify best/worst properties
5. Check for anomalies

**Expected Duration**: 2-3 minutes

### Step 5: Documentation and Summary
**Objective**: Update documentation

**Actions**:
1. Update README.md
2. Update manifest.json
3. Create execution summary

**Expected Duration**: 3-5 minutes

## Success Criteria
- ✅ Environment updated with rdkit-pypi and lightgbm
- ✅ Featurization script runs without errors
- ✅ Training script completes 5-fold CV for all 9 properties
- ✅ CV scores file created with valid metrics
- ✅ Average Spearman correlation logged
- ✅ Mean Spearman >= 0.70 (conservative baseline)

## Expected Outputs
1. `pyproject.toml` (modified) - Added rdkit-pypi, lightgbm
2. `workflow/05_generate_baseline_features.py` - Featurization script
3. `workflow/06_train_baseline_model.py` - Training script
4. `results/baseline_features_train.pkl` - Feature matrix (~20-40 MB)
5. `results/baseline_cv_scores.csv` - Performance metrics
6. `README.md` (updated) - Comprehensive documentation
7. `manifest.json` (updated) - File registry

## Performance Targets
- Mean Spearman (all 9 tasks): ≥ 0.75 (target), ≥ 0.80 (stretch)
- Tier 1 properties (LogD, KSOL): ≥ 0.85
- Tier 2 properties (HLM, MLM): ≥ 0.75
- Tier 3 properties (Caco-2): ≥ 0.65
- Tier 4 properties (MPPB, MBPB, MGMB): ≥ 0.60

**Benchmark**: Kosmos AI baseline = 0.8087 mean Spearman

## Timeline Estimate
- **Total**: 30-50 minutes

## Next Steps After Completion
1. Review baseline performance metrics
2. Implement advanced GNN model (Track 2)
3. Develop ensemble (Track 3)
4. Generate final test predictions
