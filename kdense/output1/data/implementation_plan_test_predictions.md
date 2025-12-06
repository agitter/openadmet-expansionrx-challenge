# Implementation Plan: Generate Test Set Predictions

## Objective
Generate predictions for the blind test set (`expansion_data_test_blinded.csv`) using both the baseline LightGBM model and the advanced GNN model, addressing the critical missing deliverable identified by the review agent.

## Context
- **Baseline model CV performance**: Mean Spearman 0.7682
- **GNN model CV performance**: Mean Spearman 0.7779
- **Issue**: Models were trained for cross-validation but not saved. Test predictions were never generated.
- **Solution**: Retrain both models on the full training set, then generate test predictions.

## Implementation Steps

### Step 1: Environment Verification
- Verify UV package manager is installed and working
- Check that all required packages are available (rdkit, lightgbm, torch, pandas, numpy, scipy)
- Run `uv sync` to ensure environment consistency

### Step 2: Data Preparation for Test Set
**Script**: `workflow/10_prepare_test_features.py`
- Load test data from `user_data/expansion_data_test_blinded.csv`
- Generate same molecular features used in training:
  - Morgan fingerprints (2048-bit, radius=2)
  - RDKit 2D descriptors (217 features)
- Save test features for baseline model
- Save test data for GNN model (SMILES only, no targets)

### Step 3: Retrain Baseline Model on Full Training Set
**Script**: `workflow/11_retrain_baseline_full.py`
- Load full training data and baseline features
- Train 9 separate LightGBM models (one per property) on ALL training data
- Apply same log-transforms as in CV: HLM CLint, MLM CLint, Caco-2 Efflux, MBPB
- Use same hyperparameters: 500 trees, lr=0.05, max_depth=8
- Save trained models to `results/baseline_models/` (9 .pkl files)
- Log training metrics

### Step 4: Retrain GNN Model on Full Training Set
**Script**: `workflow/12_retrain_gnn_full.py`
- Load normalized training data from `workflow/gnn_train_data_normalized.csv`
- Train multi-task neural network on ALL training data
- Use same architecture: 2048 input → 300 hidden × 2 layers → 9 outputs
- Training: Adam optimizer, lr=0.001, early stopping on validation split (10%)
- Save trained model to `results/gnn_model_full.pt`
- Log training metrics

### Step 5: Generate Baseline Predictions
**Script**: `workflow/13_predict_baseline.py`
- Load 9 trained baseline models from `results/baseline_models/`
- Load test features
- Generate predictions for each property
- Apply inverse log-transform where needed
- Save predictions to `results/baseline_test_predictions.csv`
- Format: [Molecule Name, SMILES, 9 property columns]

### Step 6: Generate GNN Predictions
**Script**: `workflow/14_predict_gnn.py`
- Load trained GNN model from `results/gnn_model_full.pt`
- Load test data and generate Morgan fingerprints
- Generate predictions (normalized scale)
- Apply inverse Z-score transform using `results/target_scaler.pkl`
- Save predictions to `results/gnn_test_predictions.csv`
- Format: [Molecule Name, SMILES, 9 property columns]

### Step 7: Create Ensemble Predictions
**Script**: `workflow/15_create_ensemble.py`
- Load baseline and GNN predictions
- Create simple ensemble: arithmetic mean of both models
- Save ensemble predictions to `results/ensemble_test_predictions.csv`
- Format: [Molecule Name, SMILES, 9 property columns]

### Step 8: Submission File Preparation
**Script**: `workflow/16_prepare_submission.py`
- Load ensemble predictions (best performing approach)
- Validate format matches leaderboard requirements
- Check for missing values, invalid predictions
- Create final submission file: `results/test_predictions.csv`
- Generate submission summary statistics

### Step 9: Update Documentation
- Update `README.md` with:
  - Full training results
  - Test prediction generation process
  - Model performance comparison
  - File locations and descriptions
- Update `manifest.json` with all new artifacts

### Step 10: Final Verification
- Verify all expected output files exist
- Check prediction file format and dimensions
- Validate no NaN/Inf values in final predictions
- Generate execution summary

## Success Criteria

✅ **Primary Deliverables**:
- [ ] Baseline model retrained on full training set
- [ ] GNN model retrained on full training set
- [ ] Baseline test predictions generated
- [ ] GNN test predictions generated
- [ ] Ensemble predictions created
- [ ] Final `test_predictions.csv` file created
- [ ] All output files properly formatted

✅ **Quality Standards**:
- [ ] Test predictions cover all 2,282 test molecules
- [ ] Predictions exist for all 9 properties
- [ ] No NaN or Inf values in final predictions
- [ ] Format matches leaderboard requirements
- [ ] Documentation updated comprehensively

✅ **Technical Requirements**:
- [ ] All models trained with same hyperparameters as CV
- [ ] Same preprocessing/feature generation as training
- [ ] Proper inverse transforms applied (log, Z-score)
- [ ] Reproducible with fixed random seeds

## Expected Outputs

1. **Test Features**:
   - `results/baseline_features_test.pkl` (molecular features for baseline)

2. **Trained Models**:
   - `results/baseline_models/model_LogD.pkl`
   - `results/baseline_models/model_KSOL.pkl`
   - ... (9 models total)
   - `results/gnn_model_full.pt`

3. **Test Predictions**:
   - `results/baseline_test_predictions.csv` (2,282 × 11)
   - `results/gnn_test_predictions.csv` (2,282 × 11)
   - `results/ensemble_test_predictions.csv` (2,282 × 11)
   - `results/test_predictions.csv` (FINAL SUBMISSION FILE)

4. **Documentation**:
   - Updated `README.md`
   - Updated `manifest.json`
   - Training logs in `logs/`

## Risk Mitigation

- **Risk**: Models may perform differently on full training set vs CV
  - **Mitigation**: Log training metrics to compare with CV performance

- **Risk**: Test set may have molecules that fail featurization
  - **Mitigation**: Implement robust error handling, use fallback predictions

- **Risk**: Predictions may be out of reasonable range
  - **Mitigation**: Apply sanity checks, clip to reasonable bounds if needed

## Timeline Estimate

- Steps 1-2: ~5 minutes (data prep)
- Steps 3-4: ~3 minutes (model training)
- Steps 5-7: ~2 minutes (predictions)
- Steps 8-10: ~2 minutes (submission prep and docs)
- **Total**: ~12 minutes

## Notes

- This addresses the critical blocking issue identified by review_agent
- Uses established, validated approaches from successful CV experiments
- Ensemble approach expected to provide best performance based on literature
- All predictions will be on the original scale (not normalized/log-transformed)
