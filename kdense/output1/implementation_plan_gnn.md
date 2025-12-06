# Implementation Plan: Multi-Task Graph Neural Network for ADMET Prediction

**Date**: 2025-12-05
**Task**: Implement, train, and evaluate a multi-task GNN for predicting 9 ADMET properties
**Context**: Baseline LightGBM model complete (Spearman: 0.7682). Now implementing advanced GNN model.

---

## Success Criteria

1. ✅ GNN environment successfully configured (`torch`, `torch_geometric`, `chemprop`)
2. ✅ Data preparation script creates normalized training file and saves scaler
3. ✅ Chemprop training completes without errors using scaffold-based CV
4. ✅ Final `results/gnn_cv_scores.csv` created with performance metrics
5. ✅ Mean Spearman correlation logged and compared to baseline (0.7682)
6. ✅ Expected: GNN performance competitive with or exceeding baseline

---

## Implementation Steps

### Step 0: Pre-Execution Inspection ✓
- [x] Survey session directory structure
- [x] Identify completed baseline work (scripts 01-06, baseline_cv_scores.csv)
- [x] Confirm training data available (results/train_data.pkl)
- [x] Review literature requirements (Z-score normalization critical)

### Step 1: Environment Setup & Package Installation
**Objective**: Install PyTorch, PyTorch Geometric, and Chemprop libraries

**Actions**:
1. Verify UV installation and version
2. Add core dependencies to pyproject.toml:
   - `torch>=2.1.0` - Deep learning framework
   - `torch_geometric>=2.4.0` - GNN operations
   - `chemprop>=2.0.0` - High-level molecular GNN library
3. Run `uv sync` to install all dependencies
4. Verify installation by importing packages in test script
5. Log package versions for reproducibility

**Outputs**:
- Updated `pyproject.toml`
- Installation verification log

**Success Criteria**:
- All packages install without errors
- Import tests pass successfully
- Versions logged

---

### Step 2: GNN Data Preparation Script
**Script**: `workflow/07_prepare_gnn_data.py`

**Objective**: Prepare normalized training data for Chemprop with critical Z-score normalization

**Critical Requirements from Literature Review**:
- **Z-score normalization is ESSENTIAL** for multi-task GNN success
- Without normalization: gradient imbalance, KSOL dominates training
- With normalization: mean Spearman 0.8175+ across all endpoints

**Actions**:
1. Load training data from `results/train_data.pkl`
2. Extract SMILES and 9 target properties
3. **CRITICAL**: Implement Z-score normalization:
   - Calculate mean and std for each of 9 properties (on available data only)
   - Apply normalization: `z = (x - mean) / std`
   - Handle missing values appropriately (don't include in mean/std calculation)
4. Save fitted scaler object to `results/target_scaler.pkl`:
   - Store mean and std for each property
   - Essential for inverse transform during test prediction
5. Create CSV file formatted for Chemprop:
   - Column 1: SMILES strings
   - Columns 2-10: 9 normalized target values
   - Missing values: Keep as NaN (Chemprop handles internally)
6. Save as `workflow/gnn_train_data_normalized.csv`
7. Log normalization statistics (mean, std for each property)
8. Validate output: check for infinite values, verify dimensions

**Outputs**:
- `workflow/gnn_train_data_normalized.csv` - Training data for Chemprop
- `results/target_scaler.pkl` - Normalization parameters (mean, std for 9 properties)
- Normalization statistics log

**Success Criteria**:
- CSV file created with correct dimensions (5,326 rows × 10 columns)
- Scaler object saved successfully
- No infinite or invalid normalized values
- Missing values preserved correctly

---

### Step 3: GNN Training Script
**Script**: `workflow/08_train_gnn_model.py`

**Objective**: Train multi-task GNN using Chemprop with scaffold-based cross-validation

**Chemprop Configuration**:
- **Task type**: Multi-task regression (9 properties)
- **Split type**: `scaffold_balanced` (fair comparison with LightGBM baseline)
- **Evaluation metric**: Spearman correlation
- **Cross-validation**: Let Chemprop handle CV loops (likely 5-fold)
- **Output directory**: `results/chemprop_gnn_model/`

**Implementation Options**:

**Option A: Python API** (Recommended for integration)
```python
from chemprop import train
from chemprop.args import TrainArgs

args = TrainArgs()
args.data_path = 'workflow/gnn_train_data_normalized.csv'
args.dataset_type = 'regression'
args.split_type = 'scaffold_balanced'
args.num_folds = 5
args.metric = 'spearman'
args.save_dir = 'results/chemprop_gnn_model/'
args.epochs = 30
args.batch_size = 50

train(args)
```

**Option B: Command-Line Interface**
```bash
chemprop_train \
    --data_path workflow/gnn_train_data_normalized.csv \
    --dataset_type regression \
    --split_type scaffold_balanced \
    --num_folds 5 \
    --metric spearman \
    --save_dir results/chemprop_gnn_model/ \
    --epochs 30 \
    --batch_size 50 \
    --quiet
```

**Progress Monitoring**:
- Print epoch progress every 5 epochs
- Log fold completion after each CV fold
- Estimate total training time

**Outputs**:
- `results/chemprop_gnn_model/` directory containing:
  - Trained model checkpoints (5 models for 5 folds)
  - Cross-validation predictions CSV
  - Training logs and metrics
  - Configuration file

**Success Criteria**:
- Training completes without errors
- All 5 CV folds complete successfully
- Model checkpoints saved
- CV predictions file generated

---

### Step 4: Results Analysis and Aggregation
**Script**: `workflow/09_analyze_gnn_results.py`

**Objective**: Parse Chemprop outputs and create standardized results file

**Actions**:
1. Locate Chemprop CV predictions file in `results/chemprop_gnn_model/`
2. Parse predictions for all 9 properties across all folds
3. Calculate performance metrics for each property:
   - **Spearman correlation**: Primary metric (ranking performance)
   - **MA-RAE**: Mean absolute relative to average error (baseline comparison)
   - Calculate mean and std across folds
4. Denormalize predictions using `results/target_scaler.pkl`:
   - Load scaler parameters (mean, std)
   - Inverse transform: `x = z * std + mean`
5. Calculate metrics on denormalized predictions
6. Create standardized output matching baseline format
7. Save to `results/gnn_cv_scores.csv` with columns:
   - property, n_samples, spearman_mean, spearman_std, ma_rae_mean, ma_rae_std
8. Calculate overall mean Spearman across all 9 properties
9. Compare with baseline performance (0.7682)
10. Generate performance comparison report

**Outputs**:
- `results/gnn_cv_scores.csv` - Aggregated performance metrics
- `results/gnn_performance_comparison.txt` - Baseline vs GNN comparison
- Performance visualization (optional): bar chart comparing baseline vs GNN

**Success Criteria**:
- CSV file created with metrics for all 9 properties
- Mean Spearman calculated and logged
- Comparison with baseline documented
- No calculation errors or missing metrics

---

### Step 5: Documentation and Summary
**Objective**: Update documentation with GNN implementation details

**Actions**:
1. Update `README.md` with:
   - GNN implementation section (Step 3)
   - New dependencies (torch, torch_geometric, chemprop)
   - GNN architecture details
   - Performance comparison table (baseline vs GNN)
   - Files generated
2. Update `manifest.json` with new outputs
3. Document key findings:
   - Which properties improved vs baseline
   - Overall performance gain/loss
   - Training time and computational requirements
4. Note any issues encountered and resolutions

**Outputs**:
- Updated `README.md`
- Updated `manifest.json`
- Documentation of findings

---

## Expected Outcomes

### Performance Expectations (from Literature)
- **Baseline LightGBM**: Mean Spearman 0.7682 ✓ (achieved)
- **Multi-task GNN (with normalization)**: Mean Spearman 0.82-0.85
- **Expected improvement**: +5-10% relative gain
- **Largest gains expected on**:
  - Sparse endpoints (Tier 4: MPPB, MBPB, MGMB)
  - Properties benefiting from information sharing

### Computational Requirements
- Training time: 10-30 minutes (depending on epochs and hardware)
- Memory: 2-4 GB RAM
- GPU: Optional but recommended for faster training

### Key Deliverables
1. ✅ `workflow/gnn_train_data_normalized.csv` - Normalized training data
2. ✅ `results/target_scaler.pkl` - Normalization parameters
3. ✅ `results/chemprop_gnn_model/` - Trained models and outputs
4. ✅ `results/gnn_cv_scores.csv` - Performance metrics
5. ✅ `results/gnn_performance_comparison.txt` - Analysis report
6. ✅ Updated `README.md` and documentation

---

## Risk Mitigation

### Potential Issues and Solutions

1. **Chemprop version compatibility**
   - Risk: API changes between versions
   - Solution: Check Chemprop version, adapt code accordingly
   - Fallback: Use CLI interface if Python API has issues

2. **PyTorch Geometric installation**
   - Risk: Complex dependencies, platform-specific builds
   - Solution: Use UV to handle dependency resolution
   - Fallback: Use pre-built wheels if source build fails

3. **Memory issues with large graphs**
   - Risk: GPU/CPU memory overflow
   - Solution: Reduce batch size, use CPU if needed

4. **Long training time**
   - Risk: Process timeout
   - Solution: Add frequent progress prints (every epoch)
   - Reduce epochs if necessary (minimum 20 for convergence)

5. **Poor GNN performance**
   - Risk: GNN underperforms baseline
   - Solution: Expected - will address in ensemble (Step 4)
   - Document findings for future hybrid model

---

## Technical Notes

### Z-Score Normalization Implementation
```python
# Critical: Calculate on non-missing values only
mean = df[property].dropna().mean()
std = df[property].dropna().std()
df[property + '_normalized'] = (df[property] - mean) / std
```

### Scaffold-Based Splitting
- Chemprop handles internally with `split_type='scaffold_balanced'`
- Ensures molecules with same Murcko scaffold stay in same fold
- Fair comparison with baseline LightGBM

### Masked Loss Handling
- Chemprop automatically masks missing values in multi-task learning
- No explicit loss masking needed in configuration

---

## References

1. **Kosmos AI Report**: `results/01_initial_review.md`
   - Z-score normalization requirement (Critical Discovery 2)
   - Multi-task GNN performance: Spearman 0.8175-0.835

2. **MacDermott-Opeskin 2025**: `converted_md/MacDermottOpeskin2025.pdf.md`
   - ASAP-Polaris-OpenADMET blind challenge
   - Top methods: Multi-task GNNs with pre-training

3. **Baseline Results**: `results/baseline_cv_scores.csv`
   - LightGBM performance: Mean Spearman 0.7682
   - Property-specific benchmarks

---

**Plan Status**: Ready for execution
**Estimated Time**: 45-60 minutes (including training)
**Dependencies**: UV, PyTorch, PyTorch Geometric, Chemprop
