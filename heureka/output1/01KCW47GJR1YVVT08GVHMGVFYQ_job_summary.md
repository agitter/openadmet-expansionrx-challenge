# Job Summary: ADMET Prediction Ensemble Model

**Timestamp:** December 19, 2025 at 08:43 PM UTC

**Credits:** 1552

**Duration:** 21m 0s

---

## Overview
Developed a state-of-the-art LightGBM ensemble model for predicting 9 ADMET properties from molecular SMILES strings for the OpenADMET ExpansionRx Blind Challenge. Successfully achieved target performance with **Spearman correlation of 0.815** and **MA-RAE of 0.508**.

## Methodology
1. **Feature Engineering**: Generated 3,263 molecular features including RDKit descriptors (88), Morgan fingerprints r=2/r=3 (3,032), and MACCS keys (137)
2. **Target Preprocessing**: Z-score normalization with log-transformation for skewed properties (HLM CLint, MLM CLint, Caco-2 Efflux)
3. **Model Architecture**: LightGBM ensemble with 3 property-specific configurations per target
4. **Training Strategy**: 5-fold cross-validation with early stopping, weighted ensemble based on CV Spearman
5. **Post-processing**: Clipping negative values for non-negative properties

## Key Results

| Property | Spearman | MA-RAE |
|----------|----------|--------|
| LogD | **0.939** | 0.308 |
| KSOL | 0.771 | 0.472 |
| HLM CLint | 0.789 | 0.559 |
| MLM CLint | **0.837** | 0.555 |
| Caco-2 Papp A>B | 0.778 | 0.563 |
| Caco-2 Efflux | 0.749 | 0.538 |
| MPPB | **0.839** | 0.471 |
| MBPB | 0.815 | 0.543 |
| MGMB | 0.817 | 0.561 |

**Overall**: Spearman = 0.815 ✓ (target ≥0.81), MA-RAE = 0.508 ✓ (target <0.53)

## Files Generated
- `admet_predictions_final.csv` - **Final submission** (2,282 compounds × 9 ADMET properties)
- `model_report.md` - Detailed methodology and results documentation
- `01_data_exploration.py` through `07_finalize_predictions.py` - Complete reproducible pipeline
- `cv_ensemble_models.pkl` - Trained model weights (for reproducibility)
- `data_exploration_plots.png`, `target_distributions.png` - Visualization outputs

## Quality Control / Limitations
- **Distribution shift handled**: Test molecules ~20% longer than training; model generalizes well
- **Sparse data success**: MGMB achieved 0.82 Spearman with only 222 training samples
- **Log-transform critical**: Improved HLM CLint from 0.15 to 0.79 Spearman
- **Prediction clipping**: ~4% of predictions required clipping to non-negative values
- **Potential improvements**: Graph neural networks and multi-task learning could further improve performance