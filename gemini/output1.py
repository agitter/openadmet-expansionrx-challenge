import pandas as pd
import numpy as np
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 1. Configuration & Helper Functions
# ==========================================

TARGETS = [
    'LogD', 'KSOL', 'HLM CLint', 'MLM CLint', 
    'Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux', 
    'MPPB', 'MBPB', 'MGMB'
]

# targets that benefit from Log-Transform (spanning orders of magnitude)
LOG_TRANSFORM_TARGETS = [
    'KSOL', 'HLM CLint', 'MLM CLint', 
    'Caco-2 Permeability Papp A>B', 'Caco-2 Permeability Efflux'
]

def generate_features(smiles_series):
    """
    Generates a hybrid feature set: Morgan Fingerprints + RDKit Descriptors.
    """
    fp_list = []
    desc_list = []
    
    # Get list of descriptor functions
    descriptor_funcs = [x[1] for x in Descriptors.descList]
    descriptor_names = [x[0] for x in Descriptors.descList]
    
    print(f"Generating features for {len(smiles_series)} compounds...")
    
    for smi in tqdm(smiles_series):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Handle invalid SMILES with a zero-vector placeholder
            fp_list.append(np.zeros(2048))
            desc_list.append(np.zeros(len(descriptor_names)))
            continue
            
        # 1. Morgan Fingerprint (ECFP4 equivalent)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
        fp_list.append(np.array(fp))
        
        # 2. RDKit Descriptors (Physicochemical props)
        descs = []
        for func in descriptor_funcs:
            try:
                val = func(mol)
                # Handle infinity or nan in descriptors
                if not np.isfinite(val):
                    val = 0.0
            except:
                val = 0.0
            descs.append(val)
        desc_list.append(descs)
        
    # Combine into a single matrix
    X_fp = np.array(fp_list)
    X_desc = np.array(desc_list)
    
    # Simple normalization for descriptors (LightGBM is robust, but this helps)
    # We'll just pass raw, LightGBM handles scaling well. 
    # But concatenating them is key.
    return np.hstack([X_fp, X_desc])

# ==========================================
# 2. Load Data
# ==========================================
print("Loading datasets...")
train_df = pd.read_csv("expansion_data_train.csv")
test_df = pd.read_csv("expansion_data_test_blinded.csv")

# ==========================================
# 3. Feature Engineering
# ==========================================
# Combine SMILES to generate features once (handles potential overlap/consistency)
train_smiles = train_df['SMILES'].tolist()
test_smiles = test_df['SMILES'].tolist()

X_train_full = generate_features(train_smiles)
X_test_full = generate_features(test_smiles)

print(f"Feature Matrix Shape: {X_train_full.shape}")

# ==========================================
# 4. Training & Prediction (Multi-Task Ensemble)
# ==========================================

submission_df = test_df[['Molecule Name']].copy()

for target in TARGETS:
    print(f"\nTraining models for Target: {target}")
    
    # Filter training data for this target (drop NaNs)
    mask = train_df[target].notna()
    y_train_raw = train_df.loc[mask, target].values
    X_train_curr = X_train_full[mask]
    
    # 4a. Log-Transform if applicable
    is_log = target in LOG_TRANSFORM_TARGETS
    if is_log:
        # Use log1p to handle zeros safely: y = log(x + 1)
        y_train = np.log1p(y_train_raw)
    else:
        y_train = y_train_raw
        
    # 4b. Ensemble Training (5 Seeds)
    n_seeds = 5
    test_preds_accum = np.zeros(X_test_full.shape[0])
    
    for i in range(n_seeds):
        seed = 42 + i
        
        # LightGBM Parameters (Optimized for small-medium chemical datasets)
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': seed,
            'n_jobs': -1,
            'verbosity': -1
        }
        
        # Simple split for early stopping
        # In a real rigorous setting, we'd use scaffold split, 
        # but random split is sufficient for the "blind" ensemble component here.
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_curr, y_train, test_size=0.1, random_state=seed)
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        # Predict on Test Set
        preds = model.predict(X_test_full)
        test_preds_accum += preds
        
    # Average predictions
    avg_preds = test_preds_accum / n_seeds
    
    # 4c. Inverse Transform if applicable
    if is_log:
        final_preds = np.expm1(avg_preds) # Inverse of log1p
        # Clip to ensure no negative values for physical properties
        final_preds = np.maximum(final_preds, 0)
    else:
        final_preds = avg_preds
        
    # 4d. Post-processing Constraints
    # % Unbound targets should be clipped to [0, 100]
    if 'Unbound' in target or target in ['MPPB', 'MBPB', 'MGMB']:
        final_preds = np.clip(final_preds, 0, 100)
        
    submission_df[target] = final_preds

# ==========================================
# 5. Save Submission
# ==========================================
submission_filename = 'submission_expansion_admet.csv'
submission_df.to_csv(submission_filename, index=False)
print(f"\nProcessing complete! Predictions saved to {submission_filename}")
print("Head of submission:")
print(submission_df.head())
