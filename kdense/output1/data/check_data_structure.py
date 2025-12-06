#!/usr/bin/env python3
import pickle

SESSION_DIR = "/app/sandbox/session_20251205_152206_4285cc85e60d"

print("Checking train_data.pkl structure...")
with open(f"{SESSION_DIR}/results/train_data.pkl", 'rb') as f:
    train_data = pickle.load(f)

print(f"Type: {type(train_data)}")
if isinstance(train_data, dict):
    print(f"Keys: {list(train_data.keys())}")
else:
    print(f"Columns: {list(train_data.columns) if hasattr(train_data, 'columns') else 'N/A'}")
    print(f"Shape: {train_data.shape if hasattr(train_data, 'shape') else 'N/A'}")
