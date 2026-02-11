#!/usr/bin/env python3
"""
train_models.py

Train 8 logistic regression models (one per island × observation type)
using L2 regularization.

Settings:
    - Logistic Regression with L2 penalty
    - C=1.0 (default regularization strength)
    - No feature scaling
    - No cross-validation or hyperparameter tuning

Usage:
    python train_models.py
"""

import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Paths
TRAIN_DIR = './out/species/train'
MODEL_DIR = './out/models'

ISLANDS = ['san_clemente', 'santa_catalina', 'santa_cruz', 'santa_rosa']
LABEL_TYPES = ['human', 'herb']


def main():
    print("=" * 60)
    print("TRAINING LOGISTIC REGRESSION MODELS")
    print("=" * 60)
    print("Settings:")
    print("  - Regularization: L2")
    print("  - C: 1.0")
    print("  - Feature scaling: None")
    print("  - Hyperparameter tuning: None")
    print()
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    summary = []
    
    for island in ISLANDS:
        for label_type in LABEL_TYPES:
            print(f"\n--- {island} / {label_type} ---")
            
            # Load training data
            filename = f'quercus_tomentella_{island}_{label_type}_train.parquet'
            filepath = os.path.join(TRAIN_DIR, filename)
            
            df = pd.read_parquet(filepath)
            
            # Extract features (128 bands) and labels
            band_cols = [c for c in df.columns if c.startswith('band_')]
            X = df[band_cols].values
            y = df['label'].values
            
            n_pos = (y == 1).sum()
            n_neg = (y == 0).sum()
            print(f"  Training samples: {len(y)} ({n_pos} pos, {n_neg} neg)")
            print(f"  Features: {X.shape[1]}")
            
            # Train logistic regression with L2 regularization
            model = LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
            
            model.fit(X, y)
            print(f"  Model trained successfully")
            
            # Save model
            model_filename = f'logreg_{island}_{label_type}.pkl'
            model_path = os.path.join(MODEL_DIR, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'island': island,
                    'label_type': label_type,
                    'feature_names': band_cols,
                    'n_train_pos': n_pos,
                    'n_train_neg': n_neg
                }, f)
            
            print(f"  [SAVED] {model_path}")
            
            summary.append({
                'island': island,
                'label_type': label_type,
                'n_pos': n_pos,
                'n_neg': n_neg,
                'model_file': model_filename
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    print(f"\n[INFO] Models saved to: {MODEL_DIR}")
    print(f"[INFO] Total models trained: {len(summary)}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    
    # ========================================
    # SANITY CHECK
    # ========================================
    print(f"\n{'='*60}")
    print("SANITY CHECK")
    print(f"{'='*60}")
    
    TEST_DIR = './out/species/test'
    
    all_ok = True
    
    for island in ISLANDS:
        for label_type in LABEL_TYPES:
            # Load train pixel IDs
            train_file = f'quercus_tomentella_{island}_{label_type}_train.parquet'
            train_df = pd.read_parquet(os.path.join(TRAIN_DIR, train_file))
            train_pixels = set(train_df['pixel_id'].unique())
            
            # Load test pixel IDs
            test_file = f'quercus_tomentella_{island}_{label_type}_test.parquet'
            test_df = pd.read_parquet(os.path.join(TEST_DIR, test_file))
            test_pixels = set(test_df['pixel_id'].unique())
            
            # Check overlap
            overlap = train_pixels & test_pixels
            
            if len(overlap) > 0:
                print(f"  [ERROR] {island}/{label_type}: {len(overlap)} overlapping pixels!")
                all_ok = False
            else:
                print(f"  [OK] {island}/{label_type}: No train/test pixel overlap")
            
            # Verify model loads correctly
            model_file = f'logreg_{island}_{label_type}.pkl'
            model_path = os.path.join(MODEL_DIR, model_file)
            with open(model_path, 'rb') as f:
                saved = pickle.load(f)
            
            # Check model has expected components
            expected_keys = {'model', 'island', 'label_type', 'feature_names', 'n_train_pos', 'n_train_neg'}
            if set(saved.keys()) != expected_keys:
                print(f"  [WARN] {island}/{label_type}: Model pickle has unexpected keys")
            
            # Verify positive/negative ratio is approximately 10:1
            ratio = saved['n_train_neg'] / saved['n_train_pos'] if saved['n_train_pos'] > 0 else 0
            if abs(ratio - 10.0) > 0.5:
                print(f"  [WARN] {island}/{label_type}: Neg/Pos ratio = {ratio:.1f} (expected ~10)")
    
    print()
    if all_ok:
        print("✓ ALL SANITY CHECKS PASSED")
    else:
        print("✗ SOME CHECKS FAILED - REVIEW OUTPUT ABOVE")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
