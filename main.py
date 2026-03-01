"""
Heart Disease Prediction - V9 (Grand Finale: Category Interaction & Multi-Seed)
Main Goal: Achieve maximum robustness through deep feature engineering and 5-seed averaging.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

# 1. Data Loading
INPUT_DIR = 'playground-series-s6e2'
train = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))

# 2. Feature Engineering (Preprocess)
def preprocess(df):
    df = df.copy()
    
    # Original Categorical Columns
    original_cats = ['Sex', 'Chest pain type', 'EKG results', 'Slope of ST', 'Thallium', 'Exercise angina']
    
    # [V9 New] High-Risk Category Combination
    # Combining Thallium, Vessels, and Sex to identify high-risk profiles
    df['Thal_Vessel_Sex'] = df['Thallium'].astype(str) + "_" + \
                            df['Number of vessels fluro'].astype(str) + "_" + \
                            df['Sex'].astype(str)
    
    # Label Encoding for Categorical Data
    cat_cols = original_cats + ['Thal_Vessel_Sex']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # --- Engineered Features (Derived from V1-V8 Trials) ---
    df['Vessel_Thal_Risk'] = df['Number of vessels fluro'] * df['Thallium']
    df['Thal_Age'] = df['Thallium'] * df['Age']
    df['MaxHR_per_Age'] = df['Max HR'] / (df['Age'] + 1e-5)
    df['Vessel_ST_Interaction'] = df['Number of vessels fluro'] * df['ST depression']
    df['Thal_Pain_Risk'] = df['Thallium'] * df['Chest pain type']
    df['BP_diff_Mean'] = df['BP'] - df['BP'].mean()
    df['Chol_diff_Mean'] = df['Cholesterol'] - df['Cholesterol'].mean()
    df['HR_Age_Index'] = df['Max HR'] * (100 - df['Age'])
    
    # Non-linear Combination of ST Depression and Slope
    df['ST_Slope_Combo'] = df['ST depression'] * (df['Slope of ST'] + 1)

    # Target Mapping
    if 'Heart Disease' in df.columns:
        df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    return df

train = preprocess(train)
test = preprocess(test)

# Define Features and Target
features = [c for c in train.columns if c not in ['id', 'Heart Disease']]
X, y = train[features], train['Heart Disease']

# 3. K-fold & 5-Seed Averaging
seeds = [42, 0, 2026, 777, 123] 
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for seed in seeds:
    print(f"\n--- Training Seed: {seed} ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'learning_rate': 0.015,
        'random_state': seed,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5
    }

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.train(
            params,
            lgb.Dataset(X_tr, label=y_tr),
            num_boost_round=3000,
            valid_sets=[lgb.Dataset(X_val, label=y_val)],
            callbacks=[lgb.early_stopping(200)]
        )

        oof_preds[val_idx] += model.predict(X_val) / len(seeds)
        test_preds += model.predict(test[features]) / (len(seeds) * 5)

# 4. Final Evaluation
score = roc_auc_score(y, oof_preds)
print(f"\n{'*' * 30}\nGRAND FINALE OOF AUC: {score:.5f}\n{'*' * 30}")

# 5. Export Submission
pd.DataFrame({'id': test['id'], 'Heart Disease': test_preds}).to_csv('submission_v9_final.csv', index=False)