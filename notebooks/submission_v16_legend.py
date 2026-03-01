import pandas as pd
import numpy as np
import os
import zipfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. SETUP & DATA LOADING (Automation)
# ==========================================
# GitHubに上げる際、誰の環境でも動くようにパスと解凍処理を整理
ZIP_FILE_PATH = '/content/playground-series-s6e2.zip'
EXTRACT_DIR = 'playground-series-s6e2'

if os.path.exists(ZIP_FILE_PATH):
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print(f"📦 Data extracted to: {EXTRACT_DIR}")

train_path = os.path.join(EXTRACT_DIR, 'train.csv')
test_path = os.path.join(EXTRACT_DIR, 'test.csv')

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# ==========================================
# 2. DOMAIN FEATURE ENGINEERING
# ==========================================
def feature_engineering_legend(df):
    df = df.copy()
    
    # RPP (二重積): 心筋の酸素需要。心臓の物理的な負荷を示す
    if 'Max HR' in df.columns and 'Systolic BP' in df.columns:
        df['RPP'] = df['Max HR'] * df['Systolic BP']
    
    # Age_BP_Risk: 老化×高血圧。加齢による血管リスクの増幅
    if 'Age' in df.columns and 'Systolic BP' in df.columns:
        df['Age_BP_Risk'] = df['Age'] * df['Systolic BP']
        
    # Pulse_Pressure: 血管のしなやかさ（動脈硬化の指標）
    if 'Systolic BP' in df.columns and 'Diastolic BP' in df.columns:
        df['Pulse_Pressure'] = df['Systolic BP'] - df['Diastolic BP']

    # HR_Efficiency: 年齢に対する心拍の余力
    if 'Max HR' in df.columns and 'Age' in df.columns:
        df['HR_Efficiency'] = df['Max HR'] / (df['Age'] + 1)

    return df

train_df = feature_engineering_legend(train)
test_df = feature_engineering_legend(test)

# カテゴリ変数の処理 (Robust LabelEncoding)
cat_cols = train_df.select_dtypes(include=['object']).columns
for col in cat_cols:
    if col == 'Heart Disease': continue
    le = LabelEncoder()
    # TrainとTestを結合してfitさせ、未知のカテゴリによるエラーを完封
    all_data = pd.concat([train_df[col].astype(str), test_df[col].astype(str)])
    le.fit(all_data)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

# データの分離
X = train_df.drop(['id', 'Heart Disease'], axis=1, errors='ignore')
y = train_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
X_test = test_df.drop(['id'], axis=1, errors='ignore')

# 訓練データとテストデータで列の並びを完全に一致させる (実務の鉄則)
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# ==========================================
# 3. CV & MODELING (5-Seed Averaging)
# ==========================================
# 5つのシードを用いて「運」を排除し、0.95超えを盤石にする
seeds = [42, 2026, 777, 123, 999]
test_preds = np.zeros(len(test))
oof_preds = np.zeros(len(train))

print(f"🚀 V16 Final Run: Training on {len(seeds)} Seeds for maximum stability...")

for seed in seeds:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        # 決定木の数と深さを最適化
        model = RandomForestClassifier(
            n_estimators=1000, 
            max_depth=12, 
            min_samples_leaf=5, 
            random_state=seed,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        
        test_preds += model.predict_proba(X_test)[:, 1] / (5 * len(seeds))
        oof_preds[val_idx] += model.predict_proba(X_val)[:, 1] / len(seeds)

print(f"🏆 Final Cross-Validation AUC: {roc_auc_score(y, oof_preds):.5f}")

# ==========================================
# 4. SUBMISSION
# ==========================================
submission = pd.DataFrame({'id': test['id'], 'Heart Disease': test_preds})
submission.to_csv('submission_v16_legend.csv', index=False)
print("🏁 Legend pipeline successfully finished. Ready for GitHub push.")
