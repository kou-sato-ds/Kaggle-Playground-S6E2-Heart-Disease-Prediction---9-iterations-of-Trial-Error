import pandas as pd
import numpy as np

def apply_v9_features(df):
    """
    Kaggle S6E2(心臓病予測)でスコアアップに寄与した
    ドメイン知識に基づく特徴量エンジニアリング
    """
    # 1. カテゴリカル相互作用（性別 × 主要血管数）
    # 特定の組み合わせ（例：女性で血管異常あり）のリスクを強調
    df['sex_ca_interaction'] = df['sex'].astype(str) + "_" + df['ca'].astype(str)

    # 2. 臨床的カットオフに基づく離散化（ビン分割）
    # 連続値としての血圧だけでなく、「高血圧（140以上）」という状態を抽出
    df['is_high_blood_pressure'] = (df['trestbps'] >= 140).astype(int)

    # 3. 特徴量スケーリングの準備（年齢比率）
    # 加齢によるコレステロール値の影響度を算出（ドメイン知識の活用）
    df['chol_per_age'] = df['chol'] / (df['age'] + 1)

    return df