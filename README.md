![Python CI](https://github.com/kou-sato-ds/Kaggle-Playground-S6E2-Heart-Disease-Prediction---9-iterations-of-Trial-Error/actions/workflows/python-ci.yml/badge.svg)

# 🫀 Kaggle Heart Disease Prediction: 9 Iterations of Trial & Error

Kaggle Playground Series (S6E2) における心臓病予測プロジェクト。
単一のモデル構築に留まらず、**9段階にわたる特徴量エンジニアリングとアンサンブル手法の深化**を記録した、実践的なモデル開発のポートフォリオです。

---

## 📈 モデル進化のロードマップ (Development Roadmap)

```mermaid
graph TD
    v1[<b>v1-v3: Baseline</b><br/>LightGBM / Simple Encoding] --> v5[<b>v5-v7: Interaction</b><br/>Category Interax / Target Encoding]
    v5 --> v9[<b>v9: Final Master</b><br/>XGB+LGBM Ensemble / 5-Seed Averaging]

    style v9 fill:#f1c40f,stroke:#333,color:#333