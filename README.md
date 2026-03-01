Version,変更点,スコア (RMSE),考察
v1,Baseline (LightGBM),15.4,まずは基本モデルで構築
v3,Target Encoding追加,14.2,カテゴリ変数の処理で改善
v9,Ensemble (XGB+LGB),13.1,アンサンブルが有効だった

# Kaggle-Playground-S6E2-Heart-Disease-Prediction---9-iterations-of-Trial-Error
A comprehensive record of 9 iterative feature engineering stages for heart disease prediction in the Kaggle Playground Series (S6E2), featuring advanced categorical interactions and 5-seed ensemble techniques.

Kaggle Heart Disease Prediction (Playground Series S6E2)
This repository documents my journey of 9 iterative model improvements, focusing on practical implementation skills and feature hypothesis testing.

Key Outcomes
Best Public Score: 0.95337
Methodology: 5-Seed Averaging, Stratified 5-Fold CV, and Advanced Feature Engineering.
Evolutionary Journey (V1 - V9)
Foundation (V1-V5): Established a robust cross-validation framework using LightGBM.
Statistical Insights (V6-V7): Engineered features like BP_diff_Mean to detect anomalies relative to global averages.
Medical Logic (V8): Applied medical domain knowledge (e.g., Blood Pressure thresholds > 140) and data normalization via Log Transformations.
The Grand Finale (V9): Implemented multi-categorical interactions (Thallium + Vessels + Sex) to pinpoint high-risk patient profiles.
Learning Approach
Dedicated to "Sakyo" (the practice of hand-writing code) to master the underlying logic of data processing and model tuning. Each version represents a step toward becoming a more capable Data Scientist.


## Visualizations

### Feature Importance (Insights from V8/V9)
Analyzing the impact of engineered features like `HR_Age_Index` and `MaxHR_per_Age`.
![Feature Importance](image_a6a500.png)

### Final Leaderboard Achievement
Achieved a personal best of 0.95337 after 9 iterative improvements.
<img width="1477" height="195" alt="image" src="https://github.com/user-attachments/assets/5241c693-ab23-49fd-b23a-28d94b59cc0a" />


### Day 2 Update: Breaking the 0.95 Barrier (Iteration 10-14)
- **Problem**: Encountered 0.5 score baseline due to pipeline logic errors.
- **Solution**: Rebuilt a robust "Defensive Programming" pipeline ensuring no `KeyError` and data alignment.
- **Domain Engineering**:
  - Integrated **Rate Pressure Product (RPP)** to capture myocardial oxygen demand.
  - Added **Age_BP_Risk** to weight hypertension risks in elderly profiles.
- **Outcome**: Achieved a personal best of **0.95018** through 5-Seed Averaging and feature synergy.
