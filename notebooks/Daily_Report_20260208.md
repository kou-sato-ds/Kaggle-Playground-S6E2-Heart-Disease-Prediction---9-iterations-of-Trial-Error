Data Science Learning Report (2026-02-08)
1. Project Overview
Participated in the Kaggle Playground Series S6E2 (Heart Disease Prediction). Focused on improving implementation skills and hypothesis testing through 9 iterative stages (V1-V9).

2. Key Achievements
Final Score (Public Score): 0.95337 (Personal Best)

Methodology: 5-Seed Averaging, Stratified 5-Fold CV, Advanced Feature Engineering.

Practice: Completed full "Sakyo" (hand-written code transcription) for Version 9 to deepen understanding of model logic.

3. Iteration Journey (V1 - V9)
V1 - V5 (Foundation): Built a robust validation framework using Stratified K-Fold.

V6 - V7 (Statistical Insights): Introduced BP_diff_Mean (deviation from average blood pressure).

V8 (Domain Knowledge): Applied medical logic (High-risk flags for BP > 140) and log transformations.

V9 (Grand Finale): Pinpointed high-risk profiles using deep categorical interactions (Thallium + Vessels + Sex).

4. Code Deep Dive (Technical Notes)
Feature Engineering
Interaction Features: Created unique profiles by combining columns using astype(str) + "_" + ....

Non-linear Combo (ST_Slope_Combo): Used (df['Slope of ST'] + 1) to prevent zero-multiplication, preserving the weight of ST depression.

Model Training
Stratified K-Fold: Used skf.split(X, y) to ensure each fold has the same ratio of heart disease cases as the original data.

5-Seed Averaging: Implemented oof_preds[val_idx] += model.predict(X_val) / len(seeds) to average 5 different perspectives for maximum stability.

5. Reflections & Next Steps
Impact of "Sakyo": Hand-writing the code made functions like enumerate and split intuitive rather than just syntax.

Visualization: Confirmed that HR_Age_Index significantly contributed to the model via Feature Importance plots.

Next Step: Moving toward Data Scientist Certification (DS検定) and SQL/Pre-processing mastery.
