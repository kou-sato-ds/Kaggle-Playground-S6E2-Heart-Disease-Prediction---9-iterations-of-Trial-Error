# Kaggle Heart Disease Prediction (Playground Series S6E2)

This repository documents my journey of 9 iterative model improvements, focusing on practical implementation skills and feature hypothesis testing.

## Key Outcomes
- **Best Public Score**: 0.95337
- **Methodology**: 5-Seed Averaging, Stratified 5-Fold CV, and Advanced Feature Engineering.

## Evolutionary Journey (V1 - V9)
1. **Foundation (V1-V5)**: Established a robust cross-validation framework using LightGBM.
2. **Statistical Insights (V6-V7)**: Engineered features like `BP_diff_Mean` to detect anomalies relative to global averages.
3. **Medical Logic (V8)**: Applied medical domain knowledge (e.g., Blood Pressure thresholds > 140) and data normalization via Log Transformations.
4. **The Grand Finale (V9)**: Implemented multi-categorical interactions (`Thallium` + `Vessels` + `Sex`) to pinpoint high-risk patient profiles.

## Learning Approach
Dedicated to "Sakyo" (the practice of hand-writing code) to master the underlying logic of data processing and model tuning. Each version represents a step toward becoming a more capable Data Scientist.