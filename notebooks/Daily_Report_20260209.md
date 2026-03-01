# 📅 Data Science Learning Report (2026-02-09)

## 1. Today's Learning (DS Certification Theory)
Focused on understanding evaluation metrics using the Official Reference Book.
- **Topic**: ROC Curve and AUC (Area Under the Curve)
- **Key Takeaway**: 
  - **Y-axis**: True Positive Rate (Sensitivity) - How many actual positives did we catch?
  - **X-axis**: False Positive Rate - How many negatives did we wrongly flag as positive?
- **Insight**: A high AUC (like yesterday's 0.95337) means the model successfully pushes the curve toward the top-left corner.

## 2. Practical Implementation (Mini-Sakyo)
Practiced calculating AUC manually using Python:
```python
from sklearn.metrics import roc_auc_score
y_true = [1, 0, 1, 1, 0]
y_pred = [0.9, 0.1, 0.8, 0.7, 0.2]
score = roc_auc_score(y_true, y_pred)
print(f"AUC Score: {score:.4f}")
