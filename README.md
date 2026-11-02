# Industrial Predictive Maintenance (AI4I 2020)

Predicting machine failure from sensor readings, with a focus on **minimizing false alarms** (False Discovery Rate). In a real factory, a false alarm halts production unnecessarily, so precision matters more than raw accuracy.

Part of the Elevvo Machine Learning Internship (Task 9).

## Dataset
AI4I 2020 Predictive Maintenance Dataset: 10,000 rows, 6 sensor features, 3.39% failures (highly imbalanced).

## Approach
1. **Data cleaning:** dropped identifier columns (`UDI`, `Product ID`), renamed columns, encoded `type` (L/M/H as 0/1/2).
2. **Data leakage prevention:** removed the five failure-type columns (TWF, HDF, PWF, OSF, RNF) from the inputs, since they reveal the answer and are not known before a failure occurs.
3. **Feature engineering:** `temp_diff` (process temp - air temp), `power` (torque x angular speed), `wear_torque` (tool wear x torque).
4. **Stratified 80/20 train/test split** to preserve the failure rate.
5. **Models:** Logistic Regression (baseline), Random Forest, XGBoost, with class imbalance handled (`class_weight` / `scale_pos_weight`).
6. **Threshold tuning:** the decision threshold was chosen on the training set only, using 5-fold cross-validated probabilities (target precision >= 0.95), then evaluated once on the test set.
7. **Cross-validation** on the full dataset to check stability.

## Results (test set: 2,000 rows, 68 failures)
| Model | Precision | Recall | FDR |
|---|---|---|---|
| Logistic Regression | 0.14 | 0.82 | 86% |
| XGBoost (threshold 0.5) | 0.74 | 0.84 | 26% |
| Random Forest, 6 features | 0.90 | 0.54 | 9.8% |
| Random Forest, 9 features | 0.96 | 0.72 | 3.9% |
| **Random Forest, 9 features + tuned threshold (0.43)** | **0.96** | **0.81** | **3.5%** |

5-fold cross-validation on the full data (339 failures, threshold 0.5): precision 0.972 +/- 0.023, recall 0.791 +/- 0.030, F1 0.871 +/- 0.015 (FDR about 2.8%).

FDR = FP / (TP + FP).

## Key findings
- Accuracy is misleading here: a model that always predicts "no failure" scores 96.6%.
- Engineered features raised recall from 0.54 to 0.72 at similar precision. They account for about 39% of the Random Forest feature importance.
- `temp_diff` is more informative than air or process temperature alone.
- Torque and tool wear are the strongest linear indicators of failure (higher on average for failed machines). Rotational speed has a weak linear correlation but the highest Random Forest importance, which suggests a non-linear relationship.
- XGBoost with a large `scale_pos_weight` found more failures but produced many more false alarms (FDR 26%), which conflicts with the goal of this task.
- Threshold tuning (0.43) kept precision at 0.96 with recall 0.81 on the test split. Cross-validation shows recall of about 0.79 even at 0.5, so part of the gain on a single split is within noise.

## Limitations
The test split contains only 68 failures, so small differences between models are within noise. Cross-validation on the full data confirms the Random Forest is stable. Correlation does not imply causation, so "indicator" means "most associated with failure", not "cause".

## Tools
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost.

## Files
- `industrial-predictive-maintenance.ipynb`: full analysis (code and outputs)
