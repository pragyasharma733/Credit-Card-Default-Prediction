# Credit Card Default Prediction

Predicting which credit card customers will default, using a highly imbalanced dataset where only **3.3%** of customers default.

## Question
Given a customer's card balance, income and student status, can we flag likely defaulters early enough to act, while keeping the number of false alarms manageable?

## Data
`Default.csv` is the *Default* dataset from *An Introduction to Statistical Learning* (ISLR). It contains 10,000 simulated customers.

| Column | Description |
|---|---|
| `default` | Target: `Yes` / `No` |
| `student` | Whether the customer is a student |
| `balance` | Average monthly credit card balance |
| `income` | Annual income |

## Approach
1. 80/20 train/test split (`random_state=42`).
2. Preprocessing pipeline: `StandardScaler` for numeric features and `OneHotEncoder` for `student`.
3. Compared four classifiers using 10-fold cross-validated decision scores:
   - Logistic Regression
   - Logistic Regression (`class_weight="balanced"`)
   - SGDClassifier
   - SGDClassifier (`class_weight="balanced"`)
4. Evaluated each with ROC curves, ROC AUC and precision/recall-vs-threshold curves.
5. Chose a decision threshold targeting **≥ 80% recall**, since missing a defaulter costs more than a false alarm.

## Results
| Model | CV ROC AUC |
|---|---|
| Logistic Regression | 0.950 |
| SGDClassifier | 0.943 |

The final logistic regression model reaches a **test ROC AUC of 0.94**.

## How to run
```bash
pip install numpy pandas scikit-learn matplotlib
jupyter notebook "Credit Card Default Prediction.ipynb"
```

## Limitations and next steps
- The threshold selection picks the *first* index where recall ≥ 0.8. Recall decreases as the threshold rises, so this selects the lowest threshold (recall = 1.0), which explains the very low precision (~5%) and test accuracy (~41%). Selecting the highest threshold that still gives ≥ 80% recall raises precision to roughly 25%.
- Add a cost-based threshold (e.g. cost of a missed default vs cost of a manual review).
- Report a precision–recall AUC, which is more informative than ROC AUC at a 3% positive rate.
- Interpret the coefficients, e.g. the effect of student status once balance is accounted for.
