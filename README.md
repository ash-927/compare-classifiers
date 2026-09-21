# Comparing Classifiers: Bank Marketing Campaign

Practical Application III — a comparison of Logistic Regression, K-Nearest Neighbors, Decision Trees, and Support Vector Machines on the task of predicting whether a bank client will subscribe to a term deposit, based on a real-world telemarketing dataset.

## Business Objective

The dataset comes from a Portuguese banking institution's phone-based marketing campaigns. The goal is to build a model that predicts whether a client will subscribe to a term deposit, so the bank can focus its calling effort on the clients most likely to convert — improving campaign efficiency and reducing wasted outreach.

## Data

- **Source**: [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **File used**: `data/bank-additional-full.csv` (41,188 rows, 20 features + target)
- **Target**: `y` — has the client subscribed a term deposit? (`yes`/`no`)
- **Class balance**: ~89% "no" / ~11% "yes" (imbalanced)
- **Background**: [CRISP-DM-BANK.pdf](CRISP-DM-BANK.pdf) — the paper accompanying the dataset (Moro, Cortez & Rita, 2014), describing the campaigns and feature engineering context.

## Approach

Work is in [classifiers.ipynb](classifiers.ipynb), following the CRISP-DM process:

1. **Data understanding & cleaning** — inspect features, identify placeholder "unknown" values and the `pdays` sentinel (999).
2. **Feature engineering** — encode the target and one-hot encode categorical features.
3. **Train/test split** — 80/20, stratified on the target.
4. **Baseline** — majority-class accuracy (~89%) as the bar to beat.
5. **Model comparison** — fit Logistic Regression, KNN, Decision Tree, and SVM with default settings; compare fit time, train/test accuracy.
6. **Improving the model**:
   - Hyperparameter tuning via `GridSearchCV` for each model.
   - Switching the evaluation metric from accuracy to F1-score, precision, recall, and ROC-AUC, since accuracy is dominated by the majority class on this imbalanced dataset.

## Requirements

- Python 3
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `jupyter`

## Running

```bash
jupyter notebook classifiers.ipynb
```
