# Predicting Credit Card Fraud Using Machine Learning

Individual final project for the **3MTT × WESOnline (DeepTech_Ready) Project Mentorship Program, Cohort 4**.

**Author:** Isa Haruna Aliyu

---

## 1. Project Overview

This project builds and compares machine learning models that classify credit card transactions as **legitimate** or **fraudulent**. Fraud is very rare in the data, so the project focuses on handling class imbalance and on judging models with metrics that suit fraud detection, not accuracy alone.

## 2. Problem Statement

Financial institutions lose money to fraudulent transactions, but fraud is only a tiny share of all transactions. A model that labels everything "legitimate" would look accurate and catch no fraud. The problem is to detect as many frauds as possible without raising too many false alarms.

## 3. Objectives

1. Develop a machine learning model that identifies fraudulent credit card transactions.
2. Compare Logistic Regression, Random Forest, and XGBoost.
3. Address class imbalance (class weighting, random undersampling, random oversampling, SMOTE).
4. Evaluate the models with suitable classification metrics.

## 4. Dataset

- **Source:** Provided by the programme mentor as a ZIP file containing `creditcard.csv`.
- **Size:** 284,807 transactions, 31 columns.
- **Columns:** `Time`, `V1`–`V28` (anonymised numeric features), `Amount`, `Class`.
- **Target:** `Class` (0 = legitimate, 1 = fraud).
- **Missing values:** none.
- **Duplicates:** 1,081 duplicate rows (1,062 legitimate, 19 fraud), removed before splitting.
- **After removing duplicates:** 283,726 rows: 283,253 legitimate and 473 fraud (**0.167% fraud**).
- The V1–V28 features are anonymised, so their individual meanings are not known.

The dataset is **not** included in this repository because of its size (about 144 MB). It was provided by the programme mentor.

## 5. Technologies Used

Python, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, imbalanced-learn, XGBoost, joblib, Google Colab, Git and GitHub.

## 6. Data Preprocessing

1. Removed duplicate rows (before any splitting, so identical rows cannot appear in both train and test sets).
2. Separated features (`X`) from the target (`y`).
3. **Stratified** 80/20 train/test split (`random_state=42`): 226,980 training rows (378 fraud) and 56,746 test rows (95 fraud).
4. Scaled only `Time` and `Amount` with `RobustScaler` (Amount has large outliers). `V1`–`V28` were left unchanged.
5. The scaler was fitted on the training data only, then applied to the test data (to prevent data leakage).
6. No new features were engineered, because the features are anonymised.

## 7. Exploratory Data Analysis

Charts and checks in the notebook: class distribution, duplicates by class, Amount distribution (log scale), Amount by class (box plot), Time by class, correlation of each feature with `Class`, correlation heatmap, and box plots and density plots of the four features most correlated with fraud.

Observations (from this dataset):
- Fraud is 0.173% of the original data (492 of 284,807).
- Fraud has a lower median Amount (9.25) than legitimate transactions (22.00), but a higher mean (122.21 vs 88.29).
- The four features most linked to `Class` were V17, V14, V12 and V10. All have negative correlations, with fraud shifted to lower values.
- Correlations are moderate, so no single feature separates the classes.

## 8. Machine Learning Algorithms

- **Logistic Regression** (`max_iter=1000`)
- **Random Forest** (100 trees, `max_depth=12`)
- **XGBoost** (200 trees, `max_depth=4`, `learning_rate=0.1` before tuning)

XGBoost was tuned with `RandomizedSearchCV` (12 candidates, 3-fold stratified cross-validation on the **training data only**, scored by PR-AUC). Best settings: `n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.85, colsample_bytree=1.0, min_child_weight=1`. Cross-validated PR-AUC on training data: 0.8467.

## 9. Imbalance-Handling Techniques

Resampling was applied to **training data only**.

**Step 1: five approaches compared with Logistic Regression (same test set):**

| Method | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| No handling | 0.8615 | 0.5895 | 0.7000 | 0.9575 | 0.6951 |
| Class weight | 0.0562 | 0.8737 | 0.1057 | 0.9657 | 0.6719 |
| Undersampling | 0.0508 | 0.8737 | 0.0960 | 0.9554 | 0.6004 |
| Oversampling | 0.0569 | 0.8737 | 0.1068 | 0.9660 | 0.6719 |
| SMOTE | 0.0537 | 0.8737 | 0.1012 | 0.9600 | 0.6729 |

Undersampling had the lowest F1 and PR-AUC. The other methods were similar to each other.

**Step 2: three models × three strategies (no handling, class weight, SMOTE):**

| Model \| Strategy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| XGBoost \| SMOTE | 0.3641 | 0.8316 | 0.5064 | 0.9685 | 0.8061 |
| XGBoost \| Class weight | 0.7379 | 0.8000 | 0.7677 | 0.9653 | 0.8043 |
| XGBoost \| No handling | 0.9571 | 0.7053 | 0.8121 | 0.9730 | 0.8032 |
| Random Forest \| SMOTE | 0.6881 | 0.7895 | 0.7353 | 0.9756 | 0.7895 |
| Random Forest \| Class weight | 0.8861 | 0.7368 | 0.8046 | 0.9603 | 0.7883 |
| Random Forest \| No handling | 0.9857 | 0.7263 | 0.8364 | 0.9650 | 0.7875 |
| Logistic Regression \| No handling | 0.8615 | 0.5895 | 0.7000 | 0.9575 | 0.6951 |
| Logistic Regression \| SMOTE | 0.0537 | 0.8737 | 0.1012 | 0.9600 | 0.6729 |
| Logistic Regression \| Class weight | 0.0562 | 0.8737 | 0.1057 | 0.9657 | 0.6719 |

Accuracy was between 0.974 and 0.9995 for every model, so it could not separate them.

## 10. Final Model Evaluation (test set: 56,746 transactions, 95 fraud)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Frauds caught | Frauds missed | False alarms |
|---|---|---|---|---|---|---|---|---|
| **XGBoost (tuned, class weight)** | 0.9367 | 0.7789 | **0.8506** | 0.9775 | **0.8388** | 74 | 21 | 5 |
| Random Forest (no handling) | 0.9857 | 0.7263 | 0.8364 | 0.9650 | 0.7875 | 69 | 26 | 1 |
| Logistic Regression (class weight) | 0.0562 | 0.8737 | 0.1057 | 0.9657 | 0.6719 | 83 | 12 | 1,393 |

**Selected model: tuned XGBoost.** It had the highest F1, PR-AUC and ROC-AUC, caught 74 of 95 frauds, and raised only 5 false alarms. Logistic Regression caught the most fraud but raised 1,393 false alarms. Random Forest was the most precise but caught 5 fewer frauds and trained much more slowly.

The most important feature in the tuned XGBoost model was **V14**, followed by V4, V12, V8 and V10.

Evaluation also included confusion matrices, ROC curves, precision-recall curves, a classification report, a feature-importance chart, and a precision/recall vs threshold chart (all in the notebook).

## 11. How to Install and Run

```bash
# First clone this repository (use the green "Code" button on GitHub to copy its URL), then run:
pip install -r requirements.txt
```

1. Get `creditcard.csv` (ask the programme mentor for the dataset ZIP) and save it on your computer.
2. Open `Credit_Card_Fraud_Isa.ipynb` in Jupyter or Google Colab and run the cells in order. Change the file path in the data-loading cell to where you saved `creditcard.csv`.
3. Random Forest training takes several minutes (about 2.5 to 6.5 minutes per run in Colab).

The notebook trains all models from scratch. Trained model files are not included in this repository.

## 12. Limitations

- The test set has only 95 fraud cases, so a few cases can change the scores noticeably.
- The final model was chosen using the same test set that reported the results, which can make results slightly optimistic. A separate validation set would be better.
- Only 12 tuning candidates were tried, and the best settings were at the edge of the search ranges.
- Random Forest and Logistic Regression were not tuned.
- The decision threshold was left at the default 0.5.
- The V1–V28 features are anonymised, so the model cannot be explained in business terms.
- Results come from one dataset and one split. They may not hold on other data.

## 13. Future Improvements

- Use a separate validation set and repeated cross-validation.
- Tune the decision threshold using the cost of missed fraud vs false alarms.
- Search a wider range of XGBoost settings and tune Random Forest.
- Try other models and calibrated probabilities.
- Build a simple app for demonstrating predictions.

## Author

**Isa Haruna Aliyu** — 3MTT × WESOnline (DeepTech_Ready) Cohort 4
