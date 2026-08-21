# 🚀 Credit Card Fraud Detection

Detecting fraudulent credit card transactions in a highly imbalanced dataset.

Dataset:
https://www.kaggle.com/mlg-ulb/creditcardfraud

---

# 🎯 Objective

Identify fraudulent transactions.

It matters because fraud causes billions in losses, but normal transactions outnumber fraud 99.8% to 0.2%, making standard accuracy metrics useless.

- Logistic Regression
- SVM

---

# 📊 Dataset

Credit Card Fraud

## Features

| Time | Seconds elapsed between transaction and first transaction |
| V1-V28 | PCA reduced features (anonymized) |
| Amount | Transaction amount |

## Target

```text
Target Classes (Classification)
```

Class 0 = Normal

Class 1 = Fraud

---

# 🔍 Exploratory Data Analysis

Performed:

- Target distribution
- Distribution analysis
- Imbalance severity check

### Key Observations

- The dataset is massively imbalanced (99.8% Class 0).
- Fraudulent transactions tend to have smaller, very specific amounts.
- V-features are already PCA transformed.

---

# 🛠 Feature Engineering

Describe every engineered feature.

## Amount Log-Transform

```python
df['log_amount'] = np.log1p(df['Amount'])
```

Purpose:

The 'Amount' feature is highly right-skewed. Taking the log normalizes it, allowing linear models to process it effectively without being thrown off by massive purchases.

---

## Time of Day

Purpose:

Converted the continuous 'Time' variable (seconds) into a cyclic 'Time of Day' (hours), as fraud is often highly correlated with specific late-night hours.

---

# ⚙️ Data Preprocessing

## Missing Values

No missing values present.

---

## Encoding

No encoding needed as V1-V28 are continuous numerical vectors.

---

## Feature Scaling

RobustScaler applied to 'Amount' and 'Time' because it is robust to extreme outliers, which are common in fraud data.

---

## Train-Test Split

```text
80% Training

20% Testing
```

---

# 🤖 Models Implemented

- Logistic Regression
- SVM (Calibrated)

Chosen to benchmark how baseline algorithms handle severe class imbalances.

---

# 📈 Evaluation Metrics

Used:

- Precision
- Recall
- F1-Score
- ROC-AUC

---

# 📊 Results

## Performance Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Logistic Regression | - | 0.0547 | 0.8737 | 0.1030 | 0.9844 |
| Linear SVM (Calibrated) | - | 0.8548 | 0.5579 | 0.6752 | 0.9838 |

---

# 📉 Visualizations

Included:

- Precision-Recall Curve
- Confusion Matrix Heatmap
- ROC Curve

---

# 📌 Interpretation

Discuss:

- High accuracy (99%) is a lie. The model predicts 'Normal' for everything.
- By tuning thresholds or using class_weight='balanced', Logistic Regression catches 87% of fraud (High Recall), but sacrifices Precision (many false alarms). Linear SVM favors precision.

---

# 🧠 Key Learnings

- Accuracy is a deceptive metric on imbalanced datasets.
- Precision-Recall curves are far superior to ROC curves when the positive class is extremely rare.
- Business tradeoff: Is a blocked legitimate transaction worse than a missed fraudulent one?

---

# 🛠 Tech Stack

- Python
- Pandas
- Scikit-Learn
- Imbalanced-Learn

---

# 🚀 Future Improvements

- SMOTE (Synthetic Minority Over-sampling)
- XGBoost or Isolation Forests

---

# Author

Built as part of my **100 Days of Machine Learning**