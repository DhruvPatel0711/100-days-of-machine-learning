# 🚀 Breast Cancer Detection

Classifying breast cancer tumors as malignant or benign using SVM and Logistic Regression.

Dataset:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

---

# 🎯 Objective

Classify tumor cells based on computed features.

It matters because early and accurate detection in medical diagnostics saves lives, and false negatives are incredibly dangerous.

- Support Vector Machines (SVM)
- Logistic Regression

---

# 📊 Dataset

Breast Cancer Wisconsin

## Features

| radius_mean | Mean of distances from center to points on perimeter |
| texture_mean | Standard deviation of gray-scale values |
| perimeter_mean | Mean size of the core tumor |
| area_mean | Mean area |

## Target

```text
Target Classes (Classification)
```

Class 0 = Malignant

Class 1 = Benign

---

# 🔍 Exploratory Data Analysis

Performed:

- Multicollinearity check
- Target distribution
- Statistical summaries

### Key Observations

- Many features (radius, perimeter, area) are highly correlated with each other.
- The dataset is relatively balanced.
- Tumors with larger radii and perimeters are highly likely to be malignant.

---

# 🛠 Feature Engineering

Describe every engineered feature.

## Feature Scaling

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Purpose:

SVM calculates the geometric margin between data points. If one feature is measured in thousands and another in decimals, the larger feature will completely dominate the boundary.

---

## Multicollinearity Check

Purpose:

Identified highly correlated features (like radius and area) to understand how they might affect Logistic Regression coefficients, though SVM is somewhat immune.

---

# ⚙️ Data Preprocessing

## Missing Values

No missing values present in the built-in sklearn dataset.

---

## Encoding

Target variable was label encoded (0 and 1). No categorical features to encode.

---

## Feature Scaling

StandardScaler applied. Demonstrated metric collapse when scaling was deliberately removed for the SVM.

---

## Train-Test Split

```text
80% Training

20% Testing
```

---

# 🤖 Models Implemented

- Support Vector Machine (Linear & RBF Kernels)
- Logistic Regression

Chosen to compare a margin-maximizing algorithm (SVM) against a probabilistic algorithm (LogReg).

---

# 📈 Evaluation Metrics

Used:

- Accuracy
- F1 Score
- ROC-AUC
- Log Loss

---

# 📊 Results

## Performance Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.9825 | 0.9859 | 0.9859 | 0.9859 | 0.9971 |
| Linear SVM | 0.9561 | 0.9714 | 0.9577 | 0.9645 | 0.9964 |
| RBF SVM | 0.9737 | 0.9722 | 0.9859 | 0.9790 | 0.9974 |

---

# 📉 Visualizations

Included:

- Correlation heatmap
- Confusion Matrix
- ROC Curve

---

# 📌 Interpretation

Discuss:

- SVM completely collapsed without feature scaling (predicting only one class). Once scaled, it achieved near-perfect recall.
- In medical datasets, Recall is prioritized over Precision to ensure no malignant tumors are missed.

---

# 🧠 Key Learnings

- SVM finds the hyperplane that maximizes the margin between classes.
- Feature scaling is 100% non-negotiable for SVM.
- ROC-AUC evaluates performance across all thresholds, not just 0.5.

---

# 🛠 Tech Stack

- Python
- Pandas
- Scikit-Learn
- Seaborn

---

# 🚀 Future Improvements

- PCA for dimensionality reduction
- GridSearch for SVM kernels

---

# Author

Built as part of my **100 Days of Machine Learning**