# 🚀 Employee Attrition Prediction

Predicting employee turnover using Random Forests and interpreting results with SHAP.

Dataset:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

---

# 🎯 Objective

Predict if an employee will leave the company (Attrition = Yes).

It matters because recruiting and training new employees is incredibly expensive, and HR needs proactive retention strategies.

- Random Forest Classifier

---

# 📊 Dataset

IBM HR Analytics

## Features

| MonthlyIncome | Salary |
| OverTime | Works overtime (Yes/No) |
| DistanceFromHome | Commute distance |
| YearsAtCompany | Tenure |

## Target

```text
Target Classes (Classification)
```

Class 0 = No (Stayed)

Class 1 = Yes (Left)

---

# 🔍 Exploratory Data Analysis

Performed:

- Imbalance check
- Salary distribution by attrition
- Job role analysis

### Key Observations

- Attrition is highly imbalanced (most employees stay).
- Employees working OverTime leave at a massively higher rate.
- Lower monthly income correlates strongly with leaving.

---

# 🛠 Feature Engineering

Describe every engineered feature.

## OverTime Binary

```python
df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
```

Purpose:

Converted text to a direct binary flag. Since it's highly predictive, isolating it as a 1/0 feature gives the ensemble model a clean splitting axis.

---

## Distance-to-Work Bins

Purpose:

Binned commute distances into 'Short', 'Medium', and 'Long'. Long commutes lead to burnout, and binning smooths out the noise of exact mile counts.

---

# ⚙️ Data Preprocessing

## Missing Values

No missing values in this IBM dataset.

---

## Encoding

One-Hot Encoding for JobRole, MaritalStatus, and BusinessTravel.

---

## Feature Scaling

Not required for Random Forests.

---

## Train-Test Split

```text
80% Training

20% Testing
```

---

# 🤖 Models Implemented

- Decision Tree (Baseline)
- Random Forest Classifier

Chosen to prove that an ensemble of randomized trees (Bagging) vastly outperforms a single tree by reducing variance.

---

# 📈 Evaluation Metrics

Used:

- Accuracy
- ROC-AUC
- Feature Importance

---

# 📊 Results

## Performance Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Decision Tree | 0.7245 | 0.7997 | 0.7245 | 0.7522 | 0.6825 |
| Random Forest | 0.7925 | 0.8187 | 0.7925 | 0.8035 | 0.7552 |

---

# 📉 Visualizations

Included:

- Feature Importance Bar Chart
- SHAP Summary Plot
- SHAP Waterfall Plot

---

# 📌 Interpretation

Discuss:

- Random Forest drastically improved ROC-AUC (0.75) over the single Decision Tree (0.68).
- OverTime, MonthlyIncome, and Age were the top 3 drivers of attrition.
- SHAP plots proved that working Overtime strictly pushed the prediction toward 'Leaving', while high Income pushed it toward 'Staying'.

---

# 🧠 Key Learnings

- Bagging (Bootstrap Aggregating) prevents overfitting by building diverse trees on random subsets of data and features.
- Feature Importance tells you *what* matters, but SHAP tells you *how* it matters (direction and magnitude).

---

# 🛠 Tech Stack

- Python
- Scikit-Learn
- SHAP
- Matplotlib

---

# 🚀 Future Improvements

- XGBoost for performance
- SMOTE for imbalance

---

# Author

Built as part of my **100 Days of Machine Learning**