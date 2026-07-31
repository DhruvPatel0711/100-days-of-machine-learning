# 🚀 Medical Cost Predictor

Predicting individual medical costs billed by health insurance using Ridge and Lasso regression.

Dataset:
https://www.kaggle.com/datasets/mirichoi0218/insurance

---

# 🎯 Objective

Predict continuous medical charges based on patient attributes.

It matters because it helps insurance companies calculate premiums and helps individuals understand the financial impact of lifestyle choices like smoking.

- Linear Regression
- Ridge Regression (L2)
- Lasso Regression (L1)

---

# 📊 Dataset

Medical Insurance Dataset

## Features

| age | Age of primary beneficiary |
| bmi | Body mass index |
| children | Number of dependents |
| smoker | Smoking status (yes/no) |

## Target

```text
Target (Regression)
```

charges (Continuous Numerical Value in USD)

---

# 🔍 Exploratory Data Analysis

Performed:

- Missing value analysis
- Distribution analysis
- Correlation analysis

### Key Observations

- Smokers incur drastically higher medical costs than non-smokers.
- BMI has a compounding effect on costs when combined with smoking.
- Charges are right-skewed.

---

# 🛠 Feature Engineering

Describe every engineered feature.

## BMI Categories

```python
df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
```

Purpose:

Medical risks often jump at specific BMI thresholds rather than scaling linearly.

---

## Smoker Interaction Term

Purpose:

Created a combined feature `bmi * smoker`. The danger of high BMI is massively amplified if the patient also smokes, and this interaction term captures that non-linear relationship.

---

# ⚙️ Data Preprocessing

## Missing Values

Checked for missing values; none found in the raw dataset.

---

## Encoding

One-Hot Encoding used for 'sex', 'smoker', and 'region'.

---

## Feature Scaling

StandardScaler applied. Regularization (Ridge/Lasso) heavily penalizes large coefficients, so all features must be on the same scale.

---

## Train-Test Split

```text
80% Training

20% Testing
```

---

# 🤖 Models Implemented

- Linear Regression
- Ridge Regression (L2 Penalty)
- Lasso Regression (L1 Penalty)

Chosen to demonstrate how L1 and L2 regularization prevent overfitting and handle feature selection compared to standard Linear Regression.

---

# 📈 Evaluation Metrics

Used:

- MAE
- RMSE
- R²

---

# 📊 Results

## Performance Comparison

| Model | MAE | RMSE | R² |
|--------|----------|-----------|--------|
| Linear Regression | MAE: 2846.81 | RMSE: 4584.30 | R²: 0.8856 |
| Ridge Regression | MAE: 2846.53 | RMSE: 4583.43 | R²: 0.8857 |
| Lasso Regression | MAE: 2846.75 | RMSE: 4582.43 | R²: 0.8857 |

---

# 📉 Visualizations

Included:

- Correlation heatmap
- Coefficient Shrinkage Plot
- Predicted vs Actual Plot

---

# 📌 Interpretation

Discuss:

- Lasso successfully forced useless features to exactly zero, performing automatic feature selection.
- Ridge shrank coefficients but kept them all active.
- The smoker feature had the largest coefficient by a massive margin.

---

# 🧠 Key Learnings

- Lasso (L1) creates sparse models by eliminating features.
- Ridge (L2) prevents any single feature from having too much weight.
- Regularization requires scaled data to work properly.

---

# 🛠 Tech Stack

- Python
- Pandas
- Scikit-Learn
- Matplotlib

---

# 🚀 Future Improvements

- Polynomial features
- Decision Trees

---

# Author

Built as part of my **100 Days of Machine Learning**