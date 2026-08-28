# 🚀 Overfitting Demonstration Model

A deliberate demonstration of model overfitting and how cross-validation fixes it.

Dataset:
Standard Datasets

---

# 🎯 Objective

Force a model to achieve 100% training accuracy but fail on test data.

It matters because understanding the Bias-Variance tradeoff is the most critical conceptual foundation in machine learning.

- Ridge Regression
- Lasso Regression

---

# 📊 Dataset

Medical Cost Dataset

## Features

| Various | Standard features from previous sets |

## Target

```text
Target (Regression)
```

Continuous output (Medical Costs).

---

# 🔍 Exploratory Data Analysis

Performed:

- Inspecting baseline metrics
- Preparing the data for deliberate corruption

### Key Observations

- Data looks standard, but will be forced into high-variance conditions.

---

# 🛠 Feature Engineering

Describe every engineered feature.

## Polynomial Features (Degree 15)

```python
poly = PolynomialFeatures(degree=15)
X_poly = poly.fit_transform(X)
```

Purpose:

Creates mathematically massive complexity, allowing the model to draw a wildly complex line that hits every single training point perfectly.

---

## Unrestricted Complexity

Purpose:

Allowing regression models or trees to have no regularization causes them to map every noise data point, essentially memorizing the data.

---

# ⚙️ Data Preprocessing

## Missing Values

Standard median imputation.

---

## Encoding

Standard one-hot encoding.

---

## Feature Scaling

Standardization applied before polynomial expansion to prevent computational overflow.

---

## Train-Test Split

```text
80% Training

20% Testing
```

---

# 🤖 Models Implemented

- Ridge Regression
- Lasso Regression

Chosen specifically because they penalize complexity to prevent high-variance overfitting.

---

# 📈 Evaluation Metrics

Used:

- Mean CV R²
- MAE
- RMSE

---

# 📊 Results

## Performance Comparison

| Model | MAE | RMSE | R² |
|--------|----------|-----------|--------|
| Ridge | MAE: 0.4865 | RMSE: 0.6746 | R²: 0.6527 |
| Lasso | MAE: 0.4878 | RMSE: 0.6768 | R²: 0.6505 |

---

# 📉 Visualizations

Included:

- Learning Curves
- Train vs Test Metric Collapse charts

---

# 📌 Interpretation

Discuss:

- Forcing ultra-high alpha values (Alpha=1e6) drops R² to near 0, demonstrating extreme underfitting (high bias).
- Using CV helps find the exact best alpha to balance bias and variance.

---

# 🧠 Key Learnings

- A model that memorizes the training data is useless in the real world.
- Cross-validation prevents you from getting 'lucky' with a good train/test split.
- The gap between training error and validation error defines overfitting.

---

# 🛠 Tech Stack

- Python
- Scikit-Learn
- Matplotlib

---

# 🚀 Future Improvements

- Regularization techniques
- Ensemble bagging

---

# Author

Built as part of my **100 Days of Machine Learning**