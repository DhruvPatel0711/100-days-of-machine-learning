# 🚀 Wine Quality Classifier

Classifying wine quality using k-Nearest Neighbors and proving the absolute necessity of feature scaling.

Dataset:
https://archive.ics.uci.edu/ml/datasets/wine+quality

---

# 🎯 Objective

Predict if a wine is 'Good' or 'Bad' based on chemical tests.

It matters because kNN is a foundational distance-based algorithm, and this project serves as a masterclass in why distance metrics fail on unscaled data.

- k-Nearest Neighbors (kNN)
- Logistic Regression

---

# 📊 Dataset

Wine Quality (Red)

## Features

| Feature | Description |
|--------|-------------|
| alcohol | Alcohol content % |
| fixed acidity | Tartaric acid |
| volatile acidity | Acetic acid (too much causes vinegar taste) |
| sulphates | Antimicrobial additive |
| density | Density of wine |
| total sulfur dioxide | SO2 content |

## Target

```text
Target Classes (Classification)
```

Class 0 = Bad (Quality < 6)

Class 1 = Good (Quality >= 6)

---

# 🔍 Exploratory Data Analysis

Performed:

- Dataset overview
- Missing values and duplicates check
- Class distribution and feature distributions
- Correlation heatmap

### Key Observations

- Alcohol content has the highest positive correlation with Good wine.
- Volatile acidity has the highest negative correlation.
- The dataset is relatively balanced after creating the binary threshold.

---

# 🛠 Feature Engineering

## Acidity Ratio
Created a ratio of fixed acidity to volatile acidity. High volatile acidity ruins wine, so this ratio acts as a freshness index.

## Alcohol Bins
Binned alcohol content into Low, Medium, and High categories using `pd.cut`.

## Quality Threshold Binary
Converted a 1-10 quality scale into a binary classification problem to make decision boundaries clearer for kNN.

## Density × Alcohol
Combined density and alcohol to capture complex fermentation characteristics.

## Sulfur Ratio
Ratio of free sulfur dioxide to total sulfur dioxide.

## pH Category
Binned pH into Low, Medium, High using `pd.qcut`.

---

# ⚙️ Data Preprocessing

## Train-Test Split
80% Training, 20% Testing, Stratified on the target variable.

## Feature Scaling
Utilized a Pipeline implementation comparing `StandardScaler`, `MinMaxScaler`, and `passthrough` (no scaling). **This was the core focus of the project.** Demonstrated metric collapse when unscaled.

---

# 🤖 Models Implemented

- Logistic Regression
- k-Nearest Neighbors (Euclidean)
- k-Nearest Neighbors (Manhattan)
- k-Nearest Neighbors (Minkowski)

Chosen to contrast a purely spatial/distance-based algorithm (kNN) against a linear boundary algorithm (LogReg).

---

# 🎛 Hyperparameter Tuning

Used **GridSearchCV** to tune kNN:
- `k` (Number of neighbors)
- `weights` (Uniform vs Distance)
- `metric` (Distance calculations)
- `p` (Power parameter for Minkowski distance: 1=Manhattan, 2=Euclidean)

---

# 📈 Evaluation Metrics

Used:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Cross Validation

---

# 📉 Visualizations

Included:
- Confusion Matrix
- ROC Curves
- PR Curves
- k vs Accuracy
- k vs F1
- CV comparison
- Decision Boundary (only first two PCA components)
- Feature importance (using permutation importance)

---

# 📌 Explainability & Conclusions

**Explainability:** Used Permutation Importance to highlight which features drastically impacted model predictions when shuffled.

**Scaling vs No Scaling:** GridSearchCV confirmed that `StandardScaler` or `MinMaxScaler` drastically outperforms no scaling (`passthrough`) because kNN relies entirely on spatial distances. Without scaling, features with massive magnitudes dominate the calculation.

**Logistic vs kNN:** While Logistic Regression draws a rigid, linear boundary, kNN draws a highly non-linear, flexible boundary. Depending on the local density of good vs bad wine, kNN often yields higher metrics on this dataset once the optimal `k` is found via Cross Validation.

---

# 🧠 Key Learnings

- kNN relies entirely on spatial distance metrics.
- Feature scaling is 100% non-negotiable for distance-based algorithms.
- Pipelines completely eliminate data leakage during GridSearch and scaling.

---

# 🛠 Tech Stack

- Python
- Scikit-Learn
- Matplotlib
- Seaborn

---

# Author

Built as part of my **100 Days of Machine Learning**