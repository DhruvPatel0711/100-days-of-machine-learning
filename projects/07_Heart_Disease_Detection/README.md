# 🚀 Heart Disease Prediction

Predicting the presence of heart disease using Decision Trees and identifying split conditions.

Dataset:
https://archive.ics.uci.edu/ml/datasets/heart+Disease

---

# 🎯 Objective

Classify whether a patient has heart disease based on medical attributes.

It matters because Decision Trees provide white-box explainability, allowing doctors to see exactly why a prediction was made.

- Decision Trees

---

# 📊 Dataset

Heart Disease (UCI)

## Features

| age | Age in years |
| chol | Serum cholesterol in mg/dl |
| trestbps | Resting blood pressure |
| thalach | Maximum heart rate achieved |

## Target

```text
Target Classes (Classification)
```

Class 0 = No Heart Disease

Class 1 = Heart Disease

---

# 🔍 Exploratory Data Analysis

Performed:

- Target distribution
- Feature distributions
- Correlation analysis

### Key Observations

- Maximum heart rate (thalach) is negatively correlated with heart disease.
- Age and cholesterol show a positive correlation with the disease.
- Dataset is relatively balanced.

---

# 🛠 Feature Engineering

Describe every engineered feature.

## Cholesterol Categories

```python
df['chol_cat'] = pd.cut(df['chol'], bins=[0, 200, 239, 600], labels=['Desirable', 'Borderline', 'High'])
```

Purpose:

Medical diagnoses often rely on established thresholds (e.g., >240 is high risk). Categorizing helps the tree make cleaner splits.

---

## Blood Pressure Ratio

Purpose:

Created a ratio of resting blood pressure to age, testing the hypothesis that high BP is more dangerous at younger ages.

---

# ⚙️ Data Preprocessing

## Missing Values

Rows with '?' were replaced with NaN and imputed with the median.

---

## Encoding

One-Hot Encoding for categorical features like chest pain type.

---

## Feature Scaling

Not required. Decision Trees do not compute distances and are invariant to feature scaling.

---

## Train-Test Split

```text
80% Training

20% Testing
```

---

# 🤖 Models Implemented

- Decision Tree Classifier

Chosen to demonstrate Information Gain, Gini Impurity, and visually plotting the exact decision logic.

---

# 📈 Evaluation Metrics

Used:

- Accuracy
- Precision
- Recall
- F1 Score

---

# 📊 Results

## Performance Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Unpruned Tree | 0.5870 | 0.5902 | 0.5870 | 0.5883 | 0.7155 |
| Pruned Tree | 0.5870 | 0.7466 | 0.5870 | 0.6313 | 0.8425 |

---

# 📉 Visualizations

Included:

- Full Tree Visualisation (plot_tree)
- Confusion Matrix
- Overfitting comparison at different max_depths

---

# 📌 Interpretation

Discuss:

- The unpruned tree went 15 layers deep, finding absurd rules to classify single patients.
- Pruning the tree yielded far superior ROC-AUC (0.84 vs 0.71) and better precision.
- Chest pain type (cp) was the very first split, meaning it had the highest Information Gain.

---

# 🧠 Key Learnings

- Decision Trees split nodes based on purity (Gini) or chaos reduction (Entropy).
- They are massively prone to overfitting without hyperparameter pruning (`max_depth`, `min_samples_leaf`).
- They require zero feature scaling.

---

# 🛠 Tech Stack

- Python
- Pandas
- Scikit-Learn
- Matplotlib

---

# 🚀 Future Improvements

- Random Forests
- Hyperparameter GridSearch

---

# Author

Built as part of my **100 Days of Machine Learning**