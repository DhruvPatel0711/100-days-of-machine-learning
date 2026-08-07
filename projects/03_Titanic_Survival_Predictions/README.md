# 🚀 Titanic Survival Predictions

Predicting passenger survival on the Titanic using Logistic Regression.

Dataset:
https://www.kaggle.com/c/titanic

---

# 🎯 Objective

Predict whether a passenger survived (1) or died (0).

It matters as a foundational introduction to binary classification, dealing with probabilities, decision boundaries, and categorical encoding.

- Logistic Regression

---

# 📊 Dataset

Titanic Dataset

## Features

| Pclass | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| Sex | Gender |
| Age | Age in years |
| Fare | Passenger fare |

## Target

```text
Target Classes (Classification)
```

Class 0 = Died

Class 1 = Survived

---

# 🔍 Exploratory Data Analysis

Performed:

- Missing value analysis
- Categorical distribution
- Survival rate by class

### Key Observations

- Females had a significantly higher survival rate than males.
- 1st class passengers survived at a higher rate than 3rd class.
- Most missing values are in the 'Age' and 'Cabin' columns.

---

# 🛠 Feature Engineering

Describe every engineered feature.

## Family Size

```python
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
```

Purpose:

Combines siblings/spouses and parents/children to capture the total size of a family, determining if large families survived less.

---

## Title Extraction

Purpose:

Extracted titles (Mr., Mrs., Miss.) from the Name column. Titles give a better estimate of social status and age (e.g., 'Master' for young boys) than raw age.

---

# ⚙️ Data Preprocessing

## Missing Values

Missing 'Age' values were imputed using the median age of the passenger's corresponding 'Title'. Missing 'Embarked' filled with the mode.

---

## Encoding

One-Hot Encoding for 'Sex', 'Embarked', and 'Title'.

---

## Feature Scaling

StandardScaler applied to continuous variables ('Age', 'Fare') to help the logistic regression optimizer converge faster.

---

## Train-Test Split

```text
80% Training

20% Testing
```

---

# 🤖 Models Implemented

- Logistic Regression

Chosen as it predicts probabilities using the sigmoid function, translating log-odds into a 0-to-1 range.

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
| Logistic Regression | 0.7989 | 0.7500 | 0.7703 | 0.7703 | - |

---

# 📉 Visualizations

Included:

- Confusion Matrix Heatmap
- Feature distributions by survival

---

# 📌 Interpretation

Discuss:

- 'Sex_female' had the highest positive coefficient (increased odds of survival).
- 'Pclass_3' had the highest negative coefficient (decreased odds of survival).
- The model struggles with false negatives (predicting death when the passenger survived).

---

# 🧠 Key Learnings

- Logistic Regression uses a decision boundary (typically 0.5) to convert probabilities into distinct classes.
- Accuracy isn't everything; the confusion matrix reveals exactly where the model makes mistakes.

---

# 🛠 Tech Stack

- Python
- Pandas
- Scikit-Learn
- Seaborn

---

# 🚀 Future Improvements

- Hyperparameter tuning
- Ensemble models

---

# Author

Built as part of my **100 Days of Machine Learning**