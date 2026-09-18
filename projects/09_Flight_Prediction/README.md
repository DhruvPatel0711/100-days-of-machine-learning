# 🚀 Flight Price Prediction

Benchmarking XGBoost against Random Forest to predict airline ticket prices.

Dataset:
https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction

---

# 🎯 Objective

Predict ticket prices or classify ticket tiers using gradient boosting.

It matters because dynamic pricing engines require extremely low-bias, highly accurate models that XGBoost is uniquely designed to provide.

- XGBoost
- Random Forest
- Gradient Boosting

---

# 📊 Dataset

Flight Price Dataset

## Features

| airline | Airline name |
| source_city | Departure city |
| duration | Flight duration in hours |
| days_left | Days between booking and departure |

## Target

```text
Target Classes (Classification)
```

Class 0 = Economy

Class 1 = Business

---

# 🔍 Exploratory Data Analysis

Performed:

- Price vs Days Left analysis
- Airline price comparisons
- Time of day analysis

### Key Observations

- Prices skyrocket when booked less than 3 days in advance.
- Vistara and Air India dominate the Business class data.
- Duration is positively correlated with price.

---

# 🛠 Feature Engineering

Describe every engineered feature.

## Departure Hour

```python
df['dep_hour'] = pd.to_datetime(df['departure_time']).dt.hour
```

Purpose:

Time of day dictates business travel demand. Extracting the explicit hour allows the model to capture morning/evening price surges.

---

## Route Encoding

Purpose:

Combined source_city and destination_city into a single 'Route' feature, as specific city-pairs (e.g., Delhi-Mumbai) have unique pricing dynamics independent of just the source.

---

# ⚙️ Data Preprocessing

## Missing Values

Dropped minimal missing rows.

---

## Encoding

Label Encoding for target. One-Hot encoding for nominal categories (airlines, cities).

---

## Feature Scaling

Not required for tree-based models.

---

## Train-Test Split

```text
80% Training

20% Testing
```

---

# 🤖 Models Implemented

- Random Forest
- Gradient Boosting Classifier
- XGBoost Classifier

Chosen to demonstrate the evolution from Bagging (RF) to Sequential Error Correction (GBM) to Optimized Extreme Boosting (XGB).

---

# 📈 Evaluation Metrics

Used:

- ROC-AUC
- F1 Score
- Cross Validation Scores

---

# 📊 Results

## Performance Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Random Forest | 0.6475 | 0.8261 | 0.6475 | 0.6515 | 0.7628 |
| Gradient Boosting | 0.7006 | 0.6764 | 0.7006 | 0.6785 | 0.7800 |
| XGBoost | 0.6922 | 0.7459 | 0.6922 | 0.5712 | 0.7928 |

---

# 📉 Visualizations

Included:

- Feature Importances
- ROC Curve Comparison
- Confusion Matrix Comparison

---

# 📌 Interpretation

Discuss:

- XGBoost outperformed Random Forest by learning from the residuals (errors) of previous trees sequentially.
- The model is incredibly accurate because class separation (Economy vs Business) is distinct.
- Airline and Duration were the top predictors.

---

# 🧠 Key Learnings

- Bagging builds trees independently; Boosting builds them sequentially.
- XGBoost includes built-in regularization (L1/L2) preventing the sequential trees from overfitting the noise.
- XGBoost requires careful tuning of learning_rate and max_depth.

---

# 🛠 Tech Stack

- Python
- Scikit-Learn
- XGBoost
- Pandas

---

# 🚀 Future Improvements

- Hyperparameter GridSearch
- LightGBM comparison

---

# Author

Built as part of my **100 Days of Machine Learning**