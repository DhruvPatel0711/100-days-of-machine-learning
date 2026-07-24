# 🚀 House Price Prediction

Predicting the median house value for California districts using Linear Regression.

Dataset:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

---

# 🎯 Objective

Predict the continuous value of a house based on demographic and geographical data.

It matters because real estate valuation requires accurate, data-driven approaches rather than human guesswork.

- Linear Regression

---

# 📊 Dataset

California Housing Dataset

## Features

| MedInc | Median income in block group |
| HouseAge | Median house age in block group |
| AveRooms | Average number of rooms per household |
| Population | Block group population |

## Target

```text
Target (Regression)
```

MedianHouseValue (Continuous Numerical Value in $100,000s)

---

# 🔍 Exploratory Data Analysis

Performed:

- Dataset inspection
- Statistical summaries
- Distribution analysis
- Correlation analysis

### Key Observations

- Median income has the highest positive correlation with house value.
- The data contains outliers in the average rooms feature.
- The geographical data (Latitude/Longitude) shows distinct high-value clusters (e.g., Bay Area).

---

# 🛠 Feature Engineering

Describe every engineered feature.

## Price per Room Ratio

```python
df['price_per_room'] = df['MedInc'] / df['AveRooms']
```

Purpose:

Provides a normalized metric of affordability, assuming higher income per room indicates a premium neighborhood.

---

## Log Transform Skewed Features

Purpose:

Population and AveRooms are heavily right-skewed. Taking the logarithm normalizes the distribution, which is required for Linear Regression to perform well.

---

# ⚙️ Data Preprocessing

## Missing Values

The sklearn California housing dataset comes pre-cleaned with zero missing values. If there were any, median imputation would be used.

---

## Encoding

No categorical variables exist in this dataset, thus encoding was not required.

---

## Feature Scaling

StandardScaler was applied to ensure all numerical features have a mean of 0 and a standard deviation of 1. Linear regression is sensitive to the scale of features.

---

## Train-Test Split

```text
80% Training

20% Testing
```

---

# 🤖 Models Implemented

- Linear Regression

Chosen because it establishes a solid, interpretable baseline for continuous prediction tasks.

---

# 📈 Evaluation Metrics

Used:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

---

# 📊 Results

## Performance Comparison

| Model | MAE | RMSE | R² |
|--------|----------|-----------|--------|
| Linear Regression | MAE: 0.4815 | RMSE: 0.6648 | R²: 0.6627 |

---

# 📉 Visualizations

Included:

- Distribution plots
- Correlation heatmap
- Predicted vs Actual Plot

---

# 📌 Interpretation

Discuss:

- The Linear Regression model provides a reasonable baseline but struggles with the complex, non-linear geographical clusters.
- The best feature was MedInc, proving that localized wealth drives property values.

---

# 🧠 Key Learnings

- Linear Regression assumes a linear relationship; when data is non-linear, it underfits.
- R-squared explains the variance, but RMSE gives the real dollar-value error.
- Feature scaling makes gradient descent converge much faster.

---

# 🛠 Tech Stack

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

# 🚀 Future Improvements

- Hyperparameter tuning (Ridge/Lasso)
- Additional algorithms (Random Forest)
- Geospatial feature engineering

---

# Author

Built as part of my **100 Days of Machine Learning**