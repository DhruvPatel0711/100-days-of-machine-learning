# Linear Regression Notes

## Why This Topic Matters

Linear regression is the foundation of all predictive modeling for continuous numerical values. 

It provides a baseline understanding of how algorithms draw decision boundaries and calculate relationships between variables, which carries over directly into deep learning.

---

# The Linear Equation

## Definition

The mathematical formula that describes a straight line through data points, minimizing the distance between the line and the points.

### Formula (if applicable)

```text
y = b + w₁x₁ + w₂x₂ + ...
```

### Interpretation

- `y` is the prediction.
- `b` is the bias (y-intercept).
- `w` are the weights (how much a feature matters).
- `x` are the features.

### Example

Predicting salary:
Salary = $30,000 + ($5,000 * Years_of_Experience)

---

# Mean Squared Error (MSE)

## Definition

The standard loss function used to evaluate regression models.

### Why It Matters

It penalizes large errors exponentially more than small errors by squaring the differences.

### Example

Prediction = 10, Actual = 15. Error = -5. Squared Error = 25.

---

# Gradient Descent

## Definition

The optimization algorithm that iteratively tweaks the weights to minimize the MSE.

### Example

Imagine walking blindfolded down a mountain. You feel the slope with your feet and take a step in the steepest downward direction until you reach the flat bottom (minimum loss).

### Visualization explanation

A U-shaped curve (the loss surface) where a ball rolls down the sides until it settles at the lowest point.

---

# Mathematical Intuition

Linear regression uses calculus (derivatives) to find the slope of the loss function. By subtracting a fraction of the derivative (the learning rate) from the current weight, the model mathematically guarantees it moves closer to the minimum error.

---

# How It Works

Step-by-step process.

```text
Initialize random weights
↓
Make predictions
↓
Calculate MSE loss
↓
Update weights via Gradient Descent
```

---

# Advantages

- Extremely fast to train and predict.
- 100% explainable (you can read the exact formula).
- Requires very little computational power.

---

# Limitations

- Assumes a strict linear relationship between features and target.
- Highly sensitive to extreme outliers.
- Cannot capture complex, non-linear interactions without heavy feature engineering.

---

# Common Mistakes

- Forgetting to scale features, which causes gradient descent to converge very slowly.
- Including highly correlated features (multicollinearity), which destabilizes the weights.
- Using it for classification problems.

---

# Practical Interpretation

Explain how to interpret outputs.

Examples:

> What does a high R-squared value mean?
It means a large percentage of the variance in the target is explained by your features (e.g., 0.85 means 85% explained).

> When should this method be used?
When predicting continuous numbers like prices, ages, or temperatures, and you need a highly interpretable baseline.

> When should it not be used?
When the data is highly non-linear or when predicting categories (e.g., Spam vs Not Spam).

---

# Industry Applications

Examples:

- Finance: Predicting future stock prices based on moving averages.
- Real Estate: Estimating property values (House Price Prediction).
- Retail: Forecasting next month's sales volume.

---

# Quick Comparison

| Concept | Meaning |
|----------|---------|
| R-squared | Percentage of variance explained (0 to 1) |
| RMSE | Average error in the actual units of the target |
| Learning Rate | The step size taken during gradient descent |

---

# Resources

- Google ML Crash Course — Linear Regression, Reducing Loss
- StatQuest — "Linear Regression", "R-squared explained", "Mean Squared Error"
- 3Blue1Brown — "Gradient Descent"
