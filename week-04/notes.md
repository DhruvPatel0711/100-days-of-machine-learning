# Logistic Regression Notes

## Why This Topic Matters

Linear regression fails completely when trying to predict categories (like Yes/No). It produces impossible probabilities like -0.5 or 1.2.

Logistic regression solves this by mathematically squishing the output between 0 and 1, making it the industry standard for baseline binary classification.

---

# Log Odds

## Definition

Logistic regression models the log of the odds rather than the probability directly.

### Formula (if applicable)

```text
log( p / (1 - p) ) = b + w₁x₁ + w₂x₂ + ...
```

### Interpretation

Because a linear equation can output anything from -∞ to +∞, we must predict "Log Odds" first, because log odds also span from -∞ to +∞.

### Example

Probability = 0.80. Odds = 4 to 1. Log Odds = 1.38. The linear equation outputs the 1.38.

---

# The Sigmoid Function

## Definition

The mathematical function that converts the log odds back into a readable probability between 0 and 1.

### Why It Matters

It creates the classic "S-curve" that ensures probabilities never exceed 100% or drop below 0%.

### Example

A log odds output of 0 passed through the sigmoid function becomes exactly 0.5 (50% probability).

---

# Decision Boundary

## Definition

The threshold used to convert the continuous probability into a hard class label.

### Example

If threshold = 0.5:
Probability 0.80 → Class 1 (Survived).
Probability 0.30 → Class 0 (Died).

### Visualization explanation

A literal line drawn through a scatterplot separating the blue dots (Class 0) from the red dots (Class 1).

---

# Mathematical Intuition

Instead of minimizing Mean Squared Error, Logistic Regression uses Maximum Likelihood Estimation (MLE). It iteratively adjusts the S-curve to maximize the mathematical probability that the model would have generated the exact true labels seen in the training data.

---

# How It Works

Step-by-step process.

```text
Linear Equation predicts Log Odds (z)
↓
Sigmoid function converts z to Probability (p)
↓
Compare p to Decision Boundary (e.g., 0.5)
↓
Assign Class Label (0 or 1)
```

---

# Advantages

- Highly interpretable coefficients (you know exactly how much Age affects the odds of survival).
- Outputs properly calibrated probabilities, not just hard labels.
- Very fast to train and extremely lightweight in production.

---

# Limitations

- Assumes a linear decision boundary (cannot easily classify a circle of points inside another circle of points).
- Struggles with complex, non-linear relationships without heavy feature engineering.
- Vulnerable to severe class imbalances without threshold tuning.

---

# Common Mistakes

- Using MSE as the loss function instead of Log Loss.
- Treating the raw output of the linear equation as a probability.
- Forgetting to encode categorical variables (like Gender) before training.

---

# Practical Interpretation

Explain how to interpret outputs.

Examples:

> What does a high positive coefficient mean?
As that feature increases, the odds of the positive class (Class 1) drastically increase.

> When should this method be used?
As the first baseline model for any binary classification task (e.g., Titanic Survival, Spam Detection).

> When should it not be used?
When predicting continuous numerical values like Price or Temperature.

---

# Industry Applications

Examples:

- Banking: Predicting if a customer will default on a loan (Default = 1, Paid = 0).
- Healthcare: Diagnosing a benign vs malignant tumor based on measurements.
- Marketing: Predicting if a user will click an ad (CTR prediction).

---

# Quick Comparison

| Concept | Meaning |
|----------|---------|
| Probability | Chance of an event (0 to 1) |
| Odds | Ratio of success to failure (0 to ∞) |
| Log Loss | The loss function used to optimize Logistic Regression |

---

# Resources

- Google ML Crash Course — Logistic Regression, Classification
- StatQuest — "Logistic Regression" (main video + odds ratio video)
