# Ridge & Lasso Regularization Notes

## Why This Topic Matters

Standard linear regression often overfits when there are too many features. It assigns wild, massive weights to noise.

Regularization forces the model to stay simple by mathematically penalizing complex, large weights, making the model far more robust on unseen data in the real world.

---

# L2 Regularization (Ridge)

## Definition

Adds a penalty to the loss function equal to the squared magnitude of the coefficients.

### Formula (if applicable)

```text
New Loss = MSE + λ * Σ(w²)
```

### Interpretation

It shrinks all weights toward zero, but never exactly to zero. It prevents any single feature from dominating the model.

### Example

If Age has a massive weight of 5000, Ridge will pull it down to 500 to ensure the model doesn't over-rely on it.

---

# L1 Regularization (Lasso)

## Definition

Adds a penalty to the loss function equal to the absolute value of the coefficients.

### Why It Matters

It performs automatic feature selection by forcing the weights of useless features to exactly 0.0.

### Example

In a dataset with 500 features, Lasso might shrink 450 of them to exactly 0, leaving only the 50 most important features active.

---

# Hyperparameter Alpha (λ)

## Definition

The tuning knob that controls the strength of the penalty.

### Example

Alpha = 0 is standard Linear Regression. Alpha = 100 is heavy regularization (mostly zeros).

### Visualization explanation

As alpha increases, you can visualize the coefficient lines dropping drastically toward the x-axis (zero) on a coefficient shrinkage plot.

---

# Mathematical Intuition

Ridge uses a circular constraint region (L2 norm), which causes the loss function to intersect it smoothly, leaving small but non-zero weights. Lasso uses a diamond-shaped constraint region (L1 norm). The sharp corners of the diamond cause the loss function to intersect exactly on the axes, which forces coefficients to mathematically become exactly zero.

---

# How It Works

Step-by-step process.

```text
Calculate standard MSE
↓
Add penalty term (L1 or L2)
↓
Optimize total combined loss
↓
Output shrunken coefficients
```

---

# Advantages

- Prevents overfitting on high-variance datasets.
- Lasso acts as a built-in feature selection tool.
- Handles multicollinearity better than standard regression.

---

# Limitations

- Requires all features to be perfectly scaled (StandardScaler), otherwise the penalty is applied unevenly.
- You must manually tune the alpha hyperparameter using Cross-Validation.
- Lasso can arbitrarily drop one of two highly correlated features.

---

# Common Mistakes

- Forgetting to scale features before applying Ridge/Lasso.
- Setting alpha too high, which causes massive underfitting (predicting a flat line).
- Using Lasso when you actually want to keep all features active.

---

# Practical Interpretation

Explain how to interpret outputs.

Examples:

> What does a high alpha value mean?
It means you are heavily penalizing complexity, forcing the model to be very simple and rigid.

> When should this method be used?
When you have more features than rows of data, or when standard linear regression is overfitting the training data.

> When should it not be used?
When you have very few features and massive amounts of data, where standard regression won't overfit anyway.

---

# Industry Applications

Examples:

- Healthcare: Predicting medical costs (Medical Cost Predictor) where many lifestyle features might just be noise.
- Genomics: Identifying which of 10,000 genes actually predict a disease (Lasso is perfect here).
- Finance: Stress-testing economic indicators.

---

# Quick Comparison

| Concept | Meaning |
|----------|---------|
| Ridge (L2) | Shrinks weights small, keeps them all |
| Lasso (L1) | Shrinks weights to exactly zero (feature selection) |
| Alpha (λ) | The strength of the regularization penalty |

---

# Resources

- Krish Naik — Ridge and Lasso practical
- StatQuest — "Regularization Part 1 (Ridge)" + "Part 2 (Lasso)"
