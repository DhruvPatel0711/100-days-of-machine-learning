# Classification Pipeline Notes

## Why This Topic Matters

Real-world datasets are rarely clean and rarely balanced. E-commerce fraud might represent only 0.2% of transactions. 

A model predicting "No Fraud" every time gets 99.8% accuracy, but is utterly useless. We need Pipelines to handle complex transformations and specialized metrics to handle severe imbalance.

---

# Imbalanced Data

## Definition

A dataset where the target classes are not represented equally (e.g., 99% Class 0, 1% Class 1).

### Formula (if applicable)

```text
Accuracy = (TP + TN) / Total
```
*(Shows why Accuracy is misleading if TN is 99% of the Total).*

### Interpretation

The model naturally biases toward the majority class because predicting it yields the lowest mathematical loss.

### Example

Credit Card Fraud Detection. Normal transactions vastly outnumber fraudulent ones.

---

# Precision-Recall Curve (PR-AUC)

## Definition

A metric curve that replaces ROC-AUC when dealing with highly imbalanced data. It plots Precision vs Recall across thresholds.

### Why It Matters

ROC-AUC is artificially inflated by the massive number of True Negatives. The PR-Curve ignores True Negatives and focuses entirely on the minority positive class.

### Example

A PR-AUC of 0.85 means the model maintains high precision even as you lower the threshold to capture more fraud (recall).

---

# Sklearn Pipeline

## Definition

An object that chains together multiple preprocessing steps and a final estimator into a single, cohesive workflow.

### Example

`Pipeline([ ('scaler', StandardScaler()), ('model', LogisticRegression()) ])`

### Visualization explanation

A factory conveyor belt where raw data enters, gets scaled automatically, and feeds directly into the model for prediction without manual intervention.

---

# Mathematical Intuition

Algorithms like Logistic Regression can be mathematically forced to care about the minority class by applying `class_weight='balanced'`. This artificially multiplies the loss penalty for getting a minority class wrong by the inverse of its frequency, forcing the optimizer to pay attention to it.

---

# How It Works

Step-by-step process.

```text
Raw Imbalanced Data
↓
ColumnTransformer (Scale numbers, Encode categories)
↓
Pipeline passes processed data to Model
↓
Model optimized using PR-AUC
```

---

# Advantages

- Pipelines completely eliminate data leakage during cross-validation.
- PR-AUC provides a truthful assessment of model performance on rare events.
- Class weighting allows standard algorithms to handle imbalance naturally.

---

# Limitations

- SMOTE (synthetic oversampling) can sometimes create unrealistic "Frankenstein" data points.
- Precision and Recall are a tradeoff; you usually cannot maximize both.
- Pipelines can be tricky to debug if a specific step fails silently.

---

# Common Mistakes

- Using Accuracy as the primary metric for fraud detection.
- Scaling the entire dataset *before* performing the train-test split (Data Leakage).
- Using ROC-AUC instead of PR-AUC for a 99-to-1 imbalanced dataset.

---

# Practical Interpretation

Explain how to interpret outputs.

Examples:

> What does a high Precision but low Recall mean?
When the model flags fraud, it is almost always right. But it misses a lot of actual fraud that slips through.

> When should this method be used?
When building production-ready code (Pipelines) for rare-event detection (Fraud, Disease, Manufacturing Defects).

> When should it not be used?
Pipelines should ALWAYS be used. Never write raw fit/transform spaghetti code in production.

---

# Industry Applications

Examples:

- Finance: Credit Card Fraud Detection (minimizing false negatives).
- Cybersecurity: Detecting network intrusions and malware signatures.
- Manufacturing: Identifying defective parts on an assembly line.

---

# Quick Comparison

| Concept | Meaning |
|----------|---------|
| ROC-AUC | Best for balanced datasets (looks at FPR) |
| PR-AUC | Best for imbalanced datasets (ignores TNs entirely) |
| SMOTE | Synthetically generating new minority class rows |

---

# Resources

- IBM ML with Python — Classification module
- 100 Days of ML Code — classification problems
