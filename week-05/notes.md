# SVM & Evaluation Metrics Notes

## Why This Topic Matters

Logistic regression draws a good line, but Support Vector Machines (SVM) draw the *best* line by maximizing the safety margin between classes.

Furthermore, relying purely on "Accuracy" is dangerous in real-world scenarios. We need advanced metrics like Precision, Recall, and ROC-AUC to truly evaluate model performance.

---

# Support Vector Machine (SVM)

## Definition

A classifier that finds the hyperplane (decision boundary) that maximizes the margin (distance) between the two classes.

### Formula (if applicable)

```text
Maximize: 2 / ||w||
Subject to: y_i(w*x_i + b) >= 1
```

### Interpretation

Instead of just separating the data, it finds the widest possible "street" between the classes.

### Example

Separating malignant and benign tumors with the widest possible margin to ensure future unseen tumors are classified safely.

---

# Support Vectors

## Definition

The specific data points that lie closest to the decision boundary.

### Why It Matters

They are the only points that actually matter to the model. If you delete all other points, the SVM boundary remains identical.

### Example

The few malignant tumors that look most like benign tumors are the "support vectors" propping up the mathematical boundary.

---

# ROC-AUC

## Definition

Receiver Operating Characteristic - Area Under the Curve. It plots the True Positive Rate vs the False Positive Rate across *all possible* decision thresholds.

### Example

An AUC of 0.90 means there is a 90% chance the model will rank a randomly chosen positive instance higher than a randomly chosen negative one.

### Visualization explanation

A curve bowing upward toward the top-left corner. The closer it hugs the top-left, the closer the AUC is to 1.0 (perfect). A diagonal line is AUC 0.5 (random guessing).

---

# Mathematical Intuition

If data cannot be separated by a straight line in 2D space, SVM uses the "Kernel Trick" (like the RBF kernel) to project the data into a higher mathematical dimension (like 3D) where a flat plane *can* cleanly slice the classes apart, without actually computing the heavy 3D math.

---

# How It Works

Step-by-step process.

```text
Plot data in N-dimensional space
↓
Apply Kernel to project to higher dimensions if non-linear
↓
Find maximum margin hyperplane using support vectors
↓
Classify new points based on which side of the plane they fall
```

---

# Advantages

- Incredibly effective in high-dimensional spaces (where features > rows).
- Memory efficient since it only saves the support vectors.
- Kernels allow it to solve highly complex, non-linear classification boundaries.

---

# Limitations

- Extremely sensitive to feature scaling (unscaled data ruins the distance calculations).
- Very slow to train on large datasets (O(N^3) time complexity).
- Does not natively output probabilities (requires heavy calibration).

---

# Common Mistakes

- Forgetting to apply `StandardScaler` to the data before training an SVM.
- Looking only at 'Accuracy' when evaluating a model on imbalanced medical data.
- Using a highly complex RBF kernel when a simple Linear kernel would suffice.

---

# Practical Interpretation

Explain how to interpret outputs.

Examples:

> What does high Recall mean?
It means the model found almost all the positive cases (e.g., caught all the cancer). It is highly sensitive.

> When should this method be used?
When the boundary between classes is complex and you have a moderate amount of high-dimensional data.

> When should it not be used?
On datasets with millions of rows, as training an SVM will take days.

---

# Industry Applications

Examples:

- Healthcare: Breast Cancer Detection (where Recall is optimized over Precision).
- Bioinformatics: Protein structure classification.
- Text Categorization: Categorizing thousands of documents based on TF-IDF vectors.

---

# Quick Comparison

| Concept | Meaning |
|----------|---------|
| Precision | When predicting positive, how often is it right? |
| Recall | Out of all actual positives, how many were found? |
| F1 Score | The harmonic mean balancing Precision and Recall |

---

# Resources

- StatQuest — "Confusion Matrix", "Sensitivity and Specificity", "ROC and AUC", "Log Loss Explained", "Support Vector Machines"
- Google ML Crash Course — Classification: ROC and AUC
