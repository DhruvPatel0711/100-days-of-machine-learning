# Decision Trees Notes

## Why This Topic Matters

Unlike "black box" models (like Neural Networks or SVMs), Decision Trees provide 100% transparency. You can print the tree and show a doctor exactly why a diagnosis was made.

They are the foundational building blocks for the most powerful algorithms in ML (Random Forests and XGBoost).

---

# Decision Tree Classifier

## Definition

An algorithm that repeatedly splits data based on simple Yes/No questions to maximize the purity of the resulting groups.

### Formula (if applicable)

```text
Is Age > 50? 
  ├── Yes -> High Risk
  └── No  -> Low Risk
```

### Interpretation

It mimics human decision-making by traversing a flowchart of conditions from the root to a leaf node.

### Example

Diagnosing Heart Disease by first checking Chest Pain type, then Checking Age, then Checking Max Heart Rate.

---

# Gini Impurity & Information Gain

## Definition

The mathematical criteria the tree uses to decide *which* feature to split on.

### Why It Matters

The tree calculates Gini Impurity for every possible split. The split that causes the largest drop in impurity (Information Gain) is chosen.

### Example

A node with 50 Cats and 50 Dogs has High Impurity. A split that creates one node of 50 Cats and another of 50 Dogs has Zero Impurity (perfect split).

---

# Hyperparameter Pruning

## Definition

Artificially stopping the tree from growing indefinitely to prevent overfitting.

### Example

Setting `max_depth = 4` forces the tree to stop asking questions after 4 levels, ensuring the leaves remain generalized.

### Visualization explanation

Cutting off the deep, sprawling branches of a massively complex tree, leaving only the thick, generalized trunk and main branches.

---

# Mathematical Intuition

Gini Impurity is calculated as `1 - Σ(p_i)²`. If a node contains only one class, `p_1 = 1`, making Gini exactly 0. The algorithm uses a greedy approach, always picking the mathematically purest immediate split without looking ahead to future depths.

---

# How It Works

Step-by-step process.

```text
Start at Root Node with all data
↓
Calculate Information Gain for all features
↓
Split data using the best feature
↓
Repeat recursively until nodes are pure or max_depth is reached
```

---

# Advantages

- Highly interpretable and easily visualized.
- Requires absolutely NO feature scaling (StandardScaler is useless here).
- Handles non-linear relationships natively.

---

# Limitations

- Extremely prone to high-variance overfitting if left unpruned.
- Unstable: A tiny change in the training data can completely alter the entire structure of the tree.
- Creates rigid, orthogonal decision boundaries (stair-step patterns) rather than smooth lines.

---

# Common Mistakes

- Forgetting to set `max_depth` or `min_samples_leaf`, resulting in a tree with 10,000 leaves that memorizes the data.
- Standardizing the data before fitting a tree (wastes processing time, does nothing).

---

# Practical Interpretation

Explain how to interpret outputs.

Examples:

> What does the feature at the Root Node mean?
The very first split in the tree is mathematically the single most important predictor in the entire dataset.

> When should this method be used?
When business stakeholders require strict explainability (e.g., denying a loan and needing to explain exactly why).

> When should it not be used?
When you need extreme accuracy and explainability isn't required (use XGBoost instead).

---

# Industry Applications

Examples:

- Healthcare: Heart Disease Prediction where doctors must validate the logic.
- Credit Scoring: Loan approval flowcharts to ensure compliance with financial regulations.

---

# Quick Comparison

| Concept | Meaning |
|----------|---------|
| Root Node | The top of the tree, encompassing all data |
| Leaf Node | The end points of the tree containing the final predictions |
| Pruning | Restricting tree growth to prevent overfitting |

---

# Resources

- StatQuest — "Decision Trees", "Information Gain"
- Google ML Crash Course — Decision Forests (intro)
