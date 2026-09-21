# k-Nearest Neighbors (kNN) Notes

## Why This Topic Matters

Not all algorithms build a mathematical equation or a tree. What if we just memorized the data and compared new data to it spatially?

kNN is the quintessential instance-based learning algorithm, providing a masterclass in how ML models interpret spatial geometry and why distance calculations require strict data preprocessing.

---

# Spatial Distance

## Definition

kNN classifies a new data point based on the majority class of its 'k' closest neighbors in the N-dimensional feature space.

### Formula (if applicable)

```text
Euclidean Distance: √((x₂ - x₁)² + (y₂ - y₁)²)
```

### Interpretation

"Tell me who your friends are, and I'll tell you who you are."

### Example

Predicting Wine Quality. A new wine has chemicals identical to 5 existing wines. If 4 of those 5 are "Good", the new wine is predicted "Good".

---

# Feature Scaling Criticality

## Definition

kNN relies 100% on distance. Therefore, all features MUST be scaled to the same range (StandardScaler or MinMaxScaler).

### Why It Matters

If 'Alcohol' ranges from 8 to 15, but 'Sulfur Dioxide' ranges from 10 to 200, the Euclidean distance formula will ONLY care about Sulfur Dioxide. Alcohol will be mathematically ignored.

### Example

Removing `StandardScaler` causes kNN accuracy to instantly collapse because the algorithm becomes blinded by large-magnitude features.

---

# Choosing 'k'

## Definition

The hyperparameter dictating how many neighbors to look at.

### Example

k=1: Looks only at the single closest point (High Variance / Overfits to noise).
k=100: Looks at a massive radius, blurring boundaries (High Bias / Underfits).

### Visualization explanation

An "Elbow Curve" plotting Accuracy vs 'k'. Accuracy rises as k increases, peaks (the sweet spot), and then slowly degrades as k gets too large.

---

# Mathematical Intuition

kNN doesn't actually "train" a model. The training phase is simply memorizing the dataset in RAM. During prediction, it mathematically computes the hypotenuse (Euclidean distance) between the new point and every single point in memory, then sorts them to find the top k shortest distances.

---

# How It Works

Step-by-step process.

```text
Load training data into memory
↓
Receive new unseen data point
↓
Calculate distance to EVERY point in training data
↓
Select top 'k' closest points (majority vote wins)
```

---

# Advantages

- Zero training time (it just memorizes data).
- Intuitive and easily explainable to non-technical stakeholders.
- Can create highly non-linear, complex decision boundaries.

---

# Limitations

- Extremely slow at prediction time on large datasets (must calculate millions of distances).
- Massive memory footprint (entire dataset must stay in RAM).
- Suffers heavily from the Curse of Dimensionality.

---

# Common Mistakes

- Forgetting to scale features, rendering the model completely useless.
- Choosing an even number for 'k' in a binary classification problem (causing a 50/50 tie vote).
- Using kNN on datasets with hundreds of features.

---

# Practical Interpretation

Explain how to interpret outputs.

Examples:

> What does the Curse of Dimensionality mean?
As you add more features, the "volume" of the space grows exponentially, making all points mathematically far away from each other. Distance loses its meaning.

> When should this method be used?
On small to medium datasets where spatial similarity is the strongest indicator of class (e.g., Wine Quality chemical composition).

> When should it not be used?
In real-time low-latency systems with millions of rows.

---

# Industry Applications

Examples:

- Retail: Basic recommendation engines (Users who bought this also bought this).
- Quality Control: Wine Quality Classification based on chemical spatial proximity.
- Handwriting Recognition: Grouping similar pixel intensities.

---

# Quick Comparison

| Concept | Meaning |
|----------|---------|
| k=1 | Highly sensitive to outliers (Overfitting) |
| k=Optimal | Balances local nuance with global trends |
| k=All Data | Always predicts the majority class (Underfitting) |

---

# Resources

- StatQuest — "K-nearest neighbors", "Feature Scaling"
- IBM ML with Python — KNN module
