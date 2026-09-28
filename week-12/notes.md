# Clustering & PCA Notes

## Why This Topic Matters

Up to this point, we always had an "Answer Key" (Labels/y). But in the real world, businesses have raw data and no labels. They just want to know: "What patterns exist here?"

Unsupervised learning (like k-Means) discovers hidden segments, while Principal Component Analysis (PCA) crushes high-dimensional data down so humans can actually look at it.

---

# k-Means Clustering

## Definition

An algorithm that groups unlabeled data into 'k' distinct clusters based on spatial similarity.

### Formula (if applicable)

```text
Minimize WCSS (Within-Cluster Sum of Squares)
```

### Interpretation

It assigns each point to a cluster centroid, then moves the centroid to the center of those points, repeating until the clusters stabilize.

### Example

Grouping Mall Customers into distinct behavioral groups (e.g., High Income / Low Spend vs Low Income / High Spend) without explicitly defining those groups beforehand.

---

# The Elbow Method

## Definition

A visual technique to mathematically determine the optimal number of clusters (k).

### Why It Matters

Since there are no labels, we don't know what 'k' should be. WCSS measures how tight the clusters are. As k increases, WCSS drops. The "Elbow" is the point where adding more clusters stops providing significant tightness.

### Visualization explanation

A line graph plotting k (X-axis) against WCSS (Y-axis) that looks like a human arm. The bend in the elbow is the optimal k.

---

# Principal Component Analysis (PCA)

## Definition

A dimensionality reduction technique that creates new features (components) which capture the maximum variance of the original data.

### Example

Compressing Age, Income, and Spending Score (3D) into Principal Component 1 and Principal Component 2 (2D) so we can plot it on a flat screen.

### Visualization explanation

Taking a 3D cloud of data points and shining a flashlight on it. PCA finds the exact angle to hold the flashlight so the 2D shadow on the wall is as spread out and informative as possible.

---

# Mathematical Intuition

PCA uses eigenvectors and eigenvalues on the covariance matrix of the data to find orthogonal axes of maximum variance. k-Means uses standard Euclidean distance optimization to minimize intra-cluster variance while maximizing inter-cluster variance.

---

# How It Works

Step-by-step process.

```text
Scale the features (MinMaxScaler)
↓
Use Elbow Method to find optimal k
↓
Run k-Means to assign Cluster Labels
↓
Run PCA to reduce data to 2 dimensions for plotting
```

---

# Advantages

- k-Means is extremely fast and scales well to massive datasets.
- Discovers highly profitable business segments humans would miss.
- PCA removes highly correlated noise and speeds up other algorithms.

---

# Limitations

- k-Means assumes clusters are spherical and equally sized; it fails on complex geometric shapes (like concentric rings).
- You must manually specify 'k' in k-Means.
- PCA components are completely uninterpretable (PC1 has no human meaning).

---

# Common Mistakes

- Forgetting to scale features (k-Means uses distance, so scaling is critical).
- Assuming the clusters are the "truth" rather than just a mathematical partition.
- Using PCA and then trying to explain business logic using the PCA components.

---

# Practical Interpretation

Explain how to interpret outputs.

Examples:

> What does a high Silhouette Score mean?
It means points are very close to their own cluster center and very far away from neighboring clusters (dense, well-separated clusters).

> When should this method be used?
Customer Segmentation, Anomaly Detection, or creating labels for a dataset that doesn't have any.

> When should it not be used?
When you have a specific target you are trying to predict (use Supervised Learning).

---

# Industry Applications

Examples:

- Marketing: Customer Segmentation (Targeting ads to specific behavior clusters).
- Genomics: Grouping patients with similar genetic expressions.
- Image Compression: Using PCA to reduce image file size while retaining structure.

---

# Quick Comparison

| Concept | Meaning |
|----------|---------|
| k-Means | Groups data based on distance to centroids |
| Elbow Method | Finds optimal k using WCSS |
| PCA | Reduces dimensions by maximizing variance |

---

# Resources

- StatQuest — "k-Means Clustering", "Elbow Method", "PCA Step by Step"
- 3Blue1Brown — "Eigenvectors and Eigenvalues"
