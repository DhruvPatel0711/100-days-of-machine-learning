# 🚀 Customer Segmentation

Clustering mall customers into distinct behavioral segments to drive targeted marketing strategies.

Dataset:
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

---

# 🎯 Objective

Group unlabeled retail customers based on age, income, and spending habits to discover hidden, highly profitable business segments.

It matters because in the real world, businesses rarely have an "Answer Key" (labels). Unsupervised learning discovers actionable structure in raw data.

- KMeans
- Agglomerative Clustering
- DBSCAN
- Principal Component Analysis (PCA)

---

# 📊 Dataset

Mall Customers

## Features

| Feature | Description |
|--------|-------------|
| Age | Customer age |
| Income | Annual income (k$) |
| Spending | Spending score assigned by the mall (1-100) |
| Gender | Male or Female |

## Target

```text
Unsupervised Learning - No Target Column
```

---

# 🔍 Exploratory Data Analysis

Performed:

- Missing values and duplicate checks
- Feature distributions (histograms with KDE)
- Correlation heatmap (Age vs Spending shows negative correlation)
- Pairplots (colored by Gender)

### Key Observations

- Income and Spending distributions show distinct multi-modal behavior, strongly hinting at the presence of clusters.
- Younger customers tend to have significantly higher spending scores regardless of their income.

---

# 🛠 Feature Engineering

## Age & Spending Groups
Created discrete categories for Age (`GenZ/Young`, `Millennial/Adult`, etc.) and Spending (`Low_Spend`, `Medium_Spend`, `High_Spend`) to aid in cross-tabulation.

## Income / Spending Ratio
Calculated to identify customers who spend a disproportionately high amount of their income.

## High Value Indicators (Binary)
Engineered flags for `Young_High_Spenders` and `Premium_Customers` to easily filter these critical segments.

## Spending Percentiles
Ranked customers by spending to calculate their precise percentile within the mall's ecosystem.

---

# ⚙️ Data Preprocessing

## Feature Scaling
Compared `StandardScaler`, `MinMaxScaler`, and `No Scaling`. Because clustering relies on spatial distance (Euclidean), scaling is 100% required. We proceeded with StandardScaler.

## Finding 'k' (The optimal number of clusters)
To prevent guessing, we evaluated `k` from 2 to 10 using four distinct mathematical metrics:
- Elbow Method (WCSS)
- Silhouette Score
- Calinski-Harabasz Score
- Davies-Bouldin Score

**Optimal `k` identified as 5.**

---

# 🤖 Models Implemented

- **KMeans** (Primary model used for final cluster assignments)
- **Agglomerative Clustering** (Hierarchical baseline)
- **DBSCAN** (Density-based baseline to detect outliers/noise)

---

# 📉 Dimensionality Reduction (PCA)

Applied Principal Component Analysis (PCA) to crush the high-dimensional data into 2 components for human-readable visualization.

- **Variance Explained:** The first 2 components successfully explained over 75% of the total variance in the dataset.
- Created an **Explained Variance Curve** and a beautiful **2D PCA Cluster Scatter Plot**.

---

# 📊 Cluster Visualizations

Included:
- 4-Panel Metric Evaluation (Elbow, Silhouette, CH, DB)
- PCA 2D Scatter Plot
- Cluster Averages Heatmap
- Customer Segment Size (Bar Chart)
- **Radar Chart:** Visualizing the multi-dimensional profile of each cluster on a 0-1 normalized scale.

---

# 💼 Business Interpretation & Marketing Strategy

The primary focus of this project was translating math into business action. Based on the averages, the 5 clusters were labeled and assigned the following strategies:

### 1. Loyal High Value (High Income, High Spend)
* **Strategy:** Retention & Exclusivity. Invite them to VIP events and offer personal shoppers. No discounts.

### 2. High Income Low Spend (High Income, Low Spend)
* **Strategy:** Conversion & Upselling. Investigate why they churn and target with premium ad campaigns.

### 3. Young Premium Buyers (Low Income, High Spend)
* **Strategy:** Impulse Buying & Engagement. Target with flash sales and social media (FOMO) marketing.

### 4. Budget Customers (Low Income, Low Spend)
* **Strategy:** Volume & Value. Target with clearance sales and loyalty programs for frequent small purchases.

### 5. Standard Middle Class (Average Income, Average Spend)
* **Strategy:** Habit Forming. Send standard seasonal promotions to keep the brand top-of-mind.

---

# 🧠 Key Learnings

- k-Means is highly sensitive to feature scaling; unscaled data ruins the distance calculations.
- The Elbow curve is subjective; pairing it with Silhouette and Calinski-Harabasz scores provides a mathematically objective 'k'.
- PCA components are uninterpretable (PC1 has no human meaning), but they are essential for plotting clusters in 2D.

---

# 🛠 Tech Stack

- Python
- Scikit-Learn
- Matplotlib
- Seaborn
- Plotly (For interactive Radar Charts)

---

# Author

Built as part of my **100 Days of Machine Learning**