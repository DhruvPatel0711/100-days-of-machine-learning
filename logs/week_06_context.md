# RAW CONTEXT AGGREGATION: WEEK 6
This document contains exhaustive technical, theoretical, and applied context extracted directly from the repository's notes, notebooks, and READMEs. Do NOT treat this as an image prompt. Treat this as the semantic knowledge base to deeply understand the topic before generating visuals.

## Part 1: The Hook (Topic Introduction)
### Learning Goal
- **What concept is this notebook teaching?** Advanced Evaluation Metrics (Precision, Recall, ROC-AUC).
- **What pattern is visible?** The ROC Curve.
- **What should the reader learn?** A model guessing randomly follows the diagonal line. A perfect model hugs the top left corner.
- **What are we changing?** We are evaluating an imbalanced dataset using Accuracy vs Recall.
- **What do we expect to happen?** Accuracy will look amazing (99%), but Recall will reveal the model is failing.
# Update layout and print macro metrics in the labels or title
## Why This Topic Matters
## Definition
> What does a high Precision but low Recall mean?
### Problem Definition
- **What problem are we solving?** Detecting fraudulent credit card transactions.
- **What type of ML problem is this?** Supervised Classification (Highly Imbalanced).
- **What information does this dataset contain?** Anonymized PCA-transformed transaction features and transaction amounts.
- **What is the target?** Class (0 = Legit, 1 = Fraud).
- **What assumptions can we make before analysis?** The dataset will be massively imbalanced (99.9% legit, 0.1% fraud).
- **What should we learn from EDA?** We must verify the exact ratio of fraud to legit transactions.

================================================================================

## Part 2: The Problem (Why does it matter?)
- **Why does it matter?** Accuracy is a dangerous lie when dealing with imbalanced data.
# Isolate a hold-out test set to provide an unbiased evaluation of generalization error and detect data leakage.
# Calculate the harmonic mean of Precision and Recall. Highly preferred over Accuracy when evaluating imbalanced datasets.
- **Common Mistakes:** Using ROC-AUC to evaluate models on severely imbalanced datasets. (Use PR-AUC instead).
# Imbalanced Data
*(Shows why Accuracy is misleading if TN is 99% of the Total).*
A metric curve that replaces ROC-AUC when dealing with highly imbalanced data. It plots Precision vs Recall across thresholds.
### Why It Matters
Raw Imbalanced Data
- Pipelines completely eliminate data leakage during cross-validation.
- Scaling the entire dataset *before* performing the train-test split (Data Leakage).
- Using ROC-AUC instead of PR-AUC for a 99-to-1 imbalanced dataset.
| PR-AUC | Best for imbalanced datasets (ignores TNs entirely) |
- 100 Days of ML Code — classification problems
- **Why is this problem important?** Fraud costs banks billions. ML flags anomalies instantly.
- **Is the target balanced?** Extremely imbalanced.
- **Why choose this algorithm?** Algorithms like Random Forest handle imbalanced data well because they partition the feature space, isolating the rare fraud cases into specific leaf nodes.
- **Why these metrics?** Precision-Recall Area Under Curve (PR-AUC) and F1-Score.
- **Which one matters most?** ROC-AUC is misleading on highly imbalanced data. PR-AUC and F1-Score directly punish the model for False Positives (declaring a legit transaction as fraud, angering the customer).
- **Biggest takeaway:** Accuracy is a completely useless metric when dealing with 99% imbalanced datasets.
Detecting fraudulent credit card transactions in a highly imbalanced dataset.
- The dataset is massively imbalanced (99.8% Class 0).
## Missing Values
No missing values present.
- Accuracy is a deceptive metric on imbalanced datasets.
- Imbalanced-Learn

================================================================================

## Part 3: The Experiment & Logic (Code & Mechanism)
### Visualization Interpretation
### Visualization Analysis: Correlation Heatmap
- **Common Mistakes:** Assuming correlation implies causation. Also, failing to drop one of two highly correlated features (e.g., dropping 'Tax' if it correlates 0.95 with 'Price') which confuses linear models.
### Experiment
# Encapsulate sequential data transformations and the final estimator to guarantee identical preprocessing during inference.
# Standardize features by removing the mean and scaling to unit variance. Crucial for distance-based algorithms (e.g., kNN, SVM).
# Fit a linear model mapped through a sigmoid function to output calibrated class probabilities.
# Find the optimal maximum-margin hyperplane that separates classes in a high-dimensional feature space.
# Evaluate model robustness by partitioning data into k folds, preventing the model from just getting "lucky" on a single split.
### Visualization Analysis: Confusion Matrix
- **Correct Interpretation:** A highly accurate model will have very dark/high numbers on the main diagonal and zeros elsewhere.
# 1. Keep the full 2D probability matrices (do NOT slice with [:, 1] or [:, :])
# 2. Get the unique classes from your target data (Iris has 3 classes: 0, 1, 2)
# 3. Binarize the true labels for multi-class ROC curve evaluation
# 4. Compute the multi-class macro ROC AUC scores directly using the full 2D matrices
# Quantify the model's overall ability to discriminate between classes across all possible classification thresholds.
# --- Visualise the Multiclass ROC Curve ---
# Define colors for the 3 distinct Iris classes
# 5. Loop through each class and plot individual One-vs-Rest curves
# Logistic Regression curves per class
# SVM curves per class
# Plot reference elements
### Visualization Analysis: ROC Curve & AUC
- **Marking Values:** The critical value is the AUC (Area Under Curve) score. 0.5 is random guessing, 1.0 is a perfect model. Look at where the curve 'bows' towards the top-left (True Positive Rate = 1, False Positive Rate = 0).
- **Correct Interpretation:** The steeper the curve shoots upward before bending right, the better the model is at separating classes at various probability thresholds.
# 1. Keep the full 2D probability matrices (do NOT slice with [:, 1])
# 2. Binarize labels for the 3 Iris classes
# 3. Loop through each class to calculate and plot PR curves separately
# Calculate baseline for this specific class (how frequent it is in y_test)
# Logistic Regression PR Metrics
# SVM PR Metrics
# Optional: Plot the random baseline line for each class (drawn lightly)
### Comparison
### Common Mistakes
### Practical Takeaway
- **Industry Application:** Medical models optimize for Recall (catch every cancer case). YouTube algorithms optimize for Precision (only recommend videos the user will definitely click).
### Key Insight
# Classification Pipeline Notes
A model predicting "No Fraud" every time gets 99.8% accuracy, but is utterly useless. We need Pipelines to handle complex transformations and specialized metrics to handle severe imbalance.
### Formula (if applicable)
### Interpretation
The model naturally biases toward the majority class because predicting it yields the lowest mathematical loss.
### Example
# Precision-Recall Curve (PR-AUC)
A PR-AUC of 0.85 means the model maintains high precision even as you lower the threshold to capture more fraud (recall).
# Sklearn Pipeline
An object that chains together multiple preprocessing steps and a final estimator into a single, cohesive workflow.
`Pipeline([ ('scaler', StandardScaler()), ('model', LogisticRegression()) ])`
### Visualization explanation
A factory conveyor belt where raw data enters, gets scaled automatically, and feeds directly into the model for prediction without manual intervention.
# Mathematical Intuition
Algorithms like Logistic Regression can be mathematically forced to care about the minority class by applying `class_weight='balanced'`. This artificially multiplies the loss penalty for getting a minority class wrong by the inverse of its frequency, forcing the optimizer to pay attention to it.
# How It Works
ColumnTransformer (Scale numbers, Encode categories)
Pipeline passes processed data to Model
Model optimized using PR-AUC
# Advantages
- PR-AUC provides a truthful assessment of model performance on rare events.
- Class weighting allows standard algorithms to handle imbalance naturally.
# Limitations
- Pipelines can be tricky to debug if a specific step fails silently.
# Common Mistakes
# Practical Interpretation
When the model flags fraud, it is almost always right. But it misses a lot of actual fraud that slips through.
When building production-ready code (Pipelines) for rare-event detection (Fraud, Disease, Manufacturing Defects).
Pipelines should ALWAYS be used. Never write raw fit/transform spaghetti code in production.
# Industry Applications
# Quick Comparison
# Resources
### Dataset
### Exploratory Data Analysis
### Visualization Analysis: General Plot Interpretation
### Data Preprocessing
- **Class Balancing:** Because the data is so skewed, the model will just predict "Legit" every time and get 99.9% accuracy. We must use techniques like SMOTE (Synthetic Minority Over-sampling) or class weighting to force the model to care about the fraud cases.
### Model Selection
### Evaluation
### Conclusion
- **Possible improvements:** Deploying Anomaly Detection (unsupervised learning) as a pre-filter before the classification model.
# 🚀 Credit Card Fraud Detection
# 🎯 Objective
# 📊 Dataset
## Features
## Target
# 🔍 Exploratory Data Analysis
### Key Observations
# 🛠 Feature Engineering
## Amount Log-Transform
The 'Amount' feature is highly right-skewed. Taking the log normalizes it, allowing linear models to process it effectively without being thrown off by massive purchases.
## Time of Day
# ⚙️ Data Preprocessing
## Encoding
## Feature Scaling
RobustScaler applied to 'Amount' and 'Time' because it is robust to extreme outliers, which are common in fraud data.
## Train-Test Split
# 🤖 Models Implemented
Chosen to benchmark how baseline algorithms handle severe class imbalances.
# 📈 Evaluation Metrics
# 📊 Results
## Performance Comparison
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
# 📉 Visualizations
# 📌 Interpretation
- High accuracy (99%) is a lie. The model predicts 'Normal' for everything.
# 🧠 Key Learnings
# 🛠 Tech Stack
# 🚀 Future Improvements
# Author

================================================================================

## Part 4: The Observation & Visualization (Results)
- **Common Mistakes:** Caring only about the total sum of the diagonal (Accuracy) while ignoring a high number of False Negatives in medical or fraud datasets.
- **Beginner Mistake:** Presenting 99% accuracy on a fraud detection dataset as a success.
Accuracy = (TP + TN) / Total
ROC-AUC is artificially inflated by the massive number of True Negatives. The PR-Curve ignores True Negatives and focuses entirely on the minority positive class.
- Using Accuracy as the primary metric for fraud detection.
- **Correct Interpretation:** Visualizations bridge the gap between raw mathematical outputs and human intuition. Look for structural patterns that confirm or deny your hypotheses.
It matters because fraud causes billions in losses, but normal transactions outnumber fraud 99.8% to 0.2%, making standard accuracy metrics useless.
- F1-Score
- Precision-Recall Curve
- Confusion Matrix Heatmap
- ROC Curve
- Precision-Recall curves are far superior to ROC curves when the positive class is extremely rare.

================================================================================

## Part 5: Key Insight & Practical Takeaway
- **Marking Values:** Look specifically for values approaching 1.0 (strong positive correlation) or -1.0 (strong negative correlation) against the target variable.
- **Correct Interpretation:** Features with high absolute correlation to the target are your strongest predictors. Features highly correlated with *each other* indicate multi-collinearity.
- **How to Interpret:** Darker or more intense colors represent stronger mathematical relationships. A value of 0 means zero linear relationship.
- **Marking Values:** The top-left to bottom-right diagonal contains all correct predictions. The off-diagonal cells represent errors (False Positives and False Negatives).
- **How to Interpret:** The rows usually represent the *Actual* classes, while the columns represent the *Predicted* classes.
- **How to Interpret:** The X-axis is the False Positive Rate (cost). The Y-axis is the True Positive Rate (benefit). We want maximum benefit for minimum cost.
- **Before vs After:** Changing the threshold from 0.5 to 0.3 increased Recall but dropped Precision.
- **Memorable lesson:** Precision is about quality. Recall is about quantity.
Real-world datasets are rarely clean and rarely balanced. E-commerce fraud might represent only 0.2% of transactions.
---
A dataset where the target classes are not represented equally (e.g., 99% Class 0, 1% Class 1).
```text
```
Credit Card Fraud Detection. Normal transactions vastly outnumber fraudulent ones.
Step-by-step process.
↓
- SMOTE (synthetic oversampling) can sometimes create unrealistic "Frankenstein" data points.
- Precision and Recall are a tradeoff; you usually cannot maximize both.
Explain how to interpret outputs.
Examples:
> When should this method be used?
> When should it not be used?
- Finance: Credit Card Fraud Detection (minimizing false negatives).
- Cybersecurity: Detecting network intrusions and malware signatures.
- Manufacturing: Identifying defective parts on an assembly line.
| Concept | Meaning |
|----------|---------|
| ROC-AUC | Best for balanced datasets (looks at FPR) |
| SMOTE | Synthetically generating new minority class rows |
- IBM ML with Python — Classification module
- **Marking Values:** Always identify the max/min peaks, intersections, and the general trend line (upward, downward, or flat).
- **How to Interpret:** Read the axes first. The X-axis is the independent variable, the Y-axis is the dependent reaction.
- **Common Mistakes:** Producing a beautiful graph but failing to extract a single actionable business or engineering decision from it.
Dataset:
https://www.kaggle.com/mlg-ulb/creditcardfraud
Identify fraudulent transactions.
- Logistic Regression
- SVM
Credit Card Fraud
| Time | Seconds elapsed between transaction and first transaction |
| V1-V28 | PCA reduced features (anonymized) |
| Amount | Transaction amount |
Target Classes (Classification)
Class 0 = Normal
Class 1 = Fraud
Performed:
- Target distribution
- Distribution analysis
- Imbalance severity check
- Fraudulent transactions tend to have smaller, very specific amounts.
- V-features are already PCA transformed.
Describe every engineered feature.
```python
df['log_amount'] = np.log1p(df['Amount'])
Purpose:
Converted the continuous 'Time' variable (seconds) into a cyclic 'Time of Day' (hours), as fraud is often highly correlated with specific late-night hours.
No encoding needed as V1-V28 are continuous numerical vectors.
80% Training
20% Testing
- SVM (Calibrated)
Used:
- Precision
- Recall
- ROC-AUC
|--------|----------|-----------|--------|----|---------|
| Logistic Regression | - | 0.0547 | 0.8737 | 0.1030 | 0.9844 |
| Linear SVM (Calibrated) | - | 0.8548 | 0.5579 | 0.6752 | 0.9838 |
Included:
Discuss:
- By tuning thresholds or using class_weight='balanced', Logistic Regression catches 87% of fraud (High Recall), but sacrifices Precision (many false alarms). Linear SVM favors precision.
- Business tradeoff: Is a blocked legitimate transaction worse than a missed fraudulent one?
- Python
- Pandas
- Scikit-Learn
- SMOTE (Synthetic Minority Over-sampling)
- XGBoost or Isolation Forests
Built as part of my **100 Days of Machine Learning**

================================================================================

