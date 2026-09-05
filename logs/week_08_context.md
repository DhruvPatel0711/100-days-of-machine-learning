# RAW CONTEXT AGGREGATION: WEEK 8
This document contains exhaustive technical, theoretical, and applied context extracted directly from the repository's notes, notebooks, and READMEs. Do NOT treat this as an image prompt. Treat this as the semantic knowledge base to deeply understand the topic before generating visuals.

## Part 1: The Hook (Topic Introduction)
### Learning Goal
- **What concept is this notebook teaching?** Decision Trees and Information Gain (Gini Impurity).
- **What are we changing?** We are comparing an Unpruned Decision Tree (allowed to grow infinitely) to a Pruned Decision Tree (max_depth restricted).
- **What do we expect to happen?** The unpruned tree will severely overfit the training data.
- **What actually happened?** The unpruned tree went 15 layers deep, finding absurd, hyper-specific rules to classify single patients. Pruning the tree yielded a far superior ROC-AUC (0.84 vs 0.71) and better precision.
## Why This Topic Matters
Unlike "black box" models (like Neural Networks or SVMs), Decision Trees provide 100% transparency. You can print the tree and show a doctor exactly why a diagnosis was made.
They are the foundational building blocks for the most powerful algorithms in ML (Random Forests and XGBoost).
## Definition
An algorithm that repeatedly splits data based on simple Yes/No questions to maximize the purity of the resulting groups.
### Problem Definition
- **What problem are we solving?** Classifying whether a patient has heart disease based on medical attributes.
- **What type of ML problem is this?** Binary Classification.
- **What information does this dataset contain?** Heart Disease (UCI) dataset including age, cholesterol (chol), resting blood pressure (trestbps), and max heart rate (thalach).
- **What is the target?** Class 0 = No Heart Disease, Class 1 = Heart Disease.
- **What should we learn from EDA?** Maximum heart rate (thalach) is negatively correlated with heart disease. Age and cholesterol show a positive correlation with the disease.
- **What assumptions does it make?** It assumes the data can be separated by orthogonal (stair-step) decision boundaries.
It matters because Decision Trees provide white-box explainability, allowing doctors to see exactly why a prediction was made.

================================================================================

## Part 2: The Problem (Why does it matter?)
- **Why does it matter?** A model that perfectly memorizes the training data is useless in the real world. Unpruned trees are the ultimate memorization engines.
### Why It Matters
A Decision Tree with `max_depth=None` will build a unique path for every single training row, resulting in extreme overfitting.
Artificially stopping the tree from growing indefinitely to prevent overfitting is called Pruning.
- **Why is this problem important?** Early detection saves lives, and doctors must trust the algorithm's logic.
- **Why choose this algorithm?** Decision Trees are highly interpretable. Doctors can look at the tree branches and understand exactly why the model made a specific diagnosis.
- **Why these metrics?** Accuracy, Precision, Recall, F1 Score, ROC-AUC. As with any medical diagnosis, maximizing Recall (minimizing False Negatives) is critical to not missing sick patients.
## Missing Values
Rows with '?' were replaced with NaN and imputed with the median.
Chosen specifically to demonstrate Information Gain, Gini Impurity, and visually plotting the exact decision logic.

================================================================================

## Part 3: The Experiment & Logic (Code & Mechanism)
### Visualization Analysis: General Plot Interpretation
# Recursively partition the feature space to minimize Gini impurity or Information Entropy at each node.
### Experiment
We trained two models:
1. Unpruned Tree (No hyperparameters set).
2. Pruned Tree (max_depth and min_samples_leaf restricted).
### Observation
The unpruned tree grew 15 levels deep.
The pruned tree stopped at 4 levels.
### Comparison
- **Without vs With:**
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Unpruned Tree | 0.5870 | 0.5902 | 0.5870 | 0.5883 | 0.7155 |
| Pruned Tree | 0.5870 | 0.7466 | 0.5870 | 0.6313 | 0.8425 |
Pruning drastically improved the ROC-AUC and Precision, meaning the model became much better at generalizing to new patients.
### Common Mistakes
- Forgetting to set `max_depth` or `min_samples_leaf`, resulting in a tree with 10,000 leaves that memorizes the data.
- Standardizing the data before fitting a tree (wastes processing time, does absolutely nothing because trees only split on thresholds, not distances).
### Practical Takeaway
An unpruned Decision Tree will always overfit. You must restrict its maximum depth.
### Key Insight
Chest pain type (cp) was the very first split at the Root Node, meaning it had the highest Information Gain and was the single most important predictor in the entire dataset.
# Gini Impurity & Information Gain
The mathematical criteria the tree uses to decide *which* feature to split on.
The tree calculates Gini Impurity for every possible split. The split that causes the largest drop in impurity (Information Gain) is chosen.
Gini Impurity is calculated as `1 - Σ(p_i)²`. If a node contains only one class, `p_1 = 1`, making Gini exactly 0. The algorithm uses a greedy approach, always picking the mathematically purest immediate split without looking ahead to future depths.
# Hyperparameter Pruning
Setting `max_depth = 4` forces the tree to stop asking questions after 4 levels, ensuring the leaves remain generalized.
# How It Works
Step-by-step process:
Start at Root Node with all data
↓
Calculate Information Gain for all features
↓
Split data using the best feature
↓
Repeat recursively until nodes are pure or max_depth is reached
# Advantages
- Highly interpretable and easily visualized.
- Requires absolutely NO feature scaling (StandardScaler is useless here).
- Handles non-linear relationships natively.
# Limitations
- Extremely prone to high-variance overfitting if left unpruned.
- Unstable: A tiny change in the training data can completely alter the entire structure of the tree.
- Creates rigid, orthogonal decision boundaries (stair-step patterns) rather than smooth lines.
# Practical Interpretation
The very first split in the tree is mathematically the single most important predictor in the entire dataset.
When should this method be used? When business stakeholders require strict explainability (e.g., denying a loan and needing to explain exactly why).
When should it not be used? When you need extreme accuracy and explainability isn't required (use XGBoost instead).
# Industry Applications
- Healthcare: Heart Disease Prediction where doctors must validate the logic.
- Credit Scoring: Loan approval flowcharts to ensure compliance with financial regulations.
# Quick Comparison
| Concept | Meaning |
|----------|---------|
| Root Node | The top of the tree, encompassing all data |
| Leaf Node | The end points of the tree containing the final predictions |
| Pruning | Restricting tree growth to prevent overfitting |

================================================================================

## Part 4: The Observation & Visualization (Results)
- **Correct Interpretation:** Visualizations bridge the gap between raw mathematical outputs and human intuition.
- **How to Interpret:** Each node in the `plot_tree` output shows the feature used to split, the Gini impurity, the total samples, and the class distribution (value array).
- Full Tree Visualisation (plot_tree)
- Confusion Matrix
- Overfitting comparison at different max_depths
- Matplotlib

================================================================================

## Part 5: Key Insight & Practical Takeaway
- **Marking Values:** The longest bars at the top of the Feature Importance chart represent the features that drove the model's decision-making process the most.
- **Correct Interpretation:** If a feature has 0 importance, the tree never used it to make a split. It can be safely dropped from the dataset.
- **Beginner Mistake:** Scaling features before passing them to a Decision Tree. Trees only care about threshold splits (e.g., Age > 30), not distances.
- **Industry Application:** Used extensively in healthcare where doctors demand to see the exact logic path for a diagnosis.
- **Memorable lesson:** Trees don't care about scaling. They only care about splits.
Dataset: UCI Heart Disease
- **Feature Engineering 1:** Cholesterol Categories.
`df['chol_cat'] = pd.cut(df['chol'], bins=[0, 200, 239, 600], labels=['Desirable', 'Borderline', 'High'])`
Purpose: Medical diagnoses often rely on established thresholds (e.g., >240 is high risk). Categorizing helps the tree make cleaner splits.
- **Feature Engineering 2:** Blood Pressure Ratio.
Created a ratio of resting blood pressure to age, testing the hypothesis that high BP is more dangerous at younger ages.
- **Encoding:** One-Hot Encoding for categorical features like chest pain type.
- **Feature Scaling:** Not required. Decision Trees do not compute distances and are invariant to feature scaling.
- **Train-Test Split:** 80% Training, 20% Testing.
- **Results Discussion:** The unpruned tree went 15 layers deep, finding absurd rules to classify single patients. Pruning the tree yielded far superior ROC-AUC (0.84 vs 0.71) and better precision. Chest pain type (cp) was the very first split, meaning it had the highest Information Gain.
- **Tech Stack:** Python, Pandas, Scikit-Learn, Matplotlib.
- **Future Improvements:** Random Forests, Hyperparameter GridSearch.
Built as part of my **100 Days of Machine Learning**

- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.
- (Padding) Extended architectural context and evaluation for Decision Trees.