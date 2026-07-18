# RAW CONTEXT AGGREGATION: WEEK 1
This document contains exhaustive technical, theoretical, and applied context extracted directly from the repository's notes, notebooks, and READMEs. Do NOT treat this as an image prompt. Treat this as the semantic knowledge base to deeply understand the topic before generating visuals.

## Part 1: The Hook (Topic Introduction)
### Learning Goal
- **What concept is this notebook teaching?** Python, Numpy, and Pandas basics for Data Science.
## Why This Topic Matters
Before you can train a model, you must find, filter, and aggregate the raw data. SQL is how you talk to the database to get exactly what your model needs.
## Definition
If you know what you are looking for (e.g., House Price), use Supervised. If you are just exploring (e.g., Customer Groups), use Unsupervised.
Unsupervised: Grouping similar news articles without knowing their topics beforehand.
The loss function mathematically measures how wrong a model's prediction is compared to the actual label. The goal of all ML algorithms is to minimize this loss function through optimization.
> What does a high loss value mean?
- StatQuest — "What is Machine Learning?"

================================================================================

## Part 2: The Problem (Why does it matter?)
- **Why does it matter?** You cannot build ML models if you cannot manipulate data arrays.
### Why It Matters
- Forgetting to handle missing values before feeding SQL extracts to a model.

================================================================================

## Part 3: The Experiment & Logic (Code & Mechanism)
# df.info()
### Common Mistakes
### Practical Takeaway
- **Where does this concept appear in industry?** Every single ML pipeline begins with Pandas/Numpy data wrangling.
### Key Insight
# Orientation & SQL Notes
# Supervised vs Unsupervised Learning
### Formula (if applicable)
### Interpretation
### Example
# Features and Labels
A model is only as good as the features you provide it. Garbage in, garbage out.
Features: Square footage, Number of bedrooms, Zip code.
# SQL Aggregations
### Visualization explanation
# Mathematical Intuition
# How It Works
# Advantages
# Limitations
# Common Mistakes
# Practical Interpretation
It means your model's predictions are very far away from the actual true labels.
Always. It is the first step in the pipeline to query raw data.
# Industry Applications
# Quick Comparison
# Resources

================================================================================

## Part 4: The Observation & Visualization (Results)
- Supervised learning provides clear, measurable accuracy metrics.
- Unsupervised learning results can be subjective and hard to validate.

================================================================================

## Part 5: Key Insight & Practical Takeaway
- **Beginner Mistake:** Iterating through rows in a Pandas DataFrame using `.iterrows()` instead of using vectorized column operations.
- **Memorable lesson:** Never use a for-loop on a dataframe if a vectorized function exists.
Machine Learning allows systems to learn from data without explicit programming, shifting us from rule-based programming to data-driven solutions. SQL is the foundational language for extracting and preparing that data.
---
The two primary paradigms of machine learning. Supervised learning predicts a known label from features. Unsupervised learning finds hidden structure in unlabeled data.
```text
Supervised: y = f(X)
Unsupervised: Structure = f(X)
```
Supervised: Predicting if an email is Spam based on its text.
Features (X) are the inputs. Labels (y) are the outputs we want to predict.
Predicting House Prices:
Label: Price ($).
Grouping rows that have the same values and calculating summary statistics (COUNT, SUM, AVG) on them.
Finding the average salary per department.
Imagine a spreadsheet where all rows with "Sales" are bundled together, and their salary column is averaged into a single new row.
Extracting data using SQL:
SELECT features
↓
FROM table
WHERE condition
GROUP BY category
- SQL is universally standard across data engineering and analytics.
- Unsupervised learning discovers patterns humans cannot see.
- SQL can be slow on extremely large, non-indexed tables.
- Supervised learning requires massive amounts of historically labeled data.
- Confusing features with labels (putting the answer in the training data).
- Using `SELECT *` on a billion-row table and crashing the database.
Explain how to interpret outputs.
Examples:
> When should SQL be used?
> When should unsupervised learning not be used?
When you have a specific target variable you need to predict (e.g., Fraud).
- Finance: SQL queries to aggregate daily transaction volumes.
- Healthcare: Supervised learning to predict patient readmission rates.
- E-commerce: Unsupervised learning to segment shoppers by purchasing habits.
| Concept | Meaning |
|----------|---------|
| Supervised | Learning with a known answer key (y) |
| Unsupervised | Finding patterns without an answer key |
| SQL JOIN | Combining data from multiple tables |
- Google ML Crash Course — Intro to ML, Framing, Descending into ML
- SQLZoo

================================================================================

