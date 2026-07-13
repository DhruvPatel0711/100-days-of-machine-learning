# Orientation & SQL Notes

## Why This Topic Matters

Machine Learning allows systems to learn from data without explicit programming, shifting us from rule-based programming to data-driven solutions. SQL is the foundational language for extracting and preparing that data.

Before you can train a model, you must find, filter, and aggregate the raw data. SQL is how you talk to the database to get exactly what your model needs.

---

# Supervised vs Unsupervised Learning

## Definition

The two primary paradigms of machine learning. Supervised learning predicts a known label from features. Unsupervised learning finds hidden structure in unlabeled data.

### Formula (if applicable)

```text
Supervised: y = f(X)
Unsupervised: Structure = f(X)
```

### Interpretation

If you know what you are looking for (e.g., House Price), use Supervised. If you are just exploring (e.g., Customer Groups), use Unsupervised.

### Example

Supervised: Predicting if an email is Spam based on its text.
Unsupervised: Grouping similar news articles without knowing their topics beforehand.

---

# Features and Labels

## Definition

Features (X) are the inputs. Labels (y) are the outputs we want to predict.

### Why It Matters

A model is only as good as the features you provide it. Garbage in, garbage out.

### Example

Predicting House Prices:
Features: Square footage, Number of bedrooms, Zip code.
Label: Price ($).

---

# SQL Aggregations

## Definition

Grouping rows that have the same values and calculating summary statistics (COUNT, SUM, AVG) on them.

### Example

Finding the average salary per department.

### Visualization explanation

Imagine a spreadsheet where all rows with "Sales" are bundled together, and their salary column is averaged into a single new row.

---

# Mathematical Intuition

The loss function mathematically measures how wrong a model's prediction is compared to the actual label. The goal of all ML algorithms is to minimize this loss function through optimization.

---

# How It Works

Extracting data using SQL:

```text
SELECT features
↓
FROM table
↓
WHERE condition
↓
GROUP BY category
```

---

# Advantages

- SQL is universally standard across data engineering and analytics.
- Supervised learning provides clear, measurable accuracy metrics.
- Unsupervised learning discovers patterns humans cannot see.

---

# Limitations

- SQL can be slow on extremely large, non-indexed tables.
- Supervised learning requires massive amounts of historically labeled data.
- Unsupervised learning results can be subjective and hard to validate.

---

# Common Mistakes

- Forgetting to handle missing values before feeding SQL extracts to a model.
- Confusing features with labels (putting the answer in the training data).
- Using `SELECT *` on a billion-row table and crashing the database.

---

# Practical Interpretation

Explain how to interpret outputs.

Examples:

> What does a high loss value mean?
It means your model's predictions are very far away from the actual true labels.

> When should SQL be used?
Always. It is the first step in the pipeline to query raw data.

> When should unsupervised learning not be used?
When you have a specific target variable you need to predict (e.g., Fraud).

---

# Industry Applications

Examples:

- Finance: SQL queries to aggregate daily transaction volumes.
- Healthcare: Supervised learning to predict patient readmission rates.
- E-commerce: Unsupervised learning to segment shoppers by purchasing habits.

---

# Quick Comparison

| Concept | Meaning |
|----------|---------|
| Supervised | Learning with a known answer key (y) |
| Unsupervised | Finding patterns without an answer key |
| SQL JOIN | Combining data from multiple tables |

---

# Resources

- Google ML Crash Course — Intro to ML, Framing, Descending into ML
- StatQuest — "What is Machine Learning?"
- SQLZoo
