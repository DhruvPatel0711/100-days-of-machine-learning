# Week 1: SQL & Data Extraction Fundamentals

> *“A machine learning model is only as good as the data it is fed.”*

## Overview
Welcome to the first week of the **100 Days of ML** journey! Before training complex algorithms or tuning hyperparameters, a data scientist must know how to acquire, filter, and structure raw data. 

This week serves as the foundational block of the repository, focusing entirely on **Data Extraction**. By mastering SQL and understanding the theoretical paradigms of machine learning, we set the stage for building robust predictive pipelines in the coming weeks.

---

## 🎯 Core Learning Objectives

This module is designed to cement the core terminology and workflows required before any machine learning code is written:

1. **Paradigms of Machine Learning:** 
   - Understanding **Supervised Learning** (predicting a known label/target).
   - Understanding **Unsupervised Learning** (discovering hidden structures in unlabeled data).
2. **Data Terminology:** 
   - Identifying **Features ($X$)** vs. **Labels ($y$)** in a relational dataset.
3. **Relational Data Extraction:** 
   - Utilizing SQL as the industry-standard language to pull data from data warehouses and serve it to Python (`pandas`).

---

## 🛠️ The Sandbox Environment (`notebook.ipynb`)

The primary sandbox for this week is `notebook.ipynb`. This Jupyter Notebook bridges the gap between traditional SQL databases and Python's data science ecosystem.

### Workflow
Instead of relying on a live external server, the notebook demonstrates how to build an in-memory SQL database using the `sqlite3` library. 

1. **Data Ingestion:** Loads the `netflix_titles.csv` dataset into a `pandas` DataFrame.
2. **Database Creation:** Pushes the DataFrame into a local, in-memory SQLite database.
3. **Query Execution:** Uses `pd.read_sql_query()` to execute SQL strings directly from Python, proving that you can query SQL and receive a formatted `pandas` DataFrame in return.

### Key Analytical Queries Demonstrated:
- **Filtering:** Extracting movies released after 2020.
- **String Matching:** Searching for specific directors (e.g., David Fincher) or cast members using `LIKE`.
- **Conditional Logic:** Using `CASE WHEN` to categorize TV shows into 'SHORT', 'MODERATE', and 'LONG' bins based on season length.
- **Aggregation:** Grouping records to see how many TV shows fall into each specific length category.

---

## 📂 Standalone SQL Scripts

While the notebook connects SQL to Python, this directory also contains raw, standalone SQL scripts demonstrating fundamental database operations. 

These scripts are fully runnable and include mock `CREATE TABLE` and `INSERT INTO` commands, allowing you to test the logic in any SQL sandbox:

### `basic_queries.sql`
Focuses on the absolute basics of ML data preparation, explicitly demonstrating the theories from the notes:
- **Concept 1 (Supervised Learning):** Separating known labels ($y$) from features ($X$) (e.g., spam detection).
- **Concept 2 (Unsupervised Learning):** Extracting features without a label column (e.g., article grouping) for clustering algorithms to consume.
- **Concept 3 (Features and Labels):** Selecting clean features and labels while explicitly isolating and filtering out `NULL` values before they crash an ML algorithm.

### `advanced_joins.sql`
Focuses on building comprehensive datasets from fragmented relational tables, mapped to core aggregation techniques:
- **Concepts 1-3 (`COUNT`, `SUM`, `AVG`):** Performing baseline statistical summaries per group (e.g., total employees, payroll burden, and average salary per department).
- **Concept 4 (Combined Aggregation):** Bundling multiple aggregate functions together to compress rows into a single summary output.
- **Concept 5 (`JOIN`):** Merging fragmented relational tables into a single, flat structure (the required format for predictive modeling).
- **Concept 6 (`JOIN` + `GROUP BY`):** Aggregating and summarizing statistical features across multiple joined tables.

---

## ⚙️ Next Steps

With the theoretical foundations laid and data extraction tools configured (as detailed in `environment_setup.md`), the pipeline is ready. 

In the subsequent weeks, we will transition from extracting data to actively passing this data into Scikit-Learn models, starting with Linear Regression.
