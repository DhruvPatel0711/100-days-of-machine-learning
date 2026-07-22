# 100 Days of ML

> A structured 14-week machine learning journey documented publicly â€” every concept learned, every project built, every week posted on LinkedIn.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Follow-blue?style=flat&logo=linkedin)](https://linkedin.com/in/dhruvpatel0711)

## The Roadmap

```
Phase 0  â†’  Orientation + SQL Basics          (Week 1)
Phase 1  â†’  Linear Regression + Ridge/Lasso   (Weeks 2â€“3)
Phase 2  â†’  Classification + SVM + Metrics    (Weeks 4â€“6)
Phase 3  â†’  Model Behavior                    (Week 7)
Phase 4  â†’  Tree Models + XGBoost             (Weeks 8â€“10)
Phase 5  â†’  Unsupervised + PCA                (Weeks 11â€“12)
Phase 6  â†’  Pipelines + Deployment            (Week 13)
Phase 7  â†’  Consolidation                     (Week 14)
```

---

## Progress Tracker

| Week | Phase | Topic | Project | Status |
|---|---|---|---|---|
| 1 | 0 | Orientation + SQL | SQL queries in Jupyter | âœ… |
| 2 | 1 | Linear Regression | House Price Prediction | ðŸ”„ |
| 3 | 1 | Ridge + Lasso | Medical Cost Predictor â­ | ðŸ”œ |
| 4 | 2 | Logistic Regression | Titanic Survival | ðŸ”œ |
| 5 | 2 | SVM + Evaluation | Breast Cancer Detection | ðŸ”œ |
| 6 | 2 | Classification Pipeline | Credit Card Fraud Detection â­ | ðŸ”œ |
| 7 | 3 | Model Behavior | Overfitting Demonstration | ðŸ”œ |
| 8 | 4 | Decision Trees | Heart Disease Prediction | ðŸ”œ |
| 9 | 4 | Random Forests | Employee Attrition | ðŸ”œ |
| 10 | 4 | XGBoost | Flight Price Prediction â­ | ðŸ”œ |
| 11 | 5 | kNN | Wine Quality Classifier | ðŸ”œ |
| 12 | 5 | Clustering + PCA | Customer Segmentation â­ | ðŸ”œ |
| 13 | 6 | Pipelines + Deploy | End-to-End Pipeline + Streamlit â­ | ðŸ”œ |
| 14 | 7 | Consolidation | Cleanup + Resume + Day 100 Post | ðŸ”œ |

> â­ = Portfolio piece | ðŸ”œ = Upcoming | ðŸ”„ = In Progress | âœ… = Done

---

## Portfolio Projects (5 Core)

### 1. Medical Cost Predictor â€” Week 3
Linear vs Ridge vs Lasso on medical insurance data. Coefficient shrinkage visualised. Shows which features Lasso eliminates.
`Python` `scikit-learn` `pandas` `matplotlib`
â†’ [View Project](projects/02-medical-cost/)

---

### 2. Credit Card Fraud Detection â€” Week 6
Imbalanced dataset. Logistic regression + SVM benchmarked. Precision-recall focus over accuracy. Full metric comparison.
`Python` `scikit-learn` `seaborn` `XGBoost`
â†’ [View Project](projects/05-fraud-detection/)

---

### 3. Flight Price Prediction â€” Week 10
XGBoost vs Random Forest benchmark. GridSearchCV tuning. Feature importance charts.
`Python` `XGBoost` `scikit-learn` `pandas`
â†’ [View Project](projects/09-flight-price/)

---

### 4. Customer Segmentation â€” Week 12
k-Means clustering. Elbow method. PCA visualisation. Business interpretation per segment.
`Python` `scikit-learn` `matplotlib` `seaborn`
â†’ [View Project](projects/11-customer-segmentation/)

---

### 5. End-to-End ML Pipeline + Streamlit â€” Week 13
Full sklearn Pipeline with ColumnTransformer. MLflow experiment tracking. Deployed on Streamlit.
`Python` `scikit-learn` `MLflow` `Streamlit`
â†’ [View Project](projects/12-end-to-end-pipeline/) | [Live Demo](#)

---

## Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn, XGBoost |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Notebooks | Jupyter |
| Experiment Tracking | MLflow |
| Deployment | Streamlit |
| Version Control | Git, GitHub |

---

## Requirements

| Name | Version |
|---|---|
| numpy | 2.3.1 |
| pandas | 2.3.0 |
| matplotlib | 3.10.3 |
| seaborn | 0.13.2 |
| scikit-learn | 1.7.0 |
| jupyter | 1.1.1 |

## Repo Structure

```
100-days-of-ml/
â”œâ”€â”€ README.md
â”œâ”€â”€ logs/
â”‚   â””â”€â”€ weekly-log-template.md     â† Weekly notes template
â”œâ”€â”€ week-01/
â”‚   â”œâ”€â”€ notes.md                   â† Concepts learned
â”‚   â”œâ”€â”€ notebook.ipynb             â† Code
â”‚   â””â”€â”€ post-log.md                â† LinkedIn post link + engagement
â”œâ”€â”€ week-02/
â”‚   â””â”€â”€ ...
â”œâ”€â”€ projects/
â”‚   â”œâ”€â”€ 01-house-price/
â”‚   â”œâ”€â”€ 02-medical-cost/           â† â­ Portfolio
â”‚   â”œâ”€â”€ 03-titanic/
â”‚   â”œâ”€â”€ 04-breast-cancer/
â”‚   â”œâ”€â”€ 05-fraud-detection/        â† â­ Portfolio
â”‚   â”œâ”€â”€ 06-overfitting-demo/
â”‚   â”œâ”€â”€ 07-heart-disease/
â”‚   â”œâ”€â”€ 08-employee-attrition/
â”‚   â”œâ”€â”€ 09-flight-price/           â† â­ Portfolio
â”‚   â”œâ”€â”€ 10-wine-quality/
â”‚   â”œâ”€â”€ 11-customer-segmentation/  â† â­ Portfolio
â”‚   â””â”€â”€ 12-end-to-end-pipeline/    â† â­ Portfolio (deployed)
â””â”€â”€ requirements.txt
```
---

## Connect

| Platform | Link |
|---|---|
| LinkedIn | [linkedin.com/in/dhruvpatel0711](https://linkedin.com/in/dhruvpatel0711) |
| Portfolio | [dhruvpatel0711.github.io/my-portfolio](https://dhruvpatel0711.github.io/my-portfolio) |


---

*Updated weekly. Last update: Week 2 â€” In Progress.*
