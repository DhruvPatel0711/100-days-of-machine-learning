# Week 14: 100 Days of ML Capstone Portfolio

> *“The culmination of 14 weeks of data engineering, predictive modeling, and pipeline deployment.”*

## Overview
Welcome to the final week of the **100 Days of ML** journey! This module serves as the capstone consolidation of everything built over the last three months. 

Instead of building a singular new model, Week 14 is focused on **Deployment and Presentation**. We have built an interactive Master Portfolio Hub using **Streamlit**, which aggregates the flagship models from the previous 13 weeks into a single, unified web application.

---

## 🚀 The Capstone Application (`app.py`)

The primary deliverable this week is `app.py`, a multi-page Streamlit web application that serves as an interactive resume. 

### Features of the Master Dashboard:
- **Dynamic Routing:** A sidebar allows users (and recruiters) to seamlessly toggle between different machine learning projects without leaving the page.
- **Live Data Preprocessing:** The app programmatically traverses the repository's directories to locate the raw `.csv` files for each project, loads them into memory, and displays them via `pandas` integration.
- **On-the-Fly Inference:** Rather than relying on pre-pickled models, the app rapidly trains lightweight models (Linear Regression, Random Forests) in real-time when a user selects a project. 
- **Interactive UI:** Users can manipulate dynamic sliders and dropdowns (e.g., adjusting Passenger Class for the Titanic, or manipulating Sulphate levels for Wine) to receive live model predictions and probability scores.

### Models Featured in the Hub:
1. **Medical Cost Predictor (Week 2):** Linear Regression predicting healthcare charges.
2. **Titanic Survival (Week 4):** Random Forest Classification mapping demographics to survival probabilities.
3. **Heart Disease Detection (Week 7):** Logistic Regression assessing clinical features for disease risk.
4. **IBM HR Attrition (Week 8):** Random Forest evaluating employee flight risk and job satisfaction.
5. **Wine Quality Classifier (Week 10):** Ensemble learning classifying physicochemical tests into quality metrics.
6. **Customer Segmentation (Week 11):** Unsupervised K-Means clustering of mall customers based on income and spending habits.

---

## 📄 Professional Assets

Alongside the code deployment, this folder contains the written assets necessary for transitioning this repository into a professional portfolio:

- **`resume_additions.md`**: Pre-written, high-impact bullet points summarizing the technical achievements of this repository, ready to be pasted into a CV.
- **`portfolio_links.md`**: The centralized hub containing the live deployment URLs for the Streamlit applications hosted on Streamlit Cloud.
- **`CONTRIBUTING.md`**: (Located in the root) Guidelines for open-source interaction, proving that the repository is structured to industry standards.

---

## 🏁 Conclusion

This marks the end of the 100 Days of ML. From basic SQL queries in Week 1 to a fully deployed interactive model hub in Week 14, this repository stands as a comprehensive testament to end-to-end Machine Learning engineering.
