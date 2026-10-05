# Pipelines & Deployment Notes

## Why This Topic Matters

A machine learning model sitting in a Jupyter Notebook is a science experiment. It generates exactly zero dollars for a business.

To be useful, the model must be combined with its preprocessing steps into a Pipeline, tracked for reproducibility, and deployed as a web application so users can interact with it.

---

# Sklearn Pipelines

## Definition

An object that chains together preprocessing (scaling, encoding, imputing) and the machine learning model into a single, unbreakable workflow.

### Formula (if applicable)

```python
Pipeline([
  ('preprocessor', ColumnTransformer),
  ('model', XGBoost)
])
```

### Interpretation

When you deploy a model, the live user input (e.g., Age = 30) must undergo the exact same StandardScaler math as the training data. The Pipeline ensures this happens automatically.

### Example

Deploying a Flight Predictor. The user enters "Delhi". The pipeline automatically applies the One-Hot Encoding learned during training, then predicts the price.

---

# MLflow Experiment Tracking

## Definition

An open-source platform to manage the ML lifecycle, specifically tracking hyperparameters, metrics, and saving model artifacts.

### Why It Matters

When you train 50 different XGBoost models tweaking `max_depth` and `learning_rate`, you will forget which one had the highest ROC-AUC. MLflow logs it all to a dashboard.

### Example

Logging the `n_estimators`, the `Accuracy`, and the `pickle` file of the model in a single `mlflow.log_run()` call.

---

# Streamlit

## Definition

A Python library that turns data scripts into interactive web apps in minutes, requiring zero HTML, CSS, or JavaScript.

### Visualization explanation

A sleek webpage where a user adjusts a slider for "Age" and a dropdown for "City", clicks "Predict", and the Python pipeline executes in the backend and flashes the result on screen.

---

# Mathematical Intuition

Deployment isn't about model math, it's about software engineering math: ensuring deterministic outputs. The parameters of the scaler (mean and standard deviation) calculated during `fit()` on the training data are frozen and saved inside the Pipeline so they can be applied via `transform()` to future unseen user inputs.

---

# How It Works

Step-by-step process.

```text
Train Pipeline in Notebook
↓
Save Pipeline as a .pkl file (joblib)
↓
Build Streamlit UI (st.selectbox, st.number_input)
↓
Load .pkl file in Streamlit script
↓
Pass UI inputs to Pipeline.predict()
```

---

# Advantages

- Pipelines completely prevent data leakage.
- Streamlit allows data scientists to build full-stack apps without front-end knowledge.
- Deploying a model proves you understand end-to-end ML engineering.

---

# Limitations

- Streamlit is great for prototypes but not highly scalable for massive enterprise apps.
- `joblib` pickle files can break if the scikit-learn version on the server differs from the notebook.

---

# Common Mistakes

- Fitting the scaler on the test data or live data (the scaler must ONLY be fitted on the training data).
- Trying to manually encode user inputs in the deployment script instead of using a `ColumnTransformer` inside a `Pipeline`.
- Forgetting to include a `requirements.txt` file for the deployment server.

---

# Practical Interpretation

Explain how to interpret outputs.

Examples:

> What does a saved Pipeline object represent?
It is the frozen brain of your entire process, containing both the data transformation rules and the learned model weights.

> When should this method be used?
Every time you build a model you intend to use outside of a sandbox environment.

> When should it not be used?
N/A. Production engineering practices should always be applied.

---

# Industry Applications

Examples:

- Web Apps: Any internal company tool that requires an AI prediction.
- MLOps: Automated model retraining and deployment pipelines.
- Portfolio Building: Proving to recruiters you can build working software, not just notebooks.

---

# Quick Comparison

| Concept | Meaning |
|----------|---------|
| Pipeline | Bundles preprocessing and modeling together |
| MLflow | Logs experiments and saves artifacts |
| Streamlit | Builds the user interface for the model |

---

# Resources

- Krish Naik — "End to End ML Project"
- "MLflow Tutorial"
- "Streamlit in 20 minutes"
