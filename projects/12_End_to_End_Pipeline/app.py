import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

# Page Config
st.set_page_config(page_title="Model Comparison Dashboard", page_icon="🚀", layout="wide")

st.title("🚀 Machine Learning Model Comparison")
st.markdown("### 100 Days of ML: Week 13 - Pipelines & Deployment")
st.markdown("This interactive Streamlit dashboard trains and compares three different machine learning models on the **Wine Quality** dataset in real-time. Adjust the hyperparameters on the left to see how they impact performance!")

# Load Data
@st.cache_data
def load_data():
    # Use the dataset from week 10 project
    path = os.path.join(os.path.dirname(__file__), "..", "projects", "10_Wine_Quality_Classifier", "data", "winequality-red.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, sep=';')
    else:
        # Fallback to github URL if local path fails
        df = pd.read_csv("https://raw.githubusercontent.com/DhruvPatel0711/100-days-of-ml/main/projects/10_Wine_Quality_Classifier/data/winequality-red.csv", sep=';')
    
    # Binarize target for classification: 'good' quality > 5
    df['quality_label'] = (df['quality'] > 5).astype(int)
    X = df.drop(['quality', 'quality_label'], axis=1)
    y = df['quality_label']
    return X, y, df

X, y, df = load_data()

# Sidebar for hyperparameters
st.sidebar.header("⚙️ Model Hyperparameters")
st.sidebar.markdown("Use these sliders to tune the models on the fly. The dashboard will instantly retrain and update the metrics.")

st.sidebar.markdown("**🌲 Random Forest**")
rf_estimators = st.sidebar.slider("Number of Estimators", 10, 200, 50, step=10)
rf_depth = st.sidebar.slider("Max Depth (RF)", 2, 20, 5)

st.sidebar.markdown("**⚡ XGBoost**")
xgb_learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01)
xgb_depth = st.sidebar.slider("Max Depth (XGB)", 2, 10, 3)

# Data Splitting and Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Models
def train_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=rf_estimators, max_depth=rf_depth, random_state=42),
        "XGBoost": XGBClassifier(learning_rate=xgb_learning_rate, max_depth=xgb_depth, use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "CM": confusion_matrix(y_test, y_pred)
        }
    return results

with st.spinner("Training models in real-time..."):
    results = train_models()

# Display Metrics
st.markdown("---")
st.subheader("📊 Performance Metrics")

col1, col2, col3 = st.columns(3)

def display_metric_card(col, name, metrics):
    with col:
        st.markdown(f"### {name}")
        st.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
        st.metric("F1-Score", f"{metrics['F1-Score']:.3f}")
        st.metric("Precision", f"{metrics['Precision']:.3f}")

display_metric_card(col1, "Logistic Regression", results["Logistic Regression"])
display_metric_card(col2, "Random Forest", results["Random Forest"])
display_metric_card(col3, "XGBoost", results["XGBoost"])

# Bar Chart Comparison
st.markdown("---")
st.subheader("📈 Metric Comparison Chart")

metrics_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [r["Accuracy"] for r in results.values()],
    "Precision": [r["Precision"] for r in results.values()],
    "Recall": [r["Recall"] for r in results.values()],
    "F1-Score": [r["F1-Score"] for r in results.values()]
})

metrics_melted = pd.melt(metrics_df, id_vars=["Model"], var_name="Metric", value_name="Score")

fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=metrics_melted, x="Metric", y="Score", hue="Model", palette="mako", ax=ax)
ax.set_ylim(0.6, 1.0)
ax.set_title("Head-to-Head Model Performance")
st.pyplot(fig)

# Confusion Matrices
st.markdown("---")
st.subheader("🧩 Confusion Matrices")
cm_col1, cm_col2, cm_col3 = st.columns(3)

def plot_cm(cm, title):
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Actual Label')
    return fig

with cm_col1:
    st.pyplot(plot_cm(results["Logistic Regression"]["CM"], "Logistic Regression"))
with cm_col2:
    st.pyplot(plot_cm(results["Random Forest"]["CM"], "Random Forest"))
with cm_col3:
    st.pyplot(plot_cm(results["XGBoost"]["CM"], "XGBoost"))

st.markdown("---")
st.markdown("*(Built for **100 Days of ML** - Week 13 Deployment Demonstration)*")
