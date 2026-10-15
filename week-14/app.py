import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="100 Days of ML Portfolio", layout="wide", page_icon="🚀")

st.sidebar.title("🚀 100 Days of ML")
st.sidebar.markdown("Welcome to the Capstone Portfolio. Select a project below to interact with the models trained during this journey.")

project = st.sidebar.radio("Select Project to Explore", [
    "02. Medical Cost Predictor (Regression)",
    "03. Titanic Survival (Classification)",
    "07. Heart Disease Detection (Classification)",
    "08. IBM HR Attrition (Classification)",
    "10. Wine Quality (Ensemble Learning)",
    "11. Customer Segmentation (Clustering)"
])

base_dir = os.path.dirname(os.path.dirname(__file__))

st.title(project[4:])
st.markdown("---")

# ==========================================
# 2. MEDICAL COST PREDICTOR (REGRESSION)
# ==========================================
if "Medical" in project:
    st.write("This model uses **Linear Regression** to predict how much a person's medical insurance will cost based on their attributes.")
    path = os.path.join(base_dir, "projects", "02_Medical_Cost_Predictor", "data", "insurance.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        st.dataframe(df.head())
        
        df_encoded = pd.get_dummies(df, drop_first=True)
        X = df_encoded.drop('charges', axis=1)
        y = df_encoded['charges']
        
        model = LinearRegression()
        model.fit(X, y)
        
        st.markdown("### 🔮 Live Prediction")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 100, 30)
            bmi = st.slider("BMI", 15.0, 50.0, 25.0)
            children = st.slider("Number of Children", 0, 10, 0)
        with col2:
            smoker = st.selectbox("Smoker?", ["yes", "no"])
            sex = st.selectbox("Sex", ["male", "female"])
        
        input_data = pd.DataFrame({
            'age': [age], 'bmi': [bmi], 'children': [children],
            'sex_male': [1 if sex == 'male' else 0],
            'smoker_yes': [1 if smoker == 'yes' else 0],
            'region_northwest': [0], 'region_southeast': [0], 'region_southwest': [0]
        })
        input_data = input_data.reindex(columns=X.columns, fill_value=0)
        pred = model.predict(input_data)[0]
        st.success(f"**Estimated Insurance Charge: ${pred:,.2f}**")
    else:
        st.error("Dataset not found.")

# ==========================================
# 3. TITANIC SURVIVAL (CLASSIFICATION)
# ==========================================
elif "Titanic" in project:
    st.write("This model uses a **Random Forest Classifier** to predict if a passenger would survive the Titanic disaster.")
    path = os.path.join(base_dir, "projects", "03_Titanic_Survival_Predictions", "data", "Titanic-Dataset.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        st.dataframe(df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].head())
        
        df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].dropna()
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        st.markdown("### 🔮 Live Prediction")
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3])
            sex = st.selectbox("Sex", ["female", "male"])
        with col2:
            age = st.slider("Age", 1, 100, 30)
            fare = st.slider("Fare Paid (£)", 0.0, 500.0, 32.0)
            
        input_data = pd.DataFrame({
            'Pclass': [pclass], 'Sex': [1 if sex == 'female' else 0], 'Age': [age], 'Fare': [fare]
        })
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] * 100
        if pred == 1:
            st.success(f"**Prediction: SURVIVED!** (Probability: {prob:.1f}%)")
        else:
            st.error(f"**Prediction: DID NOT SURVIVE.** (Survival Probability: {prob:.1f}%)")
    else:
        st.error("Dataset not found.")

# ==========================================
# 7. HEART DISEASE DETECTION
# ==========================================
elif "Heart" in project:
    st.write("Uses **Logistic Regression** to predict the presence of heart disease based on clinical features.")
    path = os.path.join(base_dir, "projects", "07_Heart_Disease_Detection", "data", "heart_disease_uci.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        st.dataframe(df[['age', 'sex', 'cp', 'trestbps', 'chol', 'num']].head())
        
        # Simple preprocess
        df = df.dropna(subset=['age', 'sex', 'cp', 'trestbps', 'chol', 'num'])
        df['target'] = (df['num'] > 0).astype(int)
        df['sex'] = df['sex'].apply(lambda x: 1 if str(x).lower() == 'male' else 0)
        
        X = df[['age', 'sex', 'trestbps', 'chol']]
        y = df['target']
        
        model = LogisticRegression()
        model.fit(X, y)
        
        st.markdown("### 🔮 Live Prediction")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 20, 90, 50)
            sex = st.selectbox("Sex", ["Male", "Female"])
        with col2:
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
            chol = st.slider("Cholesterol (mg/dl)", 100, 400, 200)
            
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [1 if sex == "Male" else 0],
            'trestbps': [trestbps],
            'chol': [chol]
        })
        
        prob = model.predict_proba(input_data)[0][1] * 100
        if prob > 50:
            st.error(f"**High Risk of Heart Disease** (Probability: {prob:.1f}%)")
        else:
            st.success(f"**Low Risk of Heart Disease** (Probability: {prob:.1f}%)")
    else:
        st.error("Dataset not found.")

# ==========================================
# 8. IBM HR ATTRITION
# ==========================================
elif "Attrition" in project:
    st.write("Uses **Random Forest** to predict if an employee will leave the company (Attrition) based on IBM HR data.")
    path = os.path.join(base_dir, "projects", "08_IBM_HR_Analytics", "data", "HR-Employee-Attrition.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        st.dataframe(df[['Age', 'MonthlyIncome', 'TotalWorkingYears', 'JobSatisfaction', 'Attrition']].head())
        
        X = df[['Age', 'MonthlyIncome', 'TotalWorkingYears', 'JobSatisfaction']]
        y = df['Attrition'].map({'Yes': 1, 'No': 0})
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        st.markdown("### 🔮 Live Prediction")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Employee Age", 18, 65, 35)
            income = st.slider("Monthly Income ($)", 1000, 20000, 5000)
        with col2:
            years = st.slider("Total Working Years", 0, 40, 10)
            satisfaction = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
            
        input_data = pd.DataFrame({'Age': [age], 'MonthlyIncome': [income], 'TotalWorkingYears': [years], 'JobSatisfaction': [satisfaction]})
        prob = model.predict_proba(input_data)[0][1] * 100
        
        if prob > 50:
            st.warning(f"**High Flight Risk (Attrition)** (Probability: {prob:.1f}%)")
        else:
            st.success(f"**Likely to Stay** (Probability: {prob:.1f}%)")
    else:
        st.error("Dataset not found.")

# ==========================================
# 10. WINE QUALITY (ENSEMBLE)
# ==========================================
elif "Wine" in project:
    st.write("Classifies whether a wine is 'Good' or 'Bad' based on physicochemical tests using a **Random Forest Classifier**.")
    path = os.path.join(base_dir, "projects", "10_Wine_Quality_Classifier", "data", "winequality-red.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, sep=';')
        st.dataframe(df.head())
        
        df['quality_label'] = (df['quality'] > 5).astype(int)
        X = df.drop(['quality', 'quality_label'], axis=1)
        y = df['quality_label']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        st.markdown("### 🔮 Live Prediction")
        col1, col2 = st.columns(2)
        with col1:
            alcohol = st.slider("Alcohol %", float(X['alcohol'].min()), float(X['alcohol'].max()), float(X['alcohol'].mean()))
            sulphates = st.slider("Sulphates", float(X['sulphates'].min()), float(X['sulphates'].max()), float(X['sulphates'].mean()))
        with col2:
            volatile_acidity = st.slider("Volatile Acidity", float(X['volatile acidity'].min()), float(X['volatile acidity'].max()), float(X['volatile acidity'].mean()))
            citric_acid = st.slider("Citric Acid", float(X['citric acid'].min()), float(X['citric acid'].max()), float(X['citric acid'].mean()))
            
        input_data = pd.DataFrame(X.mean()).T
        input_data['alcohol'] = alcohol
        input_data['sulphates'] = sulphates
        input_data['volatile acidity'] = volatile_acidity
        input_data['citric acid'] = citric_acid
        
        pred = model.predict(scaler.transform(input_data))[0]
        if pred == 1:
            st.success("🍷 **Prediction: GOOD Quality Wine (Score > 5)**")
        else:
            st.warning("🍇 **Prediction: BAD Quality Wine (Score <= 5)**")
    else:
        st.error("Dataset not found.")

# ==========================================
# 11. CUSTOMER SEGMENTATION (CLUSTERING)
# ==========================================
elif "Customer" in project:
    st.write("Uses **K-Means Clustering** (Unsupervised Learning) to group mall customers into distinct segments based on Income and Spending Score.")
    path = os.path.join(base_dir, "projects", "11_Customer_Segmentation", "data", "Mall_Customers.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        st.dataframe(df.head())
        
        X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
        
        clusters = st.slider("Select Number of Clusters (k)", 2, 8, 5)
        
        kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X)
        
        st.markdown("### 📊 Live Clustering Plot")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='tab10', ax=ax)
        st.pyplot(fig)
    else:
        st.error("Dataset not found.")

st.markdown("---")
st.caption("Built with Streamlit • 100 Days of ML Portfolio • Week 14 Capstone")

