import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

st.title("ML Classification Model Dashboard")
st.write("This application demonstrates various classification models on the Breast Cancer Wisconsin dataset.")

# Load the scaler and test data
try:
    scaler = joblib.load('model/scaler.pkl')
    X_test_scaled = joblib.load('model/X_test_scaled.pkl')
    y_test = joblib.load('model/y_test.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'model_training.py' has been run to train and save models.")
    st.stop()

# Define model names
model_names = [
    'LogisticRegression',
    'DecisionTreeClassifier',
    'KNeighborsClassifier',
    'GaussianNB',
    'RandomForestClassifier',
    'XGBClassifier'
]

# Model selection dropdown
selected_model_name = st.sidebar.selectbox("Select a Model", model_names)

# Load the selected model
try:
    model = joblib.load(f'model/{selected_model_name}.pkl')
except FileNotFoundError:
    st.error(f"Model file for {selected_model_name} not found.")
    st.stop()

st.header(f"Results for {selected_model_name}")

# Make predictions
y_pred = model.predict(X_test_scaled)
if hasattr(model, "predict_proba"):
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
else:
    y_pred_proba = [0] * len(y_test) # Dummy for models without predict_proba

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba) if hasattr(model, "predict_proba") else np.nan
mcc = matthews_corrcoef(y_test, y_pred)

st.subheader("Evaluation Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("Precision", f"{precision:.4f}")
col3.metric("Recall", f"{recall:.4f}")

col4, col5, col6 = st.columns(3)
col4.metric("F1 Score", f"{f1:.4f}")
col5.metric("AUC Score", f"{auc:.4f}" if not np.isnan(auc) else "N/A")
col6.metric("MCC Score", f"{mcc:.4f}")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', ax=ax)
ax.set_title(f"Confusion Matrix for {selected_model_name}")
st.pyplot(fig)

st.subheader("Dataset Upload")
uploaded_file = st.file_uploader("Upload your own CSV for prediction (must have same features as training data)", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        input_df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(input_df.head(10))

        # Load feature names
        try:
            feature_names = joblib.load('model/feature_names.pkl')
        except FileNotFoundError:
            st.error("Feature names file not found. Please retrain models.")
            st.stop()

        # Check if features match
        # If input has 'target' column, drop it for prediction but keep for reference if needed
        if 'target' in input_df.columns:
            input_df = input_df.drop('target', axis=1)
        
        # Check if columns match (ignoring order for loose check, or strict)
        # Here we check if all required features are present
        missing_cols = set(feature_names) - set(input_df.columns)
        if missing_cols:
            st.error(f"Missing features in uploaded file: {missing_cols}")
        else:
            # Reorder columns to match training data
            X_input = input_df[feature_names]
            
            # Scale the data
            X_input_scaled = scaler.transform(X_input)
            
            # Predict
            predictions = model.predict(X_input_scaled)
            prediction_proba = model.predict_proba(X_input_scaled)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Create results DataFrame
            results_df = pd.DataFrame()
            results_df['Prediction'] = ['Malignant' if p == 0 else 'Benign' for p in predictions]
            if prediction_proba is not None:
                results_df['Probability (Benign)'] = prediction_proba
            
            st.subheader("Prediction Results")
            st.dataframe(results_df)
            
            # Download results
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
