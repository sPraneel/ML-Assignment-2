import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import joblib
import os

# Load the test data
X_test_scaled = joblib.load('model/X_test_scaled.pkl')
y_test = joblib.load('model/y_test.pkl')

# Define model names
model_names = [
    'LogisticRegression',
    'DecisionTreeClassifier',
    'KNeighborsClassifier',
    'GaussianNB',
    'RandomForestClassifier',
    'XGBClassifier'
]

results = {}

print("--- Model Evaluation ---")
for name in model_names:
    print(f"\nEvaluating {name}...")
    model = joblib.load(f'model/{name}.pkl')
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else [0] * len(y_test) # Fallback for models without predict_proba

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Check if y_pred_proba was actually computed
    if hasattr(model, "predict_proba"):
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc = np.nan # Not applicable if predict_proba is not available

    mcc = matthews_corrcoef(y_test, y_pred)

    results[name] = {
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MCC': mcc
    }

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC Score: {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  MCC Score: {mcc:.4f}")

print("\n--- Summary of Results ---")
results_df = pd.DataFrame(results).T
print(results_df)

# Optionally save results to a CSV or another format for later use
results_df.to_csv('model_evaluation_results.csv')
print("\nModel evaluation results saved to 'model_evaluation_results.csv'")
