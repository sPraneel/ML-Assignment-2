import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils import resample
import joblib
import os

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Original class distribution:")
print(y.value_counts())

# Split data into training and testing sets first to avoid leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def balance_data(X_set, y_set):
    df = pd.concat([X_set, y_set.rename('target')], axis=1)
    counts = df['target'].value_counts()
    majority_class = counts.idxmax()
    minority_class = counts.idxmin()
    
    df_majority = df[df.target == majority_class]
    df_minority = df[df.target == minority_class]
    
    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True, 
                                     n_samples=len(df_majority), 
                                     random_state=42)
    
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    return df_balanced.drop('target', axis=1), df_balanced['target']

print("Balancing training data...")
X_train_balanced, y_train_balanced = balance_data(X_train, y_train)

print("Balancing test data (for uniform evaluation distribution)...")
X_test_balanced, y_test_balanced = balance_data(X_test, y_test)

print("New Training class distribution:")
print(y_train_balanced.value_counts())

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test_balanced)

# Save the scaler and feature names
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(data.feature_names, 'model/feature_names.pkl')

# Initialize and train models
models = {
    'LogisticRegression': LogisticRegression(random_state=42),
    'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'GaussianNB': GaussianNB(),
    'RandomForestClassifier': RandomForestClassifier(random_state=42),
    'XGBClassifier': XGBClassifier(random_state=42, eval_metric='logloss')
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train_balanced)
    # Save the trained model
    joblib.dump(model, f'model/{name}.pkl')

print("All models trained and saved successfully!")

# Save balanced test data for evaluation
joblib.dump(X_test_scaled, 'model/X_test_scaled.pkl')
joblib.dump(y_test_balanced, 'model/y_test.pkl')