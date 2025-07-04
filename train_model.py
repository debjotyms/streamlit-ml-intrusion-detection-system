#!/usr/bin/env python3
"""
Script to retrain the Random Forest model with the 10 features used in main.py
This resolves the compatibility issue with scikit-learn versions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# The 10 features used in main.py
FEATURES = [
    'service', 'flag', 'src_bytes', 'dst_bytes', 'count', 'same_srv_rate',
    'diff_srv_rate', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_same_src_port_rate'
]

def create_dummy_data():
    """
    Create dummy training data with the 10 features for demonstration.
    In a real scenario, you would load your actual dataset.
    """
    np.random.seed(42)
    n_samples = 10000
    
    # Generate synthetic data
    data = {
        'service': np.random.randint(0, 70, n_samples),
        'flag': np.random.randint(0, 11, n_samples),
        'src_bytes': np.random.randint(0, 10000, n_samples),
        'dst_bytes': np.random.randint(0, 10000, n_samples),
        'count': np.random.randint(0, 100, n_samples),
        'same_srv_rate': np.random.uniform(0, 1, n_samples),
        'diff_srv_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_srv_count': np.random.randint(0, 100, n_samples),
        'dst_host_same_srv_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_same_src_port_rate': np.random.uniform(0, 1, n_samples),
    }
    
    # Create target variable (0 = anomaly, 1 = normal)
    # Make it dependent on some features for realistic patterns
    target = np.where(
        (data['src_bytes'] > 5000) | 
        (data['dst_bytes'] > 5000) | 
        (data['count'] > 50) |
        (data['same_srv_rate'] < 0.3),
        0,  # anomaly
        1   # normal
    )
    
    df = pd.DataFrame(data)
    df['class'] = target
    
    return df

def load_real_data():
    """
    Try to load real data from the data directory.
    If not available, create dummy data.
    """
    try:
        # Try to load training data
        df = pd.read_csv('data/train.csv')
        print("Loaded real training data")
        return df
    except FileNotFoundError:
        print("Real data not found. Creating dummy data for demonstration.")
        return create_dummy_data()

def preprocess_data(df):
    """
    Preprocess the data to match the expected format.
    """
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Handle missing values
    df.fillna(0, inplace=True)
    
    # Encode categorical variables
    categorical_columns = ['protocol_type', 'service', 'flag']
    label_encoders = {}
    
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Encode target variable
    if 'class' in df.columns:
        le_target = LabelEncoder()
        y = le_target.fit_transform(df['class'])
        print(f"Target classes: {le_target.classes_}")
        # Save the target encoder for later use
        joblib.dump(le_target, 'target_encoder.sav')
    else:
        print("Warning: No 'class' column found. Creating dummy target.")
        y = np.random.randint(0, 2, len(df))
    
    # Ensure we have the required features
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        print(f"Warning: Missing features {missing_features}")
        # Create dummy features for missing ones
        for feature in missing_features:
            df[feature] = 0
    
    # Select only the features we need
    X = df[FEATURES].copy()
    
    # Convert all features to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill any remaining NaN values
    X.fillna(0, inplace=True)
    
    # Save label encoders for later use
    joblib.dump(label_encoders, 'label_encoders.sav')
    
    return X, y

def train_model():
    """
    Train a new Random Forest model with the 10 features.
    """
    print("Loading and preprocessing data...")
    df = load_real_data()
    X, y = preprocess_data(df)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target distribution: {pd.Series(y).value_counts()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Apply SMOTE for balancing (optional)
    try:
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print("Applied SMOTE balancing")
    except Exception as e:
        print(f"SMOTE failed: {e}. Using original data.")
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train_balanced)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(model, 'rf.sav')
    joblib.dump(scaler, 'scaler.sav')

    print("Model saved as 'rf.sav'")
    print("Scaler saved as 'scaler.sav'")
    
    return model, scaler

def test_model():
    """
    Test the saved model with sample data.
    """
    print("\nTesting the saved model...")
    
    # Load the model and scaler
    model = joblib.load('rf.sav')
    scaler = joblib.load('scaler.sav')
    
    # Test with sample data
    sample_data = [
        [1, 2, 100, 200, 5, 0.5, 0.2, 3, 0.8, 0.1],  # Should be normal
        [50, 8, 9000, 8000, 80, 0.1, 0.9, 90, 0.2, 0.9]  # Should be anomaly
    ]
    
    sample_df = pd.DataFrame(sample_data, columns=FEATURES)
    sample_scaled = scaler.transform(sample_df)
    predictions = model.predict(sample_scaled)
    
    print("Sample predictions:")
    for i, (data, pred) in enumerate(zip(sample_data, predictions)):
        result = "normal" if pred == 1 else "anomaly"
        print(f"Sample {i+1}: {data} -> {result}")

if __name__ == "__main__":
    print("Retraining Random Forest model for intrusion detection...")
    model, scaler = train_model()
    test_model()
    print("\nModel retraining completed!")
