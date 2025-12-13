"""
Simple ML Training Script for MLOps Assignment
Trains a basic classification model on the dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from datetime import datetime


def load_data(filepath):
    """Load dataset from CSV file"""
    print(f"Loading data from {filepath}...")
    df = pd.read_data(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df):
    """Preprocess the dataset"""
    print("Preprocessing data...")
    
    # Assuming last column is target, rest are features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Handle any categorical variables if needed
    # For now, assuming all numeric
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y


def train_model(X_train, y_train):
    """Train a Random Forest model"""
    print("Training model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model training complete!")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy


def save_model(model, filepath):
    """Save trained model to disk"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    print(f"\nSaving model to {filepath}...")
    joblib.dump(model, filepath)
    print("Model saved successfully!")


def main():
    """Main training pipeline"""
    print("=" * 50)
    print("ML Training Pipeline Started")
    print(f"Timestamp: {datetime.now()}")
    print("=" * 50)
    
    # Paths
    data_path = "data/dataset.csv"
    model_path = "models/model.pkl"
    
    # Load data
    df = load_data(data_path)
    
    # Preprocess
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, model_path)
    
    print("\n" + "=" * 50)
    print("Training Pipeline Completed Successfully!")
    print(f"Final Accuracy: {accuracy:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
