"""
Create a sample dataset for MLOps training
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def create_sample_dataset(output_path='data/dataset.csv'):
    """Generate and save a sample classification dataset"""
    
    print("Generating sample dataset...")
    
    # Generate sample classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Create DataFrame
    columns = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df['target'] = y
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✓ Dataset created successfully!")
    print(f"  - Shape: {df.shape}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Samples: {len(df)}")
    print(f"  - Target distribution:\n{df['target'].value_counts()}")
    print(f"  - Saved to: {output_path}")

if __name__ == "__main__":
    create_sample_dataset()
