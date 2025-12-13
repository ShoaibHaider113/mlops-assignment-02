# Data Directory

This directory contains the dataset used for training.

## Dataset
- `dataset.csv` - Main dataset (tracked by DVC)

## Note
The actual data files are tracked by DVC and stored in the DVC remote.
Only the `.dvc` metadata files are committed to Git.

## Creating Sample Dataset
If you need to create a sample dataset for testing, use the following Python code:

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Generate sample classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

# Create DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['target'] = y

# Save to CSV
df.to_csv('data/dataset.csv', index=False)
print("Sample dataset created!")
```
