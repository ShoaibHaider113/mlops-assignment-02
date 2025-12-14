"""
Unit tests for the training pipeline
Tests data loading, model training, and output validation
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier
import joblib

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import load_data, preprocess_data, train_model, evaluate_model, save_model


class TestDataLoading:
    """Tests for data loading functionality"""
    
    def test_load_data_file_exists(self, tmp_path):
        """Test that load_data successfully loads a CSV file"""
        # Create a temporary CSV file
        test_data = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [5, 4, 3, 2, 1],
            'target': [0, 1, 0, 1, 0]
        })
        test_file = tmp_path / "test_data.csv"
        test_data.to_csv(test_file, index=False)
        
        # Load the data
        df = load_data(str(test_file))
        
        # Assertions
        assert df is not None
        assert len(df) == 5
        assert list(df.columns) == ['feature_1', 'feature_2', 'target']
    
    def test_load_data_shape(self, tmp_path):
        """Test that loaded data has correct shape"""
        test_data = pd.DataFrame({
            'col1': range(10),
            'col2': range(10, 20),
            'target': [0, 1] * 5
        })
        test_file = tmp_path / "test.csv"
        test_data.to_csv(test_file, index=False)
        
        df = load_data(str(test_file))
        
        assert df.shape == (10, 3)
    
    def test_load_data_file_not_found(self):
        """Test that load_data raises error for non-existent file"""
        with pytest.raises(FileNotFoundError):
            load_data("non_existent_file.csv")


class TestDataPreprocessing:
    """Tests for data preprocessing functionality"""
    
    def test_preprocess_data_splits_features_target(self):
        """Test that preprocess_data correctly splits features and target"""
        df = pd.DataFrame({
            'feature_1': [1, 2, 3],
            'feature_2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        
        X, y = preprocess_data(df)
        
        assert X.shape == (3, 2)
        assert y.shape == (3,)
        assert list(X.columns) == ['feature_1', 'feature_2']
    
    def test_preprocess_data_output_types(self):
        """Test that preprocessing returns correct data types"""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [5, 6, 7, 8],
            'target': [0, 1, 0, 1]
        })
        
        X, y = preprocess_data(df)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)


class TestModelTraining:
    """Tests for model training functionality"""
    
    def test_train_model_returns_classifier(self):
        """Test that train_model returns a RandomForestClassifier"""
        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        model = train_model(X, y)
        
        assert isinstance(model, RandomForestClassifier)
        assert hasattr(model, 'predict')
    
    def test_train_model_fits_correctly(self):
        """Test that the model can make predictions after training"""
        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        model = train_model(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_model_accuracy_reasonable(self):
        """Test that model achieves reasonable accuracy on simple data"""
        # Create linearly separable data
        np.random.seed(42)
        X = pd.DataFrame(np.random.rand(200, 3))
        y = pd.Series((X.iloc[:, 0] + X.iloc[:, 1] > 1).astype(int))
        
        model = train_model(X, y)
        accuracy = model.score(X, y)
        
        assert accuracy > 0.7  # Should achieve at least 70% accuracy


class TestModelEvaluation:
    """Tests for model evaluation functionality"""
    
    def test_evaluate_model_returns_accuracy(self):
        """Test that evaluate_model returns a valid accuracy score"""
        X_test = pd.DataFrame(np.random.rand(50, 5))
        y_test = pd.Series(np.random.randint(0, 2, 50))
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_test, y_test)
        
        accuracy = evaluate_model(model, X_test, y_test)
        
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0
    
    def test_evaluate_model_accuracy_range(self):
        """Test that accuracy is within valid range"""
        X = pd.DataFrame(np.random.rand(100, 4))
        y = pd.Series(np.random.randint(0, 2, 100))
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        
        accuracy = evaluate_model(model, X, y)
        
        assert 0.0 <= accuracy <= 1.0


class TestModelSaving:
    """Tests for model saving functionality"""
    
    def test_save_model_creates_file(self, tmp_path):
        """Test that save_model creates a model file"""
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = pd.DataFrame(np.random.rand(10, 3))
        y = pd.Series(np.random.randint(0, 2, 10))
        model.fit(X, y)
        
        model_path = tmp_path / "test_model.pkl"
        save_model(model, str(model_path))
        
        assert model_path.exists()
    
    def test_save_and_load_model(self, tmp_path):
        """Test that saved model can be loaded and used"""
        # Train a model
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = pd.DataFrame(np.random.rand(20, 3))
        y = pd.Series(np.random.randint(0, 2, 20))
        model.fit(X, y)
        
        # Save the model
        model_path = tmp_path / "model.pkl"
        save_model(model, str(model_path))
        
        # Load the model
        loaded_model = joblib.load(model_path)
        
        # Test predictions
        predictions = loaded_model.predict(X)
        
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)


class TestShapeValidation:
    """Tests for data shape validation"""
    
    def test_feature_matrix_shape(self):
        """Test that feature matrix has correct number of dimensions"""
        df = pd.DataFrame({
            'f1': range(5),
            'f2': range(5, 10),
            'f3': range(10, 15),
            'target': [0, 1, 0, 1, 0]
        })
        
        X, y = preprocess_data(df)
        
        assert X.ndim == 2
        assert X.shape[1] == 3  # 3 features
    
    def test_target_vector_shape(self):
        """Test that target vector is 1-dimensional"""
        df = pd.DataFrame({
            'feat1': [1, 2, 3],
            'feat2': [4, 5, 6],
            'label': [0, 1, 0]
        })
        
        X, y = preprocess_data(df)
        
        assert y.ndim == 1
        assert len(y) == 3
    
    def test_shapes_match(self):
        """Test that X and y have matching number of samples"""
        df = pd.DataFrame({
            'a': range(100),
            'b': range(100, 200),
            'c': range(200, 300),
            'target': [0, 1] * 50
        })
        
        X, y = preprocess_data(df)
        
        assert X.shape[0] == y.shape[0]


# Fixtures for pytest
@pytest.fixture
def sample_dataset():
    """Fixture to create a sample dataset"""
    return pd.DataFrame({
        'feature_1': np.random.rand(100),
        'feature_2': np.random.rand(100),
        'feature_3': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })


@pytest.fixture
def trained_model(sample_dataset):
    """Fixture to create a trained model"""
    X, y = preprocess_data(sample_dataset)
    model = train_model(X, y)
    return model
