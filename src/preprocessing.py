"""
Data Preprocessing & Feature Engineering Module for Laptop Price Prediction
"""

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def get_preprocessor():
    """Create and return the ColumnTransformer preprocessor pipeline."""
    categorical_features = [
        "Laptop",
        "Status",
        "Brand",
        "Model",
        "CPU",
        "Storage type",
        "GPU",
        "Touch"
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ],
        remainder="passthrough"
    )
    return preprocessor

def save_preprocessor(preprocessor, filepath="models/preprocessor.pkl"):
    """Save the fitted preprocessor artifact."""
    joblib.dump(preprocessor, filepath)

def load_preprocessor(filepath="models/preprocessor.pkl"):
    """Load the preprocessor artifact."""
    return joblib.load(filepath)
