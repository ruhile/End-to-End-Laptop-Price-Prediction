"""
Prediction Pipeline Module for Laptop Price Prediction
"""

import os
import joblib
import pandas as pd

_model = None

def get_model(model_path="models/laptop_price_model.pkl"):
    global _model
    if _model is None:
        if not os.path.exists(model_path):
            model_path = "../models/laptop_price_model.pkl"
        _model = joblib.load(model_path)
    return _model

def predict(input_dict):
    """
    Generate price prediction from a input dictionary or DataFrame.
    """
    model = get_model()
    if isinstance(input_dict, dict):
        df = pd.DataFrame([input_dict])
    else:
        df = input_dict
        
    pred = model.predict(df)[0]
    return round(float(pred), 2)
