"""
Model Training Module for Laptop Price Prediction
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_save_model(data_path="data/processed/laptops_clean.csv", model_dir="models"):
    if not os.path.exists(data_path):
        data_path = "../data/processed/laptops_clean.csv"
        
    df = pd.read_csv(data_path)
    target_col = 'Final Price' if 'Final Price' in df.columns else 'Price'

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    categorical_features = [
        "Laptop", "Status", "Brand", "Model", "CPU",
        "Storage type", "GPU", "Touch"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough"
    )

    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_pipeline.fit(X_train, y_train)

    preds = rf_pipeline.predict(X_test)
    print(f"Random Forest MAE : {mean_absolute_error(y_test, preds):.2f}")
    print(f"Random Forest RMSE: {mean_squared_error(y_test, preds)**0.5:.2f}")
    print(f"Random Forest R2  : {r2_score(y_test, preds):.4f}")

    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, "laptop_price_model.pkl")
    joblib.dump(rf_pipeline, out_path)
    print(f"Saved model to {out_path}")

if __name__ == "__main__":
    train_and_save_model()
