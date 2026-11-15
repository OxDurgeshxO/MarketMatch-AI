import os
import pytest
import pandas as pd
import numpy as np

def test_data_files_exist():
    assert os.path.exists("data/ecommerce_rfm_customers.csv"), "RFM dataset missing"
    assert os.path.exists("data/mall_customers_legacy.csv"), "Legacy mall dataset missing"

def test_rfm_dataset_schema():
    df = pd.read_csv("data/ecommerce_rfm_customers.csv")
    assert not df.empty, "Dataset should not be empty"
    # Check key columns
    expected_cols = ["CustomerID", "Recency", "Frequency", "Monetary"]
    for col in expected_cols:
        assert any(col.lower() in c.lower() for c in df.columns), f"Missing column {col}"

def test_model_files_exist():
    assert os.path.exists("models/kmeans_rfm_pipeline.joblib") or os.path.exists("models/scaler_rfm.joblib"), "Model artifact missing"
