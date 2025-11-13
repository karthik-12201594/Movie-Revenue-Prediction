import os
import joblib
import pandas as pd
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
    except Exception as e:
        raise CustomException(f"Failed to save object to {file_path}", e)

def load_object(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(f"Failed to load object from {file_path}", e)

def read_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise CustomException(f"Failed to read CSV: {file_path}", e)
