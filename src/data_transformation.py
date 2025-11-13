import os
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.utils import save_object, read_csv
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

@dataclass
class DataTransformationConfig:
    preprocessor_path: str

class DataTransformation:
    def __init__(self, config: DataTransformationConfig, target_col: str):
        self.config = config
        self.target_col = target_col

    def get_preprocessor(self, num_cols, cat_cols):
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        return ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ])

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = read_csv(train_path)
            test_df = read_csv(test_path)
            num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
            if self.target_col in num_cols: num_cols.remove(self.target_col)
            if self.target_col in cat_cols: cat_cols.remove(self.target_col)

            preprocessor = self.get_preprocessor(num_cols, cat_cols)
            X_train = train_df.drop(columns=[self.target_col])
            y_train = train_df[self.target_col]
            X_test = test_df.drop(columns=[self.target_col])
            y_test = test_df[self.target_col]

            preprocessor.fit(X_train)
            save_object(self.config.preprocessor_path, preprocessor)
            logger.info("Data transformation complete")
            return (preprocessor.transform(X_train), y_train,
                    preprocessor.transform(X_test), y_test)
        except Exception as e:
            raise CustomException("Error in data transformation", e)
