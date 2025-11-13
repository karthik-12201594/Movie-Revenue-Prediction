# src/predict_pipeline.py
import os
import numpy as np
import pandas as pd
from src.utils import load_object, read_csv
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

class PredictPipeline:
    """
    Loads available artifacts. Priority:
      1) If preprocessor.joblib and random_forest.joblib exist -> use them.
      2) Else if legacy artifacts (scaler.pkl, feature_names.pkl, encoders, movie_revenue_model.pkl) exist -> use legacy flow.
    Methods:
      - predict_single(input_dict): returns a float prediction
      - predict_from_csv(csv_path): returns numpy array of predictions
    """
    def __init__(self, artifacts_dir: str = None, model_filename_priority: str = None, target_column: str = "Revenue"):
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if artifacts_dir is None:
                artifacts_dir = os.path.join(project_root, "artifacts")

            self.target_column = target_column
            self.artifacts_dir = artifacts_dir

            # priority files
            self.preprocessor_path = os.path.join(artifacts_dir, "transformer", "preprocessor.joblib")
            self.model_path = os.path.join(artifacts_dir, "models", "random_forest.joblib")

            # legacy files
            self.legacy_model_path = os.path.join(artifacts_dir, "models", "movie_revenue_model.pkl")
            self.scaler_path = os.path.join(artifacts_dir, "transformer", "scaler.pkl")
            self.feature_names_path = os.path.join(artifacts_dir, "transformer", "feature_names.pkl")
            self.genre_enc_path = os.path.join(artifacts_dir, "transformer", "genre_encoder.pkl")
            self.lang_enc_path = os.path.join(artifacts_dir, "transformer", "language_encoder.pkl")

            # decide which flow to use
            if os.path.exists(self.preprocessor_path) and os.path.exists(self.model_path):
                logger.info("Found preprocessor.joblib and random_forest.joblib -> using modular flow")
                self.flow = "modular"
                self.preprocessor = load_object(self.preprocessor_path)
                self.model = load_object(self.model_path)
            elif os.path.exists(self.legacy_model_path) and os.path.exists(self.feature_names_path):
                logger.info("Found legacy artifacts -> using legacy flow")
                self.flow = "legacy"
                self.model = load_object(self.legacy_model_path)
                self.scaler = load_object(self.scaler_path) if os.path.exists(self.scaler_path) else None
                self.feature_names = load_object(self.feature_names_path)
                self.genre_encoder = load_object(self.genre_enc_path) if os.path.exists(self.genre_enc_path) else None
                self.lang_encoder = load_object(self.lang_enc_path) if os.path.exists(self.lang_enc_path) else None
            else:
                raise FileNotFoundError("Required model/preprocessor files not found in artifacts. Check artifacts/transformer and artifacts/models.")

        except Exception as e:
            logger.exception("Error initializing PredictPipeline")
            raise CustomException("Failed to initialize PredictPipeline", e)

    def _prepare_modular(self, input_df: pd.DataFrame):
        # preprocessor is a fitted ColumnTransformer or Pipeline; simply transform.
        try:
            X = self.preprocessor.transform(input_df)
            return X
        except Exception as e:
            logger.exception("Modular preprocessor transform failed")
            raise CustomException("Modular preprocessor transform failed", e)

    def _prepare_legacy(self, input_dict: dict):
        try:
            # create DF with feature_names order
            df = pd.DataFrame([input_dict])
            # ensure all expected columns present
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[self.feature_names]

            # apply simple encoders if present
            if 'Genre' in df.columns and self.genre_encoder is not None:
                try:
                    df['Genre'] = self.genre_encoder.transform(df['Genre'].astype(str))
                except Exception:
                    df['Genre'] = [self.genre_encoder.transform([v])[0] for v in df['Genre'].astype(str)]

            if 'Language' in df.columns and self.lang_encoder is not None:
                try:
                    df['Language'] = self.lang_encoder.transform(df['Language'].astype(str))
                except Exception:
                    df['Language'] = [self.lang_encoder.transform([v])[0] for v in df['Language'].astype(str)]

            X = df.values.astype(float)
            if self.scaler is not None:
                X = self.scaler.transform(X)
            return X
        except Exception as e:
            logger.exception("Legacy input preparation failed")
            raise CustomException("Legacy input preparation failed", e)

    def predict_single(self, input_dict: dict):
        try:
            if self.flow == "modular":
                df = pd.DataFrame([input_dict])
                X = self._prepare_modular(df)
                pred = self.model.predict(X)
                return float(pred[0])
            else:
                X = self._prepare_legacy(input_dict)
                pred = self.model.predict(X)
                return float(pred[0])
        except Exception as e:
            logger.exception("Prediction failed")
            raise CustomException("Prediction failed", e)

    def predict_from_csv(self, csv_path: str):
        try:
            df = read_csv(csv_path)
            if self.flow == "modular":
                X = self._prepare_modular(df)
                preds = self.model.predict(X)
                return preds
            else:
                preds = []
                for _, row in df.iterrows():
                    preds.append(self.predict_single(row.to_dict()))
                return np.array(preds)
        except Exception as e:
            logger.exception("CSV prediction failed")
            raise CustomException("CSV prediction failed", e)
