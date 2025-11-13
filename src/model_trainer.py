# src/model_trainer.py

from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.utils import save_object
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

@dataclass
class ModelTrainerConfig:
    model_path: str
    n_estimators: int = 10        # small model for quick debugging
    random_state: int = 42
    n_jobs: int = -1              # use all CPU cores for faster training


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logger.info(f"Creating RandomForest with n_estimators={self.config.n_estimators}")

            # Create model
            model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )

            # Train model
            logger.info("Starting RandomForest training ...")
            model.fit(X_train, y_train)
            logger.info("RandomForest training finished")

            # Predictions
            preds = model.predict(X_test)

            # Calculate metrics
            rmse = mean_squared_error(y_test, preds) ** 0.5  # Manual RMSE calculation
            r2 = r2_score(y_test, preds)

            logger.info(f"RMSE on test set: {rmse:.4f}")
            logger.info(f"R2 on test set: {r2:.4f}")

            # Save model
            save_object(self.config.model_path, model)
            logger.info(f"Saved trained model at {self.config.model_path}")

            # Return results
            return {
                "model_path": self.config.model_path,
                "rmse": rmse,
                "r2": r2
            }

        except Exception as e:
            logger.exception("Exception in model training")
            raise CustomException("Error during model training", e)
