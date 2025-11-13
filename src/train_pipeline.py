import os
from src.data_ingestion import DataIngestion, DataIngestionConfig
from src.data_transformation import DataTransformation, DataTransformationConfig
from src.model_trainer import ModelTrainer, ModelTrainerConfig
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

def run_training_pipeline():
    try:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_path = os.path.join(base, "data", "movie_revenue_prediction.csv")
        train_path = os.path.join(base, "artifacts", "data", "train.csv")
        test_path = os.path.join(base, "artifacts", "data", "test.csv")
        preprocessor_path = os.path.join(base, "artifacts", "transformer", "preprocessor.joblib")
        model_path = os.path.join(base, "artifacts", "models", "random_forest.joblib")

        ingestion = DataIngestion(DataIngestionConfig(
            raw_data_path=data_path,
            train_data_path=train_path,
            test_data_path=test_path
        ))
        train_csv, test_csv = ingestion.initiate_data_ingestion()

        transformation = DataTransformation(
            DataTransformationConfig(preprocessor_path=preprocessor_path),
            target_col="revenue"
        )
        X_train, y_train, X_test, y_test = transformation.initiate_data_transformation(train_csv, test_csv)

        trainer = ModelTrainer(ModelTrainerConfig(model_path=model_path))
        results = trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
        logger.info("Pipeline complete!")
        print(results)
    except Exception as e:
        raise CustomException("Error in training pipeline", e)

if __name__ == "__main__":
    run_training_pipeline()
