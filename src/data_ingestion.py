import os
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.utils import read_csv
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

@dataclass
class DataIngestionConfig:
    raw_data_path: str
    train_data_path: str
    test_data_path: str
    test_size: float = 0.2
    random_state: int = 42

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        logger.info("Starting data ingestion...")
        try:
            df = read_csv(self.config.raw_data_path)
            train_df, test_df = train_test_split(df, test_size=self.config.test_size, random_state=self.config.random_state)
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logger.info("Data ingestion complete")
            return self.config.train_data_path, self.config.test_data_path
        except Exception as e:
            raise CustomException("Error in data ingestion", e)
