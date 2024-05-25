import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifact', 'raw.csv')
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")

        try:
            data = pd.read_csv(Path(os.path.join('notebooks/data', 'train.csv')))
            logging.info("Read the dataset as dataframe")

            # Ensure the directory exists for raw data path
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Saved the raw dataset in artifact folder')

            logging.info("Performing train_test_split")
            train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
            logging.info('train_test_split completed')

            # Ensure the directories exist for train and test data paths
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info('Data ingestion part completed')

        except Exception as e:
            logging.error('Exception occurred during data ingestion stage')
            raise customexception(e, sys)

if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()
