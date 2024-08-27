from src.constants.constants import *
from src.config.configuration import *
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger.logger import logging
from src.exception.exception import CustomException
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path:str = TRAIN_FILE_PATH
    test_data_path:str = TEST_FILE_PATH
    raw_data_path:str = RAW_FILE_PATH


class DataIngestion:
    def __init__(self):
        self.DataIngestionConfig = DataIngestionConfig()

    
    def initiate_data_ingestion(self):
        try:
            data = pd.read_csv(DATASET_PATH)
            #data = pd.read_csv(os.path.join("C:\Users\bryan\Documents\Machine Learning\Proyectos\Food-Delivery-Time-Prediction\data\raw\delivery_time.csv"))

            os.makedirs(os.path.dirname(self.DataIngestionConfig.raw_data_path), exist_ok=True)
            data.to_csv(self.DataIngestionConfig.raw_data_path, index=False) 

            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

            os.makedirs(os.path.dirname(self.DataIngestionConfig.train_data_path), exist_ok=True)
            train_set.to_csv(self.DataIngestionConfig.train_data_path, index=False)

            os.makedirs(os.path.dirname(self.DataIngestionConfig.test_data_path), exist_ok=True)
            test_set.to_csv(self.DataIngestionConfig.test_data_path, index=False)

            return(
                self.DataIngestionConfig.train_data_path,
                self.DataIngestionConfig.test_data_path,
            )       
        
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    logging.info("Data Ingestion Completed")
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    logging.info("Data Transformation Completed")