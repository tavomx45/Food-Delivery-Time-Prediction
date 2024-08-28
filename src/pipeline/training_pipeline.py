from src.constants.constants import *
from src.logger.logger import logging
from src.exception.exception import CustomException
from src.config.configuration import *
import os, sys
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.data_ingestion import DataIngestion, DataIngestionConfig


class TrainingPipeline():
    
    
    def __init__(self):
        self.c = 0
        print(f"--------{self.c}----------")


    def main(self):
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("Data Transformation Completed")
        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(train_arr, test_arr)
        logging.info("Model Training Completed")
    
