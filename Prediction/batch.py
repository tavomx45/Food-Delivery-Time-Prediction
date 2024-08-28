from src.constants.constants import *
from src.config.configuration import *
from src.utils.utils import load_model
import os, sys
import pandas as pd
import numpy as np
from src.logger.logger import logging
from src.exception.exception import CustomException
import pickle
from sklearn.pipeline import Pipeline


PREDICTION_FOLDER = "batch_prediction"
PREDICTION_CSV = "prediction.csv"
PREDICTION_FILE = "output.csv"
FEATURE_ENG_FOLDER = "feature_eng"

ROOT_DIR = os.getcwd()
BATCH_PREDICTOR = os.path.join(ROOT_DIR, PREDICTION_FOLDER, PREDICTION_CSV)
FEATURE_ENG = os.path.join(ROOT_DIR, FEATURE_ENG_FOLDER)


class BatchPrediction:
    def __init__(self, input_file_path, model_file_path, transformer_file_path, feature_engineering_file_path) -> None:
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path
        self.feature_engineering_file_path = feature_engineering_file_path
    

    def start_batch_prediction(self):
        try:
            # load feature engineering pipeline path
            with open(self.feature_engineering_file_path, 'rb') as f:
                feature_pipeline = pickle.load(f)

            # load the data transformation pipeline path

            with open(self.transformer_file_path, 'rb') as f:
                processor = pickle.load(f)

            # load the model separetely
            model = load_model(file_path=self.model_file_path)

            # Create feature engineering pipeline
            feature_engineering_pipeline = Pipeline([
                ('feature_engineering', feature_pipeline),
            ])
            
            data = pd.read_csv(self.input_file_path)

            data.to_csv("data_delivery_time_prediction.csv")
            # Apply feature engineering pipeline
            data = feature_engineering_pipeline.transform(data)

            data.to_csv("data_after_feature_engineering.csv")

            FEATURE_ENGINEERING_PATH = FEATURE_ENG
            os.makedirs(FEATURE_ENGINEERING_PATH, exist_ok=True)

            file_path = os.path.join(FEATURE_ENGINEERING_PATH, "batch_feature_engineering.csv")
            data.to_csv(file_path, index=False)

            # time_taken
            data = data.drop("Time_taken (min)", axis=1)
            data.to_csv("time_taken_dropped.csv")

            transformed_data = processor.transform(data)

            file_path = os.path.join(FEATURE_ENGINEERING_PATH, "processor.csv")
            predictions = model.predict(transformed_data)

            data_prediction = pd.DataFrame(predictions, columns=["Predicted_time_taken"])

            BATCH_PREDICTION_PATH = BATCH_PREDICTOR
            os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True)

            csv_path = os.path.join(BATCH_PREDICTION_PATH, "output.csv")
            data_prediction.to_csv(csv_path, index=False)

            logging.info(f"Prediction completed and saved at {csv_path}")

        except Exception as e:
            raise CustomException(e, sys)