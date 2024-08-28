from src.constants.constants import *
from src.logger.logger import logging
from src.exception.exception import CustomException
from src.config.configuration import *
import os, sys
import pandas as pd

from src.utils.utils import load_model


class PredictionPipeline:
    
    
    def __init__(self):
        self.c = 0
        print(f"--------{self.c}----------")
    

    def predict(self, features):
        try:
            preprocessor_path = PREPROCESING_OBJ_FILE
            model_path = MODEL_FILE_PATH

            preprocessor = load_model(preprocessor_path)
            model = load_model(model_path)

            data_scaled = preprocessor.transform(features)
            
            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info(f"Error in prediction: {e}")
            raise CustomException(e, sys)
        

class CustomData:

    def __init__(self, 
                 Delivery_person_Age:int,
                 Delivery_person_Ratings:float,
                 Weather_conditions:str,
                 Road_traffic_density:str,
                 Vehicle_condition:int,
                 multiple_dliveries:int,
                 distance:float,
                 Type_of_order:str,
                 Type_of_vehicle:str,
                 Festival:str,
                 City:str):
        
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Vehicle_condition = Vehicle_condition
        self.multiple_dliveries = multiple_dliveries
        self.distance = distance
        self.Type_of_order = Type_of_order
        self.Type_of_vehicle = Type_of_vehicle
        self.Festival = Festival
        self.City = City


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Delivery_person_Age": [self.Delivery_person_Age],
                "Delivery_person_Ratings": [self.Delivery_person_Ratings],
                "Weather_conditions": [self.Weather_conditions],
                "Road_traffic_density": [self.Road_traffic_density],
                "Vehicle_condition": [self.Vehicle_condition],
                "multiple_dliveries": [self.multiple_dliveries],
                "distance": [self.distance],
                "Type_of_order": [self.Type_of_order],
                "Type_of_vehicle": [self.Type_of_vehicle],
                "Festival": [self.Festival],
                "City": [self.City]
            }

            data = pd.DataFrame(custom_data_input_dict)
            return data

        except Exception as e:
            logging.info(f"Error in creating dataframe: {e}")
            raise CustomException(e, sys)