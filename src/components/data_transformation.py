import os, sys
from src.constants.constants import *
from src.config.configuration import *
from src.utils.utils import sav_obj
from src.logger.logger import logging
from src.exception.exception import CustomException
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        logging.info("Feature Engineering Object Created")


    def distance_numpy(self, data, lat1, lon1, lat2, lon2):
        p = np.pi/100
        a = 0.5 - np.cos((data[lat2]-data[lat1])*p)/2 + np.cos(data[lat1]*p) * np.cos(data[lat2]*p) * (1-np.cos((data[lon2]-data[lon1])*p))/2
        data["distance"] = 12742 * np.arcsin(np.sqrt(a))
    
    def transform_data(self, data):
        try:
            data.drop(["ID"], axis=1, inplace=True)
            self.distance_numpy(data, 'Restaurant_latitude',
                'Restaurant_longitude',
                'Delivery_location_latitude',
                'Delivery_location_longitude')

            data.drop(['Delivery_person_ID', 'Restaurant_latitude','Restaurant_longitude',
                            'Delivery_location_latitude',
                            'Delivery_location_longitude',
                            'Order_Date','Time_Orderd','Time_Order_picked'], axis=1, inplace=True)

            logging.info("Dropping columns from original dataset")

            return data
        
        except Exception as e:
            raise CustomException(e, sys)

    def fit(self, X, y=None):
        return self

    def transform(self, X:pd.DataFrame, y=None):
        try:
            transformed_data = self.transform_data(X)
            return transformed_data

        except Exception as e:
            raise CustomException(e, sys)

@dataclass
class DataTransformationConfig:
    processed_obj_file_path:str = PREPROCESING_OBJ_FILE
    transformed_train_path:str = TRANSFORM_TRAIN_FILE_PATH
    transformed_test_path:str = TRANSFORM_TEST_FILE_PATH
    feature_eng_obj_path:str = FEATURE_ENGG_OBJ_FILE_PATH


class DataTransformation:
    def __init__(self):
        self.DataTransformationConfig = DataTransformationConfig()

    
    def get_data_trasformation_obj(self):
        try:
            Road_traffic_density = ['Low', 'Medium', 'High', 'Jam']
            Weather_conditions = ['Sunny', 'Cloudy', 'Fog', 'Sandstorms', 'Windy', 'Stormy']

            categorical_columns = ['Type_of_order','Type_of_vehicle','Festival','City']
            ordinal_encoder = ['Road_traffic_density', 'Weather_conditions']
            numerical_columns=['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition',
                              'multiple_deliveries','distance']

            # Numerical pipeline
            numerical_pipeline = Pipeline(steps = [
                ('impute', SimpleImputer(strategy = 'constant', fill_value=0)),
                ('scaler', StandardScaler(with_mean=False))
            ])

            # Categorical Pipeline
            categorical_pipeline = Pipeline(steps = [
                ('impute', SimpleImputer(strategy = 'most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown = 'ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            # ordinal Pipeline
            ordinal_pipeline = Pipeline(steps = [
                ('impute', SimpleImputer(strategy = 'most_frequent')),
                ('ordinal', OrdinalEncoder(categories=[Road_traffic_density,Weather_conditions])),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ('num', numerical_pipeline, numerical_columns),
                ('cat', categorical_pipeline, categorical_columns),
                ('ord', ordinal_pipeline, ordinal_encoder)
            ])

            logging.info("Data Transformation Object Created")
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def get_feature_engineering_obj(self):
        try:
            feature_engineering_obj = Pipeline(steps=[
                ('feature_eng', FeatureEngineering())
            ])

            logging.info("Feature Engineering Object Created")
            return feature_engineering_obj

        except Exception as e:
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Obtaining feature engineering object")
            fe_obj = self.get_feature_engineering_obj()

            logging.info("Transforming training and testing data")
            train_data = fe_obj.fit_transform(train_data)
            test_data = fe_obj.transform(test_data)    
            
            train_data.to_csv("train_data.csv", index=False)
            test_data.to_csv("test_data.csv", index=False)

            logging.info("Obtainig data transformation object")
            processing_obj = self.get_data_trasformation_obj()
            target_columns_name = "Time_taken (min)"
            
            x_train = train_data.drop(target_columns_name, axis=1)
            y_train = train_data[target_columns_name]
            
            x_test = test_data.drop(target_columns_name, axis=1)
            y_test = test_data[target_columns_name]

        
            x_train = processing_obj.fit_transform(x_train)
            x_test = processing_obj.transform(x_test)

            train_arr = np.c_[x_train, np.array(y_train)]
            test_arr = np.c_[x_test, np.array(y_test)]

            data_train = pd.DataFrame(train_arr)
            data_test = pd.DataFrame(test_arr)

            os.makedirs(os.path.dirname(self.DataTransformationConfig.transformed_train_path), exist_ok=True)
            data_train.to_csv(self.DataTransformationConfig.transformed_train_path, index=False)
            data_test.to_csv(self.DataTransformationConfig.transformed_test_path, index=False)

            logging.info("Data Transformation Completed")
            
            sav_obj(file_path=self.DataTransformationConfig.feature_eng_obj_path, obj=fe_obj)
            sav_obj(file_path=self.DataTransformationConfig.processed_obj_file_path, obj=processing_obj)
       
            return (train_arr, test_arr, self.DataTransformationConfig.processed_obj_file_path)
        
        except Exception as e:
            raise CustomException(e, sys)