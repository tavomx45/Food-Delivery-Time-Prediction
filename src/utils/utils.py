from src.logger.logger import logging
from src.exception.exception import CustomException
import os, sys
import pickle

def sav_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)