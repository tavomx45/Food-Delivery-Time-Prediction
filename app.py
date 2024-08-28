from src.constants.constants import *
from src.logger.logger import logging
from src.exception.exception import CustomException
from src.config.configuration import *
import os, sys
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData
from src.pipeline.training_pipeline import TrainingPipeline
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from Prediction.batch import *


feature_engineering_file_path = FEATURE_ENGG_OBJ_FILE_PATH 
transformer_file_path = PREPROCESING_OBJ_FILE
model_file_path = MODEL_FILE_PATH


UPLOAD_FOLDER = 'batch_prediction/UPLOADED_CSV_FILE'

app = Flask(__name__, template_folder='templates')

ALLOWED_EXTENSIONS = {'csv'}

#Route
@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('form.html')
    else:
        data = CustomData(
            Delivery_person_Age=int(request.form.get('Delivery_person_Age')),
            Delivery_person_Ratings=float(request.form.get('Delivery_person_Ratings')),
            Weather_conditions=request.form.get('Weather_conditions'),
            Road_traffic_density=request.form.get('Road_traffic_density'),
            Vehicle_condition=request.form.get('Vehicle_condition'),
            multiple_dliveries=request.form.get('multiple_dliveries'),
            distance= float(request.form.get('distance')),
            Type_of_order=request.form.get('Type_of_order'),
            Type_of_vehicle=request.form.get('Type_of_vehicle'),
            Festival=request.form.get('Festival'),
            City= request.form.get('City')
        )   

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictionPipeline()
        pred = predict_pipeline.predict(final_new_data)

        result = int(pred[0])
        return render_template('form.html', final_result=result)
    

@app.route('/batch', methods=['GET', 'POST'])
def batch_prediction():
    if request.method == "GET":
        return render_template('batch.html')
    else:
        file = request.files['csv_file']
        directory_path = UPLOAD_FOLDER
        os.makedirs(directory_path, exist_ok=True)

        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            for filename in os.listdir(os.path.join(UPLOAD_FOLDER)):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            #Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            print(file_path)

            logging.info(f"File uploaded successfully at {file_path}")

            batch = batch_prediction(file_path, model_file_path, transformer_file_path, feature_engineering_file_path)
            batch.start_batch_prediction()

            output = "batch prediction done"

            return render_template('batch.html', prediction_result = output, prediction_type='batch')
        
        else:
            return render_template('batch.html', error = "Please upload a valid file", prediction_type='batch')


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == "GET":
        return render_template('train.html')
    else:
        try:
            train_pipeline = TrainingPipeline()
            train_pipeline.main()
            return render_template('train.html', message = "Training done")
    
        except Exception as e:
            logging.error(f"Error in training: {str(e)}")
            return render_template('train.html', error = "Error in training")


if __name__ == '__main__':
    app.run(debug=True)
