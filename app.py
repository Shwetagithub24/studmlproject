from src.studmlproject.logger import logging
from src.studmlproject.exception import CustomException
from src.studmlproject.components.data_ingestion import DataIngestion
from src.studmlproject.components.data_ingestion import DataIngestionConfig
from src.studmlproject.components.data_transformation import DataTransformationConfig,DataTransformation
from src.studmlproject.components.model_tranier import ModelTrainerConfig,ModelTrainer
from flask import Flask, request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler  #to use pkl file
from src.studmlproject.pipelines.prediction_pipeline import CustomData,PredictPipeline

import sys

application=Flask(__name__)
app=application

#Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():#call this function in home.html
    '''
    #1. get data and do prediction
    '''
    if request.method=='GET':
        return render_template('home.html') 
    else:
        #post request
        data=CustomData(
             gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

    pred_df=data.get_data_as_data_frame()

    print(pred_df)

    predict_pipeline=PredictPipeline()
    results=predict_pipeline.predict(pred_df)
    
    return render_template('home.html', results=results[0])


if __name__=="__main__":
    logging.info("The execution has started")
    app.run(host='0.0.0.0',debug=True)


'''
if __name__=="__main__":
    logging.info("The execution has started")

    try:
        #data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        #data_transformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transormation(train_data_path,test_data_path)

        ## Model Training

        #model_trainer=ModelTrainer()
        #print(model_trainer.initiate_model_trainer(train_arr,test_arr))
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)
    
    '''