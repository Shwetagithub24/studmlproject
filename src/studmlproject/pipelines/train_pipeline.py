import os
import sys
import pandas as pd
from src.studmlproject.exception import CustomException
from src.studmlproject.logger import logging
from src.studmlproject.utils import save_object
from src.studmlproject.components.data_ingestion import DataIngestion
from src.studmlproject.components.data_transformation import DataTransformation
from src.studmlproject.components.model_tranier import ModelTrainer

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        
    def run_pipeline(self):
        try:
            # Step 1: Data Ingestion
            logging.info("Starting data ingestion...")
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            print("Completed data ingestion...")

            # Step 2: Data Transformation
            logging.info("Starting data transformation...")
            train_data, test_data, preprocessor = self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            print("Completed data transformation...")

            # Save the preprocessor for use in the prediction pipeline
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            save_object(preprocessor_path, preprocessor)
            logging.info(f"Preprocessor saved at {preprocessor_path}")
            print("Preprocessor.pkl saved")

            # Step 3: Model Training
            logging.info("Starting model training...")
            model = self.model_trainer.initiate_model_trainer(train_data, test_data)
            print("model training started")

            # Save the trained model
            model_path = os.path.join('artifacts', 'model.pkl')
            save_object(model_path, model)
            logging.info(f"Model saved at {model_path}")
            print("model.pkl saved")
            
            logging.info("Training pipeline completed successfully.")
            
        except Exception as e:
            logging.error("Error occurred during the training pipeline.")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
