import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.studmlproject.exception import CustomException
from src.studmlproject.logger import logging
from src.studmlproject.utils import save_object,evaluate_models


@dataclass
class ModelTrainerConfig:  # path to save best model
    train_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            #split data into train test
            logging.info("Split Train Test input data")
            X_train, X_test, y_train, y_test = (
            train_array[:, :-1],  # Features for training
            test_array[:, :-1],   # Features for testing
            train_array[:, -1],   # Labels for training
            test_array[:, -1]     # Labels for testing
            )

            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor(),
                "Linear Regression": LinearRegression(),
                "CatBoostRegressor": CatBoostRegressor(),
            }

            logging.info("Hyperparameter Tuning")

            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            logging.info("Evaluate Model")
            model_report:dict=evaluate_models(X_train,y_train,X_test, y_test, models,params)
            logging.info("Best Model score and Name")

            best_model_score = max(sorted(model_report.values()))

            #To get name of best model

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            # If accuracy of model is less than 60% raise exception
            if best_model_score<0.6:
                raise CustomException("No Best Model found")
            logging.info(f"Found best model on both train and test dataset")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square =r2_score(y_test,predicted)
            return r2_square


        except Exception as e:
            raise CustomException(e,sys)