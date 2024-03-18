import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logger
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor

from src.utils import save_objects
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os, yaml

config_path = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))}/params.yaml"
print(os.path.realpath(__file__))

#Load yaml file:x
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
print(config)
@dataclass
class ModelTrainerConfig:

    #Get Directory path of the current script:
    current_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    #Define artifacts path
    artifact_path = os.path.join(current_directory, "artifacts")

    #Define model.pkl path:
    trained_model_path = os.path.join(artifact_path, "model.pkl")

class InitiateModelTraining:
    def __init__(self) :
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        """
        This function is used to initiate the model training process.
        """
        logger.info("Initiating model training process...")
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )

            models = {
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet(),
                'SVR': SVR(),
                'DecisionTree':DecisionTreeRegressor(random_state=42),
                'RandomForest':RandomForestRegressor(random_state=42),
                'GradientBoostingRegressor':GradientBoostingRegressor(random_state=42),
                'BaggingRegressor' : BaggingRegressor(random_state=42)
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models=models)

            logger.info('\n====================================================================================\n')
            logger.info(f'Model Report : {model_report}')
            logger.info('\n====================================================================================\n')

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            logger.info('\n====================================================================================\n')
            logger.info(f'Best Model: {best_model_name} ## Best Model Score : {best_model_score}')
            logger.info('\n====================================================================================\n')

            best_model = models[best_model_name]

            #Save object:
            save_objects(self.model_trainer_config.trained_model_path, best_model)
        except Exception as e:
            logger.error("Error initiating model training process")
            raise CustomException(e)
