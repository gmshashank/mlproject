import os
import sys

from dataclasses import dataclass

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

from pathlib import Path  
file = Path(__file__).resolve()  
package_root_directory = file.parents[1]  
print(package_root_directory)
sys.path.append(str(package_root_directory))  

from exception import CustomException
from logger import logging

from utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    best_model_score_threshold = 0.6


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting the train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "CatBoosting": CatBoostRegressor(),
                "XGBoost": XGBRegressor(),
            }
            params={
              "Linear Regression":{},
              "K-Neighbors":{
                "n_neighbors":[5,7,9,11],
              },
              "Decision Tree":{
                "criterion":["squared_error","absolute_error","poisson"]
              },
              "Gradient Boosting":{
                "learning_rate":[0.5,0.1,0.05,0.01,0.005,0.001],
                "n_estimators":[8,16,32,64,128,256],
              },
              "Random Forest":{
                "n_estimators":[8,16,32,64,128,256],
              },
              "AdaBoost":{
                "learning_rate":[0.5,0.1,0.05,0.01,0.005,0.001],
                "n_estimators":[8,16,32,64,128,256],
              },
              "CatBoosting":{
                "learning_rate":[0.5,0.1,0.05,0.01,0.005,0.001],
                "n_estimators":[8,16,32,64,128,256],
              },
              "XGBoost":{
                "learning_rate":[0.1,0.05,0.01,0.001],
                "n_estimators":[8,16,32,64,128,256],
              }
            }
            
            model_report: dict = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            # get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < self.model_trainer_config.best_model_score_threshold:
                raise CustomException(
                    "No best model found for thr user defined Performance threshold"
                )

            logging.info("Found Best model on both training and test dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            y_test_predicted = best_model.predict(X_test)
            test_r2_score = r2_score(y_test, y_test_predicted)

            return test_r2_score

        except Exception as e:
            raise CustomException(e, sys)
