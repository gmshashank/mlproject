import os
import sys
import numpy as np
import pandas as pd
import dill

from pathlib import Path  
file = Path(__file__).resolve()  
package_root_directory = file.parents[1]  
print(package_root_directory)
sys.path.append(str(package_root_directory))  

from exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]

            param=params[list(models.keys())[i]]

            grid_searchcv=GridSearchCV(model,param,cv=3)
            grid_searchcv.fit(X_train,y_train)

            model.set_params(**grid_searchcv.best_params_)

            # Train the model
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            train_model_score = r2_score(y_train, y_train_pred)

            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
