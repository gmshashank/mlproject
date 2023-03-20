import sys
import os
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pathlib import Path  
file = Path(__file__).resolve()  
package_root_directory = file.parents[1]  
print(package_root_directory)
sys.path.append(str(package_root_directory))  

from exception import CustomException
from logger import logging
from utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessing_obj_file_path = os.path.join("artifacts", "pre_processor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is used for Data Transformation
        """
        try:
            numerical_features = ["reading_score", "writing_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical columns: {numerical_features}")

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_features),
                    (
                        "categorical_pipeline",
                        categorical_pipeline,
                        categorical_features,
                    ),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test Data successfully imported")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_features = ["reading_score", "writing_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training dataframe and test dataframe"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessing_obj_file_path,
                obj=preprocessing_obj,
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessing_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
