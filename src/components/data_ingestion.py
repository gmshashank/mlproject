import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
# print(package_root_directory)
sys.path.append(str(package_root_directory))

from exception import CustomException
from logger import logging
from components.data_transformations import DataTransformation, DataTransformationConfig
from components.model_trainer import ModelTrainerConfig, ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train Test Split initiated")
            train_split, test_split = train_test_split(
                df, test_size=0.2, random_state=42
            )
            train_split.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_split.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("Ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion_obj = DataIngestion()
    train_data, test_data = data_ingestion_obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    model_trainer = ModelTrainer()
    model_r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(model_r2_score)
