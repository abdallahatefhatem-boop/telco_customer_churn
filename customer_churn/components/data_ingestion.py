import os
import sys
import kagglehub
import pandas as pd
from pathlib import Path
import shutil
from customer_churn.exception import Customer_churn_Exception
from customer_churn.logger import logging
from customer_churn.entity.config_entity import DataIngestionConfig
from customer_churn.entity.artifact_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self) -> DataIngestionArtifact:
        try:
            if os.path.exists(self.config.local_data_file):
                logging.info(f"Data file already exists at {self.config.local_data_file}. Skipping download.")
                return DataIngestionArtifact(data_file_path=self.config.local_data_file)
                
            logging.info("Downloading data from kagglehub...")
            # Download latest version
            path = kagglehub.dataset_download("blastchar/telco-customer-churn")
            logging.info(f"Data downloaded to: {path}")

            path_data = os.listdir(path=path)
            full_path = os.path.join(path, path_data[0])
            
            logging.info("Copying data to artifacts folder...")
            shutil.copy(full_path, self.config.local_data_file)
            logging.info(f"Data copied to: {self.config.local_data_file}")

            return DataIngestionArtifact(data_file_path=self.config.local_data_file)
            
        except Exception as e:
            raise Customer_churn_Exception(e, sys)
