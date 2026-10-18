import os
import sys
import yaml
import joblib
from customer_churn.logger import logging
from customer_churn.exception import Customer_churn_Exception
from pathlib import Path

def read_yaml(path_to_yaml: Path) -> dict:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return content
    except Exception as e:
        raise Customer_churn_Exception(e, sys)

def create_directories(path_to_directories: list, verbose=True):
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logging.info(f"created directory at: {path}")
    except Exception as e:
        raise Customer_churn_Exception(e, sys)

def save_object(file_path: Path, obj: object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
        logging.info(f"Object saved successfully at: {file_path}")
    except Exception as e:
        raise Customer_churn_Exception(e, sys)

def load_object(file_path: Path) -> object:
    try:
        obj = joblib.load(file_path)
        logging.info(f"Object loaded successfully from: {file_path}")
        return obj
    except Exception as e:
        raise Customer_churn_Exception(e, sys)
