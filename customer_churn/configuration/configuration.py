import os
import sys
from pathlib import Path
from customer_churn.constants import *
from customer_churn.utils.main_utils import read_yaml, create_directories
from customer_churn.entity.config_entity import (DataIngestionConfig, 
                                                 DataValidationConfig,
                                                 DataTransformationConfig,
                                                 ModelTrainerConfig,
                                                 ModelEvaluationConfig)
from customer_churn.exception import Customer_churn_Exception

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = "config/config.yaml",
        params_filepath = "config/params.yaml",
        schema_filepath = "config/schema.yaml"):

        self.config = read_yaml(Path(config_filepath))
        self.params = read_yaml(Path(params_filepath))
        self.schema = read_yaml(Path(schema_filepath))

        create_directories([self.config['artifacts_root']])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            config = self.config['data_ingestion']
            create_directories([config['root_dir']])

            data_ingestion_config = DataIngestionConfig(
                root_dir=Path(config['root_dir']),
                local_data_file=Path(config['local_data_file']),
            )
            return data_ingestion_config
        except Exception as e:
            raise Customer_churn_Exception(e, sys)

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            config = self.config['data_validation']
            create_directories([config['root_dir']])

            data_validation_config = DataValidationConfig(
                root_dir=Path(config['root_dir']),
                status_file=Path(config['status_file']),
                all_schema=self.schema['columns']
            )
            return data_validation_config
        except Exception as e:
            raise Customer_churn_Exception(e, sys)

    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            config = self.config['data_transformation']
            create_directories([config['root_dir']])

            data_transformation_config = DataTransformationConfig(
                root_dir=Path(config['root_dir']),
                data_path=Path(config['data_path']),
                preprocessor_obj_file_path=Path(config['preprocessor_obj_file_path']),
                train_data_path=Path(config['train_data_path']),
                test_data_path=Path(config['test_data_path'])
            )
            return data_transformation_config
        except Exception as e:
            raise Customer_churn_Exception(e, sys)

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            config = self.config['model_trainer']
            create_directories([config['root_dir']])

            model_trainer_config = ModelTrainerConfig(
                root_dir=Path(config['root_dir']),
                train_data_path=Path(config['train_data_path']),
                test_data_path=Path(config['test_data_path']),
                model_name=config['model_name'],
                target_column=self.schema['target_column']['name'],
                params=self.params
            )
            return model_trainer_config
        except Exception as e:
            raise Customer_churn_Exception(e, sys)

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            config = self.config['model_evaluation']
            create_directories([config['root_dir']])

            model_evaluation_config = ModelEvaluationConfig(
                root_dir=Path(config['root_dir']),
                test_data_path=Path(config['test_data_path']),
                model_path=Path(config['model_path']),
                preprocessor_path=Path(config['preprocessor_path']),
                metric_file_name=Path(config['metric_file_name']),
                target_column=self.schema['target_column']['name']
            )
            return model_evaluation_config
        except Exception as e:
            raise Customer_churn_Exception(e, sys)
