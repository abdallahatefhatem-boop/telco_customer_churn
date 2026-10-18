import sys
from customer_churn.exception import Customer_churn_Exception
from customer_churn.logger import logging
from customer_churn.configuration.configuration import ConfigurationManager
from customer_churn.components.data_ingestion import DataIngestion
from customer_churn.components.data_validation import DataValidation
from customer_churn.components.data_transformation import DataTransformation
from customer_churn.components.model_trainer import ModelTrainer
from customer_churn.components.model_evaluation import ModelEvaluation

class TrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def start_data_ingestion(self):
        try:
            data_ingestion_config = self.config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.download_data()
            return data_ingestion_artifact
        except Exception as e:
            raise Customer_churn_Exception(e, sys)

    def start_data_validation(self, data_ingestion_artifact):
        try:
            data_validation_config = self.config_manager.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact = data_validation.validate_all_columns()
            return data_validation_artifact
        except Exception as e:
            raise Customer_churn_Exception(e, sys)

    def start_data_transformation(self, data_validation_artifact):
        try:
            data_transformation_config = self.config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config, data_validation_artifact=data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise Customer_churn_Exception(e, sys)

    def start_model_trainer(self, data_transformation_artifact):
        try:
            model_trainer_config = self.config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise Customer_churn_Exception(e, sys)

    def start_model_evaluation(self, data_transformation_artifact, model_trainer_artifact):
        try:
            model_evaluation_config = self.config_manager.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(
                config=model_evaluation_config, 
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise Customer_churn_Exception(e, sys)

    def run_pipeline(self):
        try:
            logging.info(">>>>>> Data Ingestion started <<<<<<")
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info(">>>>>> Data Ingestion completed <<<<<<")

            logging.info(">>>>>> Data Validation started <<<<<<")
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            logging.info(">>>>>> Data Validation completed <<<<<<")

            if not data_validation_artifact.validation_status:
                raise Exception("Data Validation Failed.")

            logging.info(">>>>>> Data Transformation started <<<<<<")
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            logging.info(">>>>>> Data Transformation completed <<<<<<")

            logging.info(">>>>>> Model Trainer started <<<<<<")
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            logging.info(">>>>>> Model Trainer completed <<<<<<")

            logging.info(">>>>>> Model Evaluation started <<<<<<")
            model_evaluation_artifact = self.start_model_evaluation(data_transformation_artifact, model_trainer_artifact)
            logging.info(">>>>>> Model Evaluation completed <<<<<<")

        except Exception as e:
            raise Customer_churn_Exception(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
