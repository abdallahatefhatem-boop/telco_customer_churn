import sys
from customer_churn.exception import Customer_churn_Exception
from customer_churn.logger import logging
from customer_churn.configuration.configuration import ConfigurationManager
from customer_churn.components.data_validation import DataValidation
from customer_churn.entity.artifact_entity import DataIngestionArtifact

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            
            # Reconstruct the DataIngestionArtifact from config
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion_artifact = DataIngestionArtifact(
                data_file_path=data_ingestion_config.local_data_file
            )

            data_validation = DataValidation(config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
            data_validation.validate_all_columns()
            
            # Call data drift detection
            data_validation.detect_data_drift()
            
        except Exception as e:
            raise Customer_churn_Exception(e, sys)

if __name__ == '__main__':
    try:
        logging.info(">>>>>> Data Validation stage started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logging.info(">>>>>> Data Validation stage completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
