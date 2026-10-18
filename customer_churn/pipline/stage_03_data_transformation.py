import sys
from customer_churn.exception import Customer_churn_Exception
from customer_churn.logger import logging
from customer_churn.configuration.configuration import ConfigurationManager
from customer_churn.components.data_transformation import DataTransformation
from customer_churn.entity.artifact_entity import DataValidationArtifact
from pathlib import Path

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_validation_config = config.get_data_validation_config()
            
            # Reconstruct DataValidationArtifact by reading status.txt
            status_file_path = data_validation_config.status_file
            with open(status_file_path, 'r') as f:
                content = f.read()
            validation_status = "True" in content

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                status_file_path=status_file_path
            )

            data_transformation = DataTransformation(config=data_transformation_config, data_validation_artifact=data_validation_artifact)
            data_transformation.initiate_data_transformation()
            
        except Exception as e:
            raise Customer_churn_Exception(e, sys)

if __name__ == '__main__':
    try:
        logging.info(">>>>>> Data Transformation stage started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logging.info(">>>>>> Data Transformation stage completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
