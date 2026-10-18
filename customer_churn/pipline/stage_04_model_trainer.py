import sys
from customer_churn.exception import Customer_churn_Exception
from customer_churn.logger import logging
from customer_churn.configuration.configuration import ConfigurationManager
from customer_churn.components.model_trainer import ModelTrainer
from customer_churn.entity.artifact_entity import DataTransformationArtifact

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            data_transformation_config = config.get_data_transformation_config()
            
            # Reconstruct DataTransformationArtifact
            data_transformation_artifact = DataTransformationArtifact(
                preprocessor_obj_file_path=data_transformation_config.preprocessor_obj_file_path,
                train_data_path=data_transformation_config.train_data_path,
                test_data_path=data_transformation_config.test_data_path
            )

            model_trainer = ModelTrainer(config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
            model_trainer.initiate_model_trainer()
            
        except Exception as e:
            raise Customer_churn_Exception(e, sys)

if __name__ == '__main__':
    try:
        logging.info(">>>>>> Model Trainer stage started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logging.info(">>>>>> Model Trainer stage completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
