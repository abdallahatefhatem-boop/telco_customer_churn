import sys
from customer_churn.exception import Customer_churn_Exception
from customer_churn.logger import logging
from customer_churn.configuration.configuration import ConfigurationManager
from customer_churn.components.data_ingestion import DataIngestion

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_data()
        except Exception as e:
            raise Customer_churn_Exception(e, sys)

if __name__ == '__main__':
    try:
        logging.info(">>>>>> Data Ingestion stage started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logging.info(">>>>>> Data Ingestion stage completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
