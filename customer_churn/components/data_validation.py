import os
import sys
import pandas as pd
from customer_churn.exception import Customer_churn_Exception
from customer_churn.logger import logging
from customer_churn.entity.config_entity import DataValidationConfig
from customer_churn.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

class DataValidation:
    def __init__(self, config: DataValidationConfig, data_ingestion_artifact: DataIngestionArtifact):
        self.config = config
        self.data_ingestion_artifact = data_ingestion_artifact

    def validate_all_columns(self) -> DataValidationArtifact:
        try:
            validation_status = None

            data = pd.read_csv(self.data_ingestion_artifact.data_file_path)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.status_file, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                    break
                else:
                    validation_status = True
                    with open(self.config.status_file, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return DataValidationArtifact(
                validation_status=validation_status,
                status_file_path=self.config.status_file
            )

        except Exception as e:
            raise Customer_churn_Exception(e, sys)
