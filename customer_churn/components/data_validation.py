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

    def detect_data_drift(self):
        try:
            import pandas as pd
            from evidently import Report
            from evidently.presets import DataDriftPreset
            
            data = pd.read_csv(self.data_ingestion_artifact.data_file_path)
            
            # Simulate drift checking by splitting the dataset
            # (In production, 'reference_data' would be historical training data)
            reference_data = data.sample(frac=0.5, random_state=42)
            current_data = data.drop(reference_data.index)
            
            report = Report(metrics=[DataDriftPreset()])
            snapshot = report.run(reference_data=reference_data, current_data=current_data)
            
            # Save the report as HTML for visual appeal
            html_report_path = os.path.join(self.config.root_dir, "drift_report.html")
            snapshot.save_html(html_report_path)
            
            # Save as JSON for DVC/programmatic tracking
            json_report_path = os.path.join(self.config.root_dir, "drift_report.json")
            with open(json_report_path, "w") as f:
                f.write(snapshot.json())
            
            logging.info(f"Data drift reports saved at: {html_report_path} and {json_report_path}")
            
        except Exception as e:
            raise Customer_churn_Exception(e, sys)
