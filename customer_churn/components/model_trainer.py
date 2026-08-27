import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from customer_churn.exception import Customer_churn_Exception
from customer_churn.logger import logging
from customer_churn.entity.config_entity import ModelTrainerConfig
from customer_churn.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from customer_churn.utils.main_utils import save_object

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        self.config = config
        self.data_transformation_artifact = data_transformation_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading train and test data for model training.")
            train_data = pd.read_csv(self.data_transformation_artifact.train_data_path, header=None)
            test_data = pd.read_csv(self.data_transformation_artifact.test_data_path, header=None)

            X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
            X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

            models = {
                "RandomForest": RandomForestClassifier(**self.config.params.get('RandomForest', {}), random_state=42),
                "XGBoost": XGBClassifier(**self.config.params.get('XGBoost', {}), random_state=42),
                "LogisticRegression": LogisticRegression(**self.config.params.get('LogisticRegression', {}), random_state=42)
            }

            model_report = {}
            logging.info("Training and evaluating multiple models...")
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                model_report[model_name] = acc
                logging.info(f"{model_name} Accuracy: {acc}")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with Accuracy: {best_model_score}")

            model_file_path = os.path.join(self.config.root_dir, self.config.model_name)
            logging.info(f"Saving best model to {model_file_path}")
            
            # Save the best model dictionary including name so evaluation knows which params to log
            model_info = {
                "model_name": best_model_name,
                "model_object": best_model
            }
            save_object(
                file_path=model_file_path,
                obj=model_info
            )

            return ModelTrainerArtifact(
                trained_model_file_path=model_file_path
            )
            
        except Exception as e:
            raise Customer_churn_Exception(e, sys)
