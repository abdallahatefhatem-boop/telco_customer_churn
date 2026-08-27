import os
import sys
import pandas as pd
import numpy as np
import json
import dagshub
import mlflow
import mlflow.sklearn
import mlflow.onnx
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxmltools
from customer_churn.exception import Customer_churn_Exception
from customer_churn.logger import logging
from customer_churn.entity.config_entity import ModelEvaluationConfig
from customer_churn.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact, ModelEvaluationArtifact
from customer_churn.utils.main_utils import load_object, read_yaml

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, data_transformation_artifact: DataTransformationArtifact, model_trainer_artifact: ModelTrainerArtifact):
        self.config = config
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact

    def eval_metrics(self, actual, pred):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(actual, pred)
        prec = precision_score(actual, pred)
        rec = recall_score(actual, pred)
        f1 = f1_score(actual, pred)
        return acc, prec, rec, f1

    def convert_to_onnx(self, model_object, model_name, num_features, output_path):
        try:
            logging.info(f"Converting {model_name} to ONNX format...")
            initial_type = [('float_input', FloatTensorType([None, num_features]))]
            
            if model_name == "XGBoost":
                onnx_model = onnxmltools.convert_xgboost(model_object, initial_types=initial_type)
            else:
                onnx_model = convert_sklearn(model_object, initial_types=initial_type, target_opset=12)
                
            onnx.save(onnx_model, output_path)
            logging.info(f"ONNX model saved at {output_path}")
            return onnx_model
        except Exception as e:
            logging.error(f"Error converting to ONNX: {str(e)}")
            raise

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Loading test data for evaluation.")
            test_data = pd.read_csv(self.config.test_data_path, header=None)
            X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]
            num_features = X_test.shape[1]

            logging.info("Loading trained model info.")
            model_info = load_object(self.config.model_path)
            model_name = model_info['model_name']
            model_object = model_info['model_object']

            logging.info("Initializing DagsHub/MLflow integration.")
            dagshub.init(repo_owner='abdallahatefhatem', repo_name='telco_customer_churn', mlflow=True)
            mlflow.set_registry_uri(self.config.mlflow_uri)
            tracking_url_type_store = "https"

            logging.info("Starting MLflow run.")
            with mlflow.start_run():
                predicted_qualities = model_object.predict(X_test)
                acc, prec, rec, f1 = self.eval_metrics(y_test, predicted_qualities)

                metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
                with open(self.config.metric_file_name, 'w') as f:
                    json.dump(metrics, f)

                # Log parameters specifically for the best model
                all_params = read_yaml(os.path.join("config", "params.yaml"))
                params = all_params.get(model_name, {})
                mlflow.log_params(params)
                mlflow.log_param("best_model_name", model_name)

                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)

                # Convert to ONNX and log
                onnx_output_path = os.path.join(self.config.root_dir, "model.onnx")
                onnx_model = self.convert_to_onnx(model_object, model_name, num_features, onnx_output_path)
                
                # Log the ONNX model to MLflow
                if tracking_url_type_store != "file":
                    mlflow.onnx.log_model(onnx_model, "onnx_model", registered_model_name=f"{model_name}_ONNX")
                else:
                    mlflow.onnx.log_model(onnx_model, "onnx_model")

            logging.info(f"Model evaluation complete. Metrics: {metrics}")

            return ModelEvaluationArtifact(
                is_model_accepted=True,
                metric_file_path=self.config.metric_file_name
            )

        except Exception as e:
            raise Customer_churn_Exception(e, sys)
