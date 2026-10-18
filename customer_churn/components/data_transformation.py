import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from customer_churn.exception import Customer_churn_Exception
from customer_churn.logger import logging
from customer_churn.entity.config_entity import DataTransformationConfig
from customer_churn.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from customer_churn.utils.main_utils import save_object

class DataTransformation:
    def __init__(self, config: DataTransformationConfig, data_validation_artifact: DataValidationArtifact):
        self.config = config
        self.data_validation_artifact = data_validation_artifact

    def get_data_transformer_object(self):
        try:
            # We don't include Churn (target) or customerID
            numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
            categorical_columns = [
                'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                'PhoneService', 'MultipleLines', 'InternetService', 
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies', 
                'Contract', 'PaperlessBilling', 'PaymentMethod'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(drop='first'))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise Customer_churn_Exception(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception("Data Validation failed.")

            data = pd.read_csv(self.config.data_path)
            
            logging.info("Cleaning data: converting TotalCharges to numeric")
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

            logging.info("Dropping customerID")
            if 'customerID' in data.columns:
                data = data.drop(columns=['customerID'])

            # Target variable mapping
            logging.info("Encoding target variable")
            data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

            # Splitting data
            logging.info("Splitting data into train and test sets")
            train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

            target_column_name = "Churn"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")
            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save transformed data as csv (for easy access/debugging)
            pd.DataFrame(train_arr).to_csv(self.config.train_data_path, index=False, header=False)
            pd.DataFrame(test_arr).to_csv(self.config.test_data_path, index=False, header=False)

            logging.info(f"Saved preprocessor object.")
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return DataTransformationArtifact(
                preprocessor_obj_file_path=self.config.preprocessor_obj_file_path,
                train_data_path=self.config.train_data_path,
                test_data_path=self.config.test_data_path
            )
            
        except Exception as e:
            raise Customer_churn_Exception(e, sys)
