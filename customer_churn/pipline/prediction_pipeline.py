import os
import sys
import pandas as pd
from customer_churn.exception import Customer_churn_Exception
from customer_churn.utils.main_utils import load_object

import onnxruntime as rt
import numpy as np

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            onnx_model_path = os.path.join("artifacts", "model_evaluation", "model.onnx")
            preprocessor_path = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")
            
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            
            # Predict using ONNX runtime
            sess = rt.InferenceSession(onnx_model_path)
            input_name = sess.get_inputs()[0].name
            label_name = sess.get_outputs()[0].name
            
            # Convert to float32 as ONNX model expects float input
            preds_onnx = sess.run([label_name], {input_name: data_scaled.astype(np.float32)})[0]
            
            return preds_onnx
            
        except Exception as e:
            raise Customer_churn_Exception(e, sys)

class CustomData:
    def __init__(self,
        gender: str,
        SeniorCitizen: int,
        Partner: str,
        Dependents: str,
        tenure: int,
        PhoneService: str,
        MultipleLines: str,
        InternetService: str,
        OnlineSecurity: str,
        OnlineBackup: str,
        DeviceProtection: str,
        TechSupport: str,
        StreamingTV: str,
        StreamingMovies: str,
        Contract: str,
        PaperlessBilling: str,
        PaymentMethod: str,
        MonthlyCharges: float,
        TotalCharges: float):
        
        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges]
            }

            return pd.DataFrame(custom_data_input_dict)
            
        except Exception as e:
            raise Customer_churn_Exception(e, sys)
