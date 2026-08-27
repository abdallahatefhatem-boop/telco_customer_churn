import pytest
from fastapi.testclient import TestClient
from app import app
from customer_churn.pipline.prediction_pipeline import CustomData

@pytest.fixture(scope="module")
def client():
    """Fixture for FastAPI TestClient."""
    with TestClient(app) as c:
        yield c

@pytest.fixture
def sample_customer_data():
    """Fixture for a valid sample CustomData instance."""
    return CustomData(
        gender="Male",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        tenure=12,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="Yes",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="Yes",
        StreamingMovies="Yes",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=89.90,
        TotalCharges=1000.50
    )

@pytest.fixture
def sample_form_payload():
    """Fixture for a valid form payload mimicking UI input."""
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.90,
        "TotalCharges": 1000.50
    }
