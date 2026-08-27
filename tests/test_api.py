def test_read_main(client):
    """Test if the main page loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Customer Churn Prediction" in response.text
    assert "<form action=\"/predict\"" in response.text

def test_predict_endpoint(client, sample_form_payload):
    """Test the prediction endpoint with valid payload."""
    response = client.post("/predict", data=sample_form_payload)
    
    assert response.status_code == 200
    assert "Prediction:" in response.text

def test_predict_endpoint_missing_data(client):
    """Test the prediction endpoint with missing form data."""
    # Sending empty form data should trigger FastAPI's 422 Unprocessable Entity
    response = client.post("/predict", data={})
    assert response.status_code == 422
