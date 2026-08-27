import pandas as pd

def test_custom_data_to_dataframe(sample_customer_data):
    """Test if CustomData correctly converts to a pandas DataFrame."""
    df = sample_customer_data.get_data_as_data_frame()
    
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df.columns) == 19
    
    # Check specific values
    assert df["gender"][0] == "Male"
    assert df["SeniorCitizen"][0] == 0
    assert df["MonthlyCharges"][0] == 89.90
    assert df["TotalCharges"][0] == 1000.50
