import joblib 
import pandas as pd

model_path = "models/predict_invoice_flag.pkl"
scaler_path = "models/invoice_scaler.pkl"

def load_model(model_path : str = model_path):
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def load_scaler():
    return joblib.load(scaler_path)

def predict_invoice_flag(input_data):
    """
    Predict freight cost for new vendor invoices

    Parameters
    -----------
     input_data : dict

    Returns
    --------
    pd.DataFrame with predicted flag
    """
    model = load_model()
    scaler = load_scaler()
    input_df =pd.DataFrame([input_data])
    scaled_input = scaler.transform(input_df)
    input_df['Predicted Flag'] = model.predict(scaled_input)
    return input_df

if __name__ == "__main__":

    sample_invoice = {
        "invoice_quantity" : 120,
        "invoice_dollars" : 2500.0,
        "Freight" : 85.0,
        "total_item_quantity" : 130,
        "total_item_dollars" : 2700
    }
    prediction = predict_invoice_flag(sample_invoice)
    print(prediction)