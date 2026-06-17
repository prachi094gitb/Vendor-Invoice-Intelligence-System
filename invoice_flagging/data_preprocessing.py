import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_invoice_data():
    conn=sqlite3.connect(r'C:\Users\prach\Documents\ML Projects\VIIS\inventory.db')

    query = """
     
    WITH purchase_agg AS (
        SELECT p.PONumber, COUNT(DISTINCT Brand) as total_Brands,
        SUM(p.Quantity) as total_item_quantity,
        SUM(p.Dollars) as total_item_dollars,
        AVG(JULIANDAY(p.ReceivingDate) - JULIANDAY(p.PODate)) as avg_receiving_delay
        FROM purchases p
        GROUP BY p.PONumber
    )
    SELECT 
    v.PONumber,
    v.Quantity as invoice_quantity,
    v.Dollars as invoice_dollars,
    v.Freight,
    (JULIANDAY(v.InvoiceDate) - JULIANDAY(v.PODate)) AS days_po_to_invoice,
    (JULIANDAY(v.PayDate) - JULIANDAY(v.InvoiceDate)) AS days_to_pay,
    pa.total_brands,
    pa.total_item_quantity,
    pa.total_item_dollars,
    pa.avg_receiving_delay

    FROM vendor_invoice v
    LEFT JOIN purchase_agg pa
    ON v.PONumber = pa.PONumber  
     
    """
    df=pd.read_sql_query(query, conn)
    conn.close()
    return df

def create_invoice_risk_label(row):
    if(abs(row["invoice_dollars"] - row["total_item_dollars"])>5):
        return 1
    
    if row["avg_receiving_delay"]>10:
        return 1
    
    return 0

def apply_labels(df):
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df

def split_data(df, features, target):
    X= df[features]
    y=df[target]

    return train_test_split(
        X, y, test_size = 0.20, random_state=42
    )

def scale_features(X_train, X_test, scaler_path):
    scaler= StandardScaler()
    X_train_scaled= scaler.fit_transform(X_train)
    X_test_scaled= scaler.transform(X_test)

    joblib.dump(scaler, scaler_path)
    return X_train_scaled, X_test_scaled
