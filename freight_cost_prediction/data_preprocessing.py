import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
 

#Data Collection
def load_vendor_invoice_data(db_path : str):
    conn=sqlite3.connect(db_path)
    query= "SELECT * FROM vendor_invoice"
    df=pd.read_sql_query(query,conn)
    conn.close()
    return df

#Data Preparation
def  prepare_features(df : pd.DataFrame):
    X = df[['Dollars']]
    y = df['Freight']
    return X,y

#Split Dataset
def split_data(X, y, test_size=0.20, random_state=42):
    return train_test_split(X, y, test_size=test_size,random_state=random_state)





