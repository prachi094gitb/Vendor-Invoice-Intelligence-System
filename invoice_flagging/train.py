from data_preprocessing import load_invoice_data, create_invoice_risk_label, apply_labels, split_data, scale_features
from model_evaluation import train_random_forest, evaluate_model
import joblib
from pathlib import Path

FEATURES = [
    "invoice_quantity", "invoice_dollars", "Freight", "total_item_quantity", "total_item_dollars"
]

TARGET = "flag_invoice"

def main():

    from pathlib import Path
    model_dir= Path("../models")
    model_dir.mkdir(exist_ok=True)

    #Load data
    df= load_invoice_data()
    df= apply_labels(df)

    #Prepare data
    scaler_path = model_dir / 'invoice_scaler.pkl'
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, scaler_path)

    #Train and evaluate model
    randomized_search = train_random_forest(X_train_scaled, y_train)

    evaluate_model(
        randomized_search.best_estimator_,
        X_test_scaled, y_test, "Random Forest Classifier"
    )

    #Save best model
    model_path = model_dir/ "predict_invoice_flag.pkl"
    joblib.dump(randomized_search.best_estimator_, model_path)

if __name__ == "__main__":
    main()