# loan_approval_function.py


import pandas as pd
import pickle as pk

def preprocess_and_predict(input_data, trained_model, threshold=0.37):

    # Binary encoding
    encoder = pk.load(open("binary_encoder.pkl", "rb"))
    input_data_encoded = encoder.transform(input_data)

    # Feature engineering (add_combined_feature)
    input_data_encoded['LTV'] = input_data_encoded['LOAN'] / input_data_encoded['VALUE']

    # Scaling
    scaler = pk.load(open("robust_scaler.pkl", "rb"))
    input_data_scaled = pd.DataFrame(scaler.transform(input_data_encoded), columns = input_data_encoded.columns)

    # Loading our model
    trained_model = pk.load(open("loan_approval_model.pkl", "rb"))


    # Adjust predictions based on the custom threshold
    predicted_probabilities = trained_model.predict_proba(input_data_scaled)[:, 1]
    adjusted_predictions = (predicted_probabilities >= threshold).astype(int)

    return adjusted_predictions
