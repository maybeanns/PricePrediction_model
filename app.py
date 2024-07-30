import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import json

# Load the saved model, scaler, and location mapping
@st.cache_resource
def load_model_and_data():
    model = joblib.load('xgb_model.joblib')
    scaler = joblib.load('scaler.joblib')
    with open('location_encoding.json', 'r') as f:
        location_mapping = json.load(f)
    return model, scaler, location_mapping

model, scaler, location_mapping = load_model_and_data()

# Prediction function
def predict_price(beds, area, location):
    price_per_sqft = 0  # We'll update this after the first prediction
    beds_to_area_ratio = beds / area
    features = np.array([[beds, area, location, price_per_sqft, beds_to_area_ratio]])
    scaled_features = scaler.transform(features)
    predicted_price = model.predict(scaled_features)[0]
    
    # Update price_per_sqft and predict again
    price_per_sqft = predicted_price * 1000000 / area
    features = np.array([[beds, area, location, price_per_sqft, beds_to_area_ratio]])
    scaled_features = scaler.transform(features)
    final_predicted_price = model.predict(scaled_features)[0]
    
    return final_predicted_price

# Streamlit app
st.title('Islamabad Property Price Predictor')

# User inputs
area = st.number_input('Area (in square feet)', min_value=100, max_value=10000, value=1000)
beds = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
location = st.selectbox('Location', list(location_mapping.keys()))

if st.button('Predict Price'):
    location_encoded = location_mapping[location]
    price = predict_price(beds, area, location_encoded)
    st.success(f'The predicted price is {price:.2f} million PKR')

# Optional: Add some information about the app
st.info('This app predicts property prices in Islamabad based on area, number of bedrooms, and location.')