import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained models
model_directory = '/mnt/data/'
random_forest_model = joblib.load(model_directory + 'random_forest_model.sav')
xgboost_model = joblib.load(model_directory + 'xgboost_model.sav')
gradient_boosting_model = joblib.load(model_directory + 'gradient_boosting_model.sav')

# Define feature list
selected_features = [
    'movement_reactions', 'potential', 'passing', 'wage_eur', 'value_eur',
    'dribbling', 'attacking_short_passing', 'international_reputation', 'skill_long_passing',
    'physic', 'age', 'skill_ball_control', 'shooting', 'skill_curve', 'weak_foot',
    'skill_moves', 'skill_dribbling', 'attacking_finishing'
]

# Streamlit app
st.title("FIFA Player Rating Prediction")

# Input form
st.header("Player Information")
input_data = {}
for feature in selected_features:
    input_data[feature] = st.number_input(f"Enter {feature.replace('_', ' ')}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Standardize input
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_df)

# Make predictions
rf_prediction = random_forest_model.predict(input_scaled)[0]
xgb_prediction = xgboost_model.predict(input_scaled)[0]
gb_prediction = gradient_boosting_model.predict(input_scaled)[0]

# Display predictions
st.header("Predicted Player Ratings")
st.write(f"Random Forest Prediction: {rf_prediction:.2f}")
st.write(f"XGBoost Prediction: {xgb_prediction:.2f}")
st.write(f"Gradient Boosting Prediction: {gb_prediction:.2f}")
