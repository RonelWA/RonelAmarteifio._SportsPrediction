# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained models
model_directory = '/content/drive/My Drive/Colab Notebooks/'

# Load Random Forest model
filename_rf = model_directory + 'random_forest_model.sav'
rf_model = joblib.load(filename_rf)

# Load XGBoost model
filename_xgb = model_directory + 'xgboost_model.sav'
xgb_model = joblib.load(filename_xgb)

# Load Gradient Boosting model
filename_gb = model_directory + 'gradient_boosting_model.sav'
gb_model = joblib.load(filename_gb)

# Function to predict using models
def predict_rating(model, data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    prediction = model.predict(scaled_data)
    return prediction

# Main function to run the app
def main():
    st.title('FIFA Player Rating Prediction')

    # Sidebar with user input
    st.sidebar.header('Input Features')
    
    # Example input features (can be adjusted based on your input requirements)
    movement_reactions = st.sidebar.slider('Movement Reactions', min_value=0, max_value=100, value=50)
    potential = st.sidebar.slider('Potential', min_value=0, max_value=100, value=50)
    passing = st.sidebar.slider('Passing', min_value=0, max_value=100, value=50)
    wage_eur = st.sidebar.number_input('Wage (EUR)', min_value=0, value=100000)
    value_eur = st.sidebar.number_input('Value (EUR)', min_value=0, value=1000000)
    dribbling = st.sidebar.slider('Dribbling', min_value=0, max_value=100, value=50)
    attacking_short_passing = st.sidebar.slider('Attacking Short Passing', min_value=0, max_value=100, value=50)
    international_reputation = st.sidebar.slider('International Reputation', min_value=0, max_value=5, value=3)
    skill_long_passing = st.sidebar.slider('Skill Long Passing', min_value=0, max_value=100, value=50)
    physic = st.sidebar.slider('Physic', min_value=0, max_value=100, value=50)
    age = st.sidebar.number_input('Age', min_value=15, max_value=50, value=25)
    skill_ball_control = st.sidebar.slider('Skill Ball Control', min_value=0, max_value=100, value=50)
    shooting = st.sidebar.slider('Shooting', min_value=0, max_value=100, value=50)
    skill_curve = st.sidebar.slider('Skill Curve', min_value=0, max_value=100, value=50)
    weak_foot = st.sidebar.slider('Weak Foot', min_value=1, max_value=5, value=3)
    skill_moves = st.sidebar.slider('Skill Moves', min_value=1, max_value=5, value=3)
    skill_dribbling = st.sidebar.slider('Skill Dribbling', min_value=0, max_value=100, value=50)
    attacking_finishing = st.sidebar.slider('Attacking Finishing', min_value=0, max_value=100, value=50)

    # Collect input data into a DataFrame
    input_data = pd.DataFrame({
        'movement_reactions': [movement_reactions],
        'potential': [potential],
        'passing': [passing],
        'wage_eur': [wage_eur],
        'value_eur': [value_eur],
        'dribbling': [dribbling],
        'attacking_short_passing': [attacking_short_passing],
        'international_reputation': [international_reputation],
        'skill_long_passing': [skill_long_passing],
        'physic': [physic],
        'age': [age],
        'skill_ball_control': [skill_ball_control],
        'shooting': [shooting],
        'skill_curve': [skill_curve],
        'weak_foot': [weak_foot],
        'skill_moves': [skill_moves],
        'skill_dribbling': [skill_dribbling],
        'attacking_finishing': [attacking_finishing]
    })

    # Predict using models
    rf_prediction = predict_rating(rf_model, input_data)
    xgb_prediction = predict_rating(xgb_model, input_data)
    gb_prediction = predict_rating(gb_model, input_data)

    st.write('### Predicted Ratings:')
    st.write(f"- Random Forest: {rf_prediction[0]:.2f}")
    st.write(f"- XGBoost: {xgb_prediction[0]:.2f}")
    st.write(f"- Gradient Boosting: {gb_prediction[0]:.2f}")

# Run the app
if __name__ == '__main__':
    main()
