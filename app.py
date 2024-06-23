import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained models
loaded_rfr_model = joblib.load('final_rfr_model.pkl')
loaded_xgb_model = joblib.load('final_xgb_model.pkl')
loaded_gb_model = joblib.load('final_gb_model.pkl')

# Function to preprocess new input data
def preprocess_input(input_df):
    selected_features = [
        'movement_reactions', 'potential', 'passing', 'wage_eur', 'value_eur',
        'dribbling', 'attacking_short_passing', 'international_reputation', 'skill_long_passing',
        'physic', 'age', 'skill_ball_control', 'shooting', 'skill_curve', 'weak_foot', 'skill_moves',
        'skill_dribbling', 'attacking_finishing'
    ]
    scaler = StandardScaler()
    X = input_df[selected_features]
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Main function to run the Streamlit app
def main():
    st.title('Football Player Rating Prediction')

    # Sidebar for user input
    st.sidebar.header('Input Features')
    # Example: You can create input fields for each feature
    # For simplicity, assuming you provide a CSV file as an input
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file for prediction", type="csv")

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

        # Preprocess the input data
        X_input = preprocess_input(input_df)

        # Get predictions from each model
        rf_prediction = loaded_rfr_model.predict(X_input)
        xgb_prediction = loaded_xgb_model.predict(X_input)
        gb_prediction = loaded_gb_model.predict(X_input)

        # Show predictions
        st.subheader('Predicted Ratings:')
        st.write('Random Forest Prediction:', rf_prediction)
        st.write('XGBoost Prediction:', xgb_prediction)
        st.write('Gradient Boosting Prediction:', gb_prediction)

        # Confidence scores (optional): You can also display confidence scores or other metrics

        # Show original input data (optional)
        st.subheader('Input Data:')
        st.write(input_df)

if __name__ == '__main__':
    main()
