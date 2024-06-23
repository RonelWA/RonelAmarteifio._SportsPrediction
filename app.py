from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, template_folder='C:\\Users\\ronel\\OneDrive\\Desktop\\templates')

# Load your trained models (ensure paths are correct)
loaded_rfr_model = joblib.load('final_rfr_model.pkl')
loaded_xgb_model = joblib.load('final_xgb_model.pkl')
loaded_gb_model = joblib.load('final_gb_model.pkl')

# Define a function to process user input and make predictions
def predict_rating(input_features):
    # Preprocess input_features to match model requirements
    # Example: Convert input_features into numpy array and scale if needed
    # Make predictions using loaded models
    rf_prediction = loaded_rfr_model.predict(input_features)[0]
    xgb_prediction = loaded_xgb_model.predict(input_features)[0]
    gb_prediction = loaded_gb_model.predict(input_features)[0]
    return rf_prediction, xgb_prediction, gb_prediction

# Define routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Extract input data from form submission
        feature1 = float(request.form['feature1'])
        # Process input features into a format suitable for prediction
        input_data = np.array([[feature1]])  # Example for single feature input
        # Get predictions
        rf_pred, xgb_pred, gb_pred = predict_rating(input_data)
        # Return prediction results to display
        return render_template('index.html', prediction=rf_pred, confidence='High')  # Example: confidence score
    # Render default template if method is GET
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
