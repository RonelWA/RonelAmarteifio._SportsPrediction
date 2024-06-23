from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load trained model
loaded_rfr_model = joblib.load('final_rfr_model.pkl')
scaler = StandardScaler()

# Example of selected features used during training
selected_features = [
    'movement_reactions', 'potential', 'passing', 'wage_eur', 'value_eur',
    'dribbling', 'attacking_short_passing', 'international_reputation', 'skill_long_passing',
    'physic', 'age', 'skill_ball_control', 'shooting', 'skill_curve','weak_foot','skill_moves','skill_dribbling','attacking_finishing'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive JSON data from POST request
        input_data = request.json
        
        # Example: Convert JSON to DataFrame and process data as needed
        input_df = pd.DataFrame(input_data, index=[0])
        X = input_df[selected_features]
        X_scaled = scaler.transform(X)
        
        # Make predictions using the loaded model
        predictions = loaded_rfr_model.predict(X_scaled)

        # Prepare response as JSON
        response = {
            'predictions': predictions.tolist()  # Convert to list if needed
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
