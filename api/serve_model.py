from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Load the trained model from the specified path
model = joblib.load('models/trained_models/logistic_regression_model.joblib')

# Define the API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Ensure the data is in the expected format
        if not data or 'features' not in data:
            return jsonify({"error": "Invalid input data"}), 400

        # Extract the features and make prediction
        features = np.array(data['features']).reshape(1, -1)  # Reshape if necessary
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)[:, 1]  # Assuming binary classification

        # Log the prediction and return the result
        logger.info(f"Prediction: {prediction}, Probability: {prediction_proba[0]}")
        return jsonify({'prediction': int(prediction[0]), 'probability': float(prediction_proba[0])})

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
