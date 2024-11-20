from flask import Flask, request, jsonify
import joblib
import logging

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the pre-trained model
MODEL_PATH = 'models/trained_models/logistic_regression_model.joblib'
model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON request
        data = request.get_json()
        if not data:
            raise ValueError("No JSON data provided in the request.")

        # Extract 'features' from the request data
        features = data.get('features')
        if not features:
            raise ValueError("Missing 'features' in request data")

        # Ensure that features is a list or array
        if not isinstance(features, list):
            raise ValueError("'features' must be a list of numerical values.")

        # Perform prediction
        prediction = model.predict([features])
        response = {'prediction': prediction[0]}

        logging.info(f"Prediction made successfully: {response}")
        return jsonify(response)
    
    except ValueError as e:
        logging.error(f"Input validation error: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Unexpected error during prediction: {e}")
        return jsonify({'error': "An unexpected error occurred."}), 500

# Default route
@app.route('/')
def home():
    return jsonify({'message': 'Fraud Detection API is running'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
