import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import logging
from flask import Flask, render_template, jsonify

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the pre-trained model
MODEL_PATH = './models/trained_models/logistic_regression_model.joblib'
model = joblib.load(MODEL_PATH)

# Load the fraud data for summary
DATA_PATH = 'data/processed/Fraud__Data.csv'
data = pd.read_csv(DATA_PATH)

# Initialize Dash app
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/')

# Endpoint to display the summary page
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint to get summary statistics
@app.route('/api/summary', methods=['GET'])
def get_summary():
    total_transactions = len(data)
    fraud_cases = len(data[data['class'] == 1])  # Assuming 'class' column is used for fraud label
    fraud_percentage = (fraud_cases / total_transactions) * 100
    
    summary = {
        "total_transactions": total_transactions,
        "fraud_cases": fraud_cases,
        "fraud_percentage": round(fraud_percentage, 2)
    }
    
    return render_template('summary.html', summary=summary)

# Endpoint to predict fraud
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON request
        data = request.get_json()
        features = data.get('features')
        if not features:
            raise ValueError("Missing 'features' in request data")

        # Perform prediction
        prediction = model.predict([features])
        response = {'prediction': prediction[0]}
        
        logging.info(f"Prediction made successfully: {response}")
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 400

# Dash layout
dash_app.layout = html.Div([
    dbc.Row([
        dbc.Col(html.Div("Total Transactions: " + str(len(data)))),
        dbc.Col(html.Div("Fraud Cases: " + str(len(data[data['class'] == 1])))),
        dbc.Col(html.Div("Fraud Percentage: " + str(round((len(data[data['class'] == 1]) / len(data)) * 100, 2)) + "%"))
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(
            id='fraud-trend',
            figure={
                'data': [
                    {'x': data['purchase_time_diff'], 'y': data[data['class'] == 1]['purchase_time_diff'], 'type': 'line', 'name': 'Fraud Cases'},
                ],
                'layout': {
                    'title': 'Fraud Cases Over Time',
                    'xaxis': {'title': 'Time'},
                    'yaxis': {'title': 'Number of Fraud Cases'}
                }
            }
        )),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(
            id='fraud-location',
            figure={
                'data': [
                    {'x': data['ip_address_int'], 'y': data[data['class'] == 1]['ip_address_int'], 'type': 'scatter', 'mode': 'markers'},
                ],
                'layout': {
                    'title': 'Fraud Cases by Location',
                    'xaxis': {'title': 'IP Address'},
                    'yaxis': {'title': 'Number of Fraud Cases'}
                }
            }
        )),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(
            id='fraud-devices',
            figure={
                'data': [
                    {'x': data['device_id'], 'y': data[data['class'] == 1]['device_id'], 'type': 'bar', 'name': 'Fraud by Device'},
                ],
                'layout': {
                    'title': 'Fraud Cases by Device',
                    'xaxis': {'title': 'Device'},
                    'yaxis': {'title': 'Number of Fraud Cases'}
                }
            }
        )),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(
            id='fraud-browsers',
            figure={
                'data': [
                    {'x': ['FireFox', 'IE', 'Opera', 'Safari'], 'y': [
                        len(data[(data['class'] == 1) & (data['browser_FireFox'] == 1)]),
                        len(data[(data['class'] == 1) & (data['browser_IE'] == 1)]),
                        len(data[(data['class'] == 1) & (data['browser_Opera'] == 1)]),
                        len(data[(data['class'] == 1) & (data['browser_Safari'] == 1)]),
                    ], 'type': 'bar', 'name': 'Fraud by Browser'},
                ],
                'layout': {
                    'title': 'Fraud Cases by Browser',
                    'xaxis': {'title': 'Browser'},
                    'yaxis': {'title': 'Number of Fraud Cases'}
                }
            }
        )),
    ])
])

if __name__ == "__main__":
    app.run(debug=True)
