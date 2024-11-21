import dash
from dash import dcc, html
import requests
import pandas as pd
import plotly.express as px

# Initialize Dash app
app = dash.Dash(__name__)

# Fetch data from Flask backend
summary_response = requests.get("http://127.0.0.1:5000/api/summary").json()
trends_response = requests.get("http://127.0.0.1:5000/api/fraud_trends").json()

# Convert trends data to DataFrame
trends_df = pd.DataFrame(list(trends_response.items()), columns=["Date", "Fraud Cases"])

# Layout of the Dashboard
app.layout = html.Div(children=[
    html.H1("Fraud Detection Dashboard"),
    
    # Summary boxes
    html.Div([
        html.Div(f"Total Transactions: {summary_response['total_transactions']}"),
        html.Div(f"Fraud Cases: {summary_response['fraud_cases']}"),
        html.Div(f"Fraud Percentage: {summary_response['fraud_percentage']}%")
    ], style={'display': 'flex', 'justify-content': 'space-around'}),
    
    # Fraud trends line chart
    dcc.Graph(
        id='fraud-trends',
        figure=px.line(trends_df, x='Date', y='Fraud Cases', title='Fraud Cases Over Time')
    ),
])

# Run Dash
if __name__ == '__main__':
    app.run_server(debug=True)
