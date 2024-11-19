# Fraud Detection API

This project serves a fraud detection machine learning model using Flask and Docker.

## Requirements

- Python 3.8+
- Docker (optional, for containerization)

## Setup

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. To run the Flask API locally:
    ```bash
    python serve_model.py
    ```

3. To build and run the Docker container:
    ```bash
    docker build -t fraud-detection-model .
    docker run -p 5000:5000 fraud-detection-model
    ```

## API Endpoints

- **POST /predict**: Make a fraud prediction
  - Request body:
    ```json
    {
      "features": [feature1, feature2, feature3, ...]
    }
    ```
  - Response:
    ```json
    {
      "prediction": 0 or 1,
      "probability": 0.85
    }
    ```

## Logging

The API logs requests and predictions. Logs can be found in the `logs/` directory.

