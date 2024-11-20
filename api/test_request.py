import requests
import numpy as np


# Your features array (NumPy array)
features = np.array([0.5, 1.2, 3.4, 2.1, 0.8, 1.0, 4.5, 3.2, 5.1, 0.7, 
                     2.3, 1.1, 3.3, 4.0, 2.5, 0.9, 1.7, 3.6, 4.8, 5.4, 
                     2.6, 0.4, 1.3, 3.9, 5.0, 4.6, 2.7, 1.6, 0.6, 3.8, 
                     4.7, 2.9, 1.5, 3.1, 5.2, 0.3, 2.8, 4.2, 3.0, 1.9, 
                     5.5, 2.2, 1.4, 3.7, 4.4, 2.0, 1.8, 4.3, 3.5, 5.3, 
                     0.2, 4.1, 2.4, 5.6, 3.0, 2.1, 0.6])

# Ensure all features are converted to native Python types (float)
features_list = features.tolist()  # Convert NumPy array to list
features_list = [float(x) for x in features_list]  # Ensure that each element is a float

# Now the features_list is JSON serializable
print(features_list)  # Just to check the result


# Define the URL of the Flask API
url = 'http://localhost:5000/predict'

# Create the payload with the correct number of features
payload = {'features': features_list}

# Make the POST request to the Flask API
response = requests.post(url, json=payload)

# Check the response from the server
if response.status_code == 200:
    print(f"Prediction: {response.json()}")
else:
    print(f"Error: {response.status_code}")
    print(f"Message: {response.json()}")
