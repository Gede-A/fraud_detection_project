import mlflow
import time
import numpy as np

# Function to simulate performance monitoring
def monitor_performance(model, X_test, y_test):
    while True:
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"Current model accuracy: {accuracy}")
        
        # Log metrics to MLflow
        mlflow.log_metric('accuracy', accuracy)
        
        time.sleep(3600)  # Monitor every hour

# Assume model and data are loaded
# monitor_performance(logreg_model, X_test, y_test)
