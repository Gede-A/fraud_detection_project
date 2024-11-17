import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Function to retrain the model
def retrain_model():
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    logreg_model = LogisticRegression(max_iter=1000)
    logreg_model.fit(X_train, y_train)
    
    # Log the new model
    mlflow.sklearn.log_model(logreg_model, 'logistic_regression_model')
    print("Retrained and logged the new model")

# Trigger retraining process
retrain_model()
