import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Load or create dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Start MLflow experiment
mlflow.set_experiment('Fraud_Detection_Experiment')

# Start a new run
with mlflow.start_run():
    # Train a Logistic Regression model
    logreg_model = LogisticRegression(max_iter=1000)
    logreg_model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = logreg_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log hyperparameters
    mlflow.log_param('max_iter', 1000)
    
    # Log metrics
    mlflow.log_metric('accuracy', accuracy)
    
    # Log the trained model
    mlflow.sklearn.log_model(logreg_model, 'logistic_regression_model')

    print(f"Logged model with accuracy: {accuracy}")
