import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# Load the datasets
fraud_data = pd.read_csv("data/processed/Fraud__Data.csv")
creditcard_data = pd.read_csv("data/processed/creditcard.csv")

# Check for missing values in each dataset before combining
print("Missing values in Fraud Data:")
print(fraud_data.isnull().sum())

print("\nMissing values in Credit Card Data:")
print(creditcard_data.isnull().sum())

# Fill missing values in the fraud data (if necessary)
fraud_data.fillna(fraud_data.mean(), inplace=True)
categorical_columns_fraud = fraud_data.select_dtypes(include=['object']).columns
for col in categorical_columns_fraud:
    fraud_data[col].fillna(fraud_data[col].mode()[0], inplace=True)

# Fill missing values in the credit card data (if necessary)
creditcard_data.fillna(creditcard_data.mean(), inplace=True)
categorical_columns_credit = creditcard_data.select_dtypes(include=['object']).columns
for col in categorical_columns_credit:
    creditcard_data[col].fillna(creditcard_data[col].mode()[0], inplace=True)

# Check again for missing values after filling
print("\nMissing values in Fraud Data after filling:")
print(fraud_data.isnull().sum())

print("\nMissing values in Credit Card Data after filling:")
print(creditcard_data.isnull().sum())

# Separate features and target for both datasets
X_fraud = fraud_data.drop(columns=['class'])
y_fraud = fraud_data['class']

X_credit = creditcard_data.drop(columns=['Class'])
y_credit = creditcard_data['Class']

# Combine both datasets (ensure the features match)
X_combined = pd.concat([X_fraud, X_credit], axis=0)
y_combined = pd.concat([y_fraud, y_credit], axis=0)

# Check for missing values after combining
print("\nMissing values in Combined Data:")
print(X_combined.isnull().sum())

# Fill any missing values that might appear after combining datasets
X_combined.fillna(X_combined.mean(), inplace=True)
categorical_columns_combined = X_combined.select_dtypes(include=['object']).columns
for col in categorical_columns_combined:
    X_combined[col].fillna(X_combined[col].mode()[0], inplace=True)

# Check again for missing values after filling
print("\nMissing values in Combined Data after filling:")
print(X_combined.isnull().sum())

# Train-Test Split for combined data
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42
)

# Load the trained model
model_path = "models/trained_models/logistic_regression_model.joblib"
model = joblib.load(model_path)

# If the model is a pipeline, extract the classifier step
if hasattr(model, 'named_steps'):
    model_step = model.named_steps['logisticregression']  # This depends on the name of your step in the pipeline
else:
    model_step = model

# Use SHAP's KernelExplainer with the model
explainer = shap.KernelExplainer(model_step.predict_proba, X_train)

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Save SHAP values
shap_values_dir = "models/shap_values/"
os.makedirs(shap_values_dir, exist_ok=True)
shap_values_file = os.path.join(shap_values_dir, "shap_values.joblib")
joblib.dump(shap_values, shap_values_file)

# Generate SHAP plots
# Summary Plot
summary_plot_path = os.path.join(shap_values_dir, "shap_summary_plot.png")
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(summary_plot_path)

print(f"SHAP values and plots saved in {shap_values_dir}")
