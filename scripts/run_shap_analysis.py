import pandas as pd
from src.shap_analysis import compute_shap_values, generate_shap_plots

# Paths
model_path = "models/trained_models/fraud_model.pkl"
data_path = "data/processed/X_test.csv"
save_dir = "models/shap_values/"

# Load data
X_test = pd.read_csv(data_path)

# Compute and save SHAP values
explainer, shap_values = compute_shap_values(model_path, X_test, save_dir)

# Generate and save SHAP plots
generate_shap_plots(explainer, shap_values, X_test, save_dir)
