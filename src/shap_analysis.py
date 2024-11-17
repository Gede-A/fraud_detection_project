import shap
import matplotlib.pyplot as plt
import joblib
import os

def compute_shap_values(model_path, X_test, save_dir="models/shap_values/"):
    os.makedirs(save_dir, exist_ok=True)

    # Load the model
    model = joblib.load(model_path)

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Save SHAP values
    shap_values_path = os.path.join(save_dir, "shap_values.pkl")
    joblib.dump(shap_values, shap_values_path)
    print(f"SHAP values saved at {shap_values_path}")

    return explainer, shap_values

def generate_shap_plots(explainer, shap_values, X_test, save_dir="models/shap_values/"):
    os.makedirs(save_dir, exist_ok=True)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(save_dir, "summary_plot.png"))
    plt.close()

    # Force plot
    instance_index = 0  # Example for first instance
    force_plot = shap.force_plot(
        explainer.expected_value, 
        shap_values[instance_index], 
        X_test.iloc[instance_index]
    )
    shap.save_html(os.path.join(save_dir, "force_plot.html"), force_plot)

    print(f"SHAP plots saved in {save_dir}")
