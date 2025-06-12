import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

def explain_model(X_test, model_path="models/fraud_model.pkl", output_path="shap_outputs/shap_summary.png"):
    """
    Generate and save a SHAP summary plot for the fraud detection model.

    Parameters:
    - X_test: pd.DataFrame - the test features to explain
    - model_path: str - path to the trained model
    - output_path: str - path to save the SHAP summary plot
    """

    # Ensure SHAP output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Model file not found. Please train the model first.")

    # Load trained model
    model = joblib.load(model_path)

    # 🔐 Ensure X_test is fully numeric (critical for SHAP)
    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.fillna(0)

    # ✅ Use SHAP TreeExplainer directly for XGBoost
    print("⚙️ Generating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Plot and save summary
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"✅ SHAP summary plot saved to {output_path}")
