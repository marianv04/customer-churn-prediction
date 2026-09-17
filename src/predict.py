import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_xgboost_model.joblib"

def load_model():
    """Load the trained churn prediction model and threshold."""
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    threshold = artifact["threshold"]

    return model, threshold

def predict_churn(customers: pd.DataFrame) -> pd.DataFrame:
    """Predict churn probabilities and classes for new customers."""

    model, threshold = load_model()

    model_input = customers.drop(
        columns=["customerID"],
        errors="ignore"
    )

    probabilities = model.predict_proba(model_input)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = customers.copy()
    results["churn_probability"] = probabilities
    results["predicted_churn"] = predictions

    return results

