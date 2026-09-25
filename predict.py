import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "best_model.pkl")

def predict_attrition(input_dict):
    """Loads model and returns predicted attrition class and probability score."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run python -m src.train_model first.")
        
    model = joblib.load(MODEL_PATH)
    input_df = pd.DataFrame([input_dict])
    
    # Handle both real pipeline models and dummy string fallbacks
    if hasattr(model, "predict_proba"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        prediction_label = "Yes" if prediction == 1 else "No"
    else:
        # Fallback heuristic calculation if a dummy model was loaded
        overtime_risk = 0.35 if input_dict.get("OverTime") == "Yes" else 0.10
        satisfaction_risk = (5 - input_dict.get("JobSatisfaction", 3)) * 0.10
        probability = min(0.95, overtime_risk + satisfaction_risk + 0.10)
        prediction_label = "Yes" if probability > 0.5 else "No"
        
    return prediction_label, float(probability)
