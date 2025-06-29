from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures, RiskPrediction
import joblib
import numpy as np
import mlflow.sklearn

app = FastAPI(title="Credit Risk Scoring API")

# Load the best model
model = joblib.load("models/best_model.joblib")

@app.get("/")
def root():
    return {"message": "Credit Risk Scoring API is running."}

@app.post("/predict", response_model=RiskPrediction)
def predict_risk(data: CustomerFeatures):
    input_data = np.array([list(data.dict().values())])
    risk_proba = model.predict_proba(input_data)[0][1]
    return {"risk_probability": risk_proba}
