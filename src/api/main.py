from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load trained model
model = joblib.load("models/best_model.joblib")

# Expected feature order from training
expected_feature_order = [
    "TransactionHour", "TransactionDay", "TransactionMonth", "TransactionYear",
    "TotalAmount", "AverageAmount", "TransactionCount", "AmountStdDev",
    "Recency", "Frequency", "Monetary",
    "Gender", "Country", "DeviceType", "Browser", "OS",
    "IsReturningCustomer", "IsEmailVerified", "AccountAgeDays", "DaysSinceLastLogin",
    "TotalLogins", "TotalCartAdds", "TotalPurchases", "HasPaymentMethod",
    "TotalComplaints", "SupportTickets"
]

# Dummy encoders (you must use actual mappings used during training)
gender_map = {"Male": 0, "Female": 1}
country_map = {"Ethiopia": 0, "Kenya": 1}
device_map = {"Mobile": 0, "Desktop": 1}
browser_map = {"Chrome": 0, "Firefox": 1}
os_map = {"Android": 0, "Windows": 1}

class CustomerFeatures(BaseModel):
    TransactionHour: int
    TransactionDay: int
    TransactionMonth: int
    TransactionYear: int
    TotalAmount: float
    AverageAmount: float
    TransactionCount: int
    AmountStdDev: float
    Recency: int
    Frequency: int
    Monetary: float
    Gender: str
    Country: str
    DeviceType: str
    Browser: str
    OS: str
    IsReturningCustomer: int
    IsEmailVerified: int
    AccountAgeDays: int
    DaysSinceLastLogin: int
    TotalLogins: int
    TotalCartAdds: int
    TotalPurchases: int
    HasPaymentMethod: int
    TotalComplaints: int
    SupportTickets: int

@app.post("/predict")
async def predict_risk(features: CustomerFeatures):
    try:
        data = features.dict()

        # Encode string categorical features
        data["Gender"] = gender_map.get(data["Gender"])
        data["Country"] = country_map.get(data["Country"])
        data["DeviceType"] = device_map.get(data["DeviceType"])
        data["Browser"] = browser_map.get(data["Browser"])
        data["OS"] = os_map.get(data["OS"])

        # Check for unmapped values
        if None in [data["Gender"], data["Country"], data["DeviceType"], data["Browser"], data["OS"]]:
            raise HTTPException(status_code=400, detail="Invalid categorical value in input.")

        # Create ordered input
        input_df = pd.DataFrame([data])
        input_array = input_df[expected_feature_order].values

        # Predict
        risk_proba = model.predict_proba(input_array)[0][1]
        return {"risk_probability": round(risk_proba, 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
