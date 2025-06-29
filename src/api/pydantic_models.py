from pydantic import BaseModel, Field, conint, confloat
from typing import Optional

class CustomerDataRequest(BaseModel):
    customer_id: Optional[str] = Field(None, description="Unique customer identifier")
    age: conint(ge=18, le=100) = Field(..., description="Customer age in years")
    income: confloat(ge=0) = Field(..., description="Annual income")
    loan_amount: confloat(ge=0) = Field(..., description="Requested loan amount")
    loan_duration_months: conint(ge=1, le=360) = Field(..., description="Loan duration in months")
    credit_history_length_years: Optional[conint(ge=0)] = Field(0, description="Length of credit history in years")
    num_credit_cards: Optional[conint(ge=0)] = Field(0, description="Number of active credit cards")
    num_past_defaults: Optional[conint(ge=0)] = Field(0, description="Number of past loan defaults")
    # Add more features as needed matching your model's expected inputs

class PredictionResponse(BaseModel):
    customer_id: Optional[str]
    credit_risk_score: float = Field(..., description="Predicted credit risk score (lower is better)")
    risk_class: str = Field(..., description="Risk classification (e.g., low, medium, high)")
    message: Optional[str] = Field(None, description="Additional information or warnings")
