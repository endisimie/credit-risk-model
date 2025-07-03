from pydantic import BaseModel, Field, conint, confloat
from typing import Optional

from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    # Temporal transaction features
    TransactionHour: int = Field(..., ge=0, le=23, description="Hour of the transaction (0-23)")
    TransactionDay: int = Field(..., ge=1, le=31, description="Day of the transaction (1-31)")
    TransactionMonth: int = Field(..., ge=1, le=12, description="Month of the transaction (1-12)")
    TransactionYear: int = Field(..., description="Year of the transaction")
    
    # Aggregate transaction features
    TotalAmount: float = Field(..., ge=0, description="Total amount spent")
    AverageAmount: float = Field(..., ge=0, description="Average transaction amount")
    TransactionCount: int = Field(..., ge=0, description="Number of transactions")
    AmountStdDev: float = Field(..., ge=0, description="Standard deviation of transaction amounts")
    
    # RFM features
    Recency: int = Field(..., ge=0, description="Days since last transaction")
    Frequency: int = Field(..., ge=0, description="Transaction frequency")
    Monetary: float = Field(..., ge=0, description="Monetary value of transactions")
    
    # Categorical/demographic features (encoded as strings)
    Gender: str = Field(..., description="Customer gender")
    Country: str = Field(..., description="Customer country")
    DeviceType: str = Field(..., description="Type of device used")
    Browser: str = Field(..., description="Browser type")
    OS: str = Field(..., description="Operating system")
    
    # Behavioral features
    IsReturningCustomer: int = Field(..., ge=0, le=1, description="1 if returning customer, else 0")
    IsEmailVerified: int = Field(..., ge=0, le=1, description="1 if email verified, else 0")
    AccountAgeDays: int = Field(..., ge=0, description="Account age in days")
    DaysSinceLastLogin: int = Field(..., ge=0, description="Days since last login")
    TotalLogins: int = Field(..., ge=0, description="Total number of logins")
    TotalCartAdds: int = Field(..., ge=0, description="Total items added to cart")
    TotalPurchases: int = Field(..., ge=0, description="Total purchases made")
    HasPaymentMethod: int = Field(..., ge=0, le=1, description="1 if payment method available, else 0")
    TotalComplaints: int = Field(..., ge=0, description="Total complaints filed")
    SupportTickets: int = Field(..., ge=0, description="Total support tickets opened")


class PredictionResponse(BaseModel):
    customer_id: Optional[str]
    credit_risk_score: float = Field(..., description="Predicted credit risk score (lower is better)")
    risk_class: str = Field(..., description="Risk classification (e.g., low, medium, high)")
    message: Optional[str] = Field(None, description="Additional information or warnings")
