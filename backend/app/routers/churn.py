"""Customer churn prediction router"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.models import churn as churn_model

router = APIRouter(prefix="/api/churn", tags=["Customer Churn"])


class ChurnInput(BaseModel):
    tenure: int = Field(..., ge=0, le=72, example=12, description="Months as customer")
    monthly_charges: float = Field(..., ge=0.0, le=200.0, example=65.5, description="Monthly charge ($)")
    total_charges: float = Field(..., ge=0.0, le=10000.0, example=800.0, description="Total charges ($)")
    num_products: int = Field(..., ge=1, le=4, example=2, description="Number of products subscribed")
    has_internet: int = Field(..., ge=0, le=1, example=1, description="Has internet service (0/1)")
    contract_type: int = Field(..., ge=0, le=2, example=0, description="0=Monthly, 1=One Year, 2=Two Year")
    tech_support: int = Field(..., ge=0, le=1, example=0, description="Has tech support (0/1)")
    payment_method: int = Field(..., ge=0, le=3, example=0, description="0=eCheck, 1=Mail, 2=Bank, 3=CCard")


@router.post("/predict")
async def predict_churn(data: ChurnInput):
    try:
        result = churn_model.predict(
            data.tenure, data.monthly_charges, data.total_charges,
            data.num_products, data.has_internet, data.contract_type,
            data.tech_support, data.payment_method
        )
        return {"success": True, "model": "Customer Churn Predictor", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def churn_info():
    return {
        "model": "Random Forest Classifier",
        "dataset": "Synthetic Telecom Churn Data (3000 samples)",
        "target": "Customer churn probability",
        "algorithm": "Random Forest (150 trees)",
    }
