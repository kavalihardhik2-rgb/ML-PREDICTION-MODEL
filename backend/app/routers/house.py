"""House price prediction router"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.models import house as house_model

router = APIRouter(prefix="/api/house", tags=["House Price"])


class HouseInput(BaseModel):
    med_inc: float = Field(..., ge=0.0, le=20.0, example=3.5, description="Median income (x$10k)")
    house_age: float = Field(..., ge=0.0, le=100.0, example=20.0, description="House age in years")
    ave_rooms: float = Field(..., ge=1.0, le=50.0, example=5.2, description="Average number of rooms")
    ave_bedrms: float = Field(..., ge=1.0, le=20.0, example=1.0, description="Average number of bedrooms")
    population: float = Field(..., ge=1.0, le=50000.0, example=1200.0, description="Block population")
    ave_occup: float = Field(..., ge=1.0, le=20.0, example=3.0, description="Average occupancy per household")
    latitude: float = Field(..., ge=32.0, le=42.0, example=37.5, description="Block latitude")
    longitude: float = Field(..., ge=-125.0, le=-114.0, example=-122.0, description="Block longitude")


@router.post("/predict")
async def predict_house(data: HouseInput):
    try:
        result = house_model.predict(
            data.med_inc, data.house_age, data.ave_rooms, data.ave_bedrms,
            data.population, data.ave_occup, data.latitude, data.longitude
        )
        return {"success": True, "model": "House Price Predictor", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def house_info():
    return {
        "model": "Gradient Boosting Regressor",
        "dataset": "California Housing Dataset",
        "target": "Median house value (USD)",
        "algorithm": "Gradient Boosting (200 estimators, depth=5)",
    }
