"""Iris prediction router"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.models import iris as iris_model

router = APIRouter(prefix="/api/iris", tags=["Iris Classifier"])


class IrisInput(BaseModel):
    sepal_length: float = Field(..., ge=0.1, le=20.0, example=5.1, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0.1, le=20.0, example=3.5, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0.1, le=20.0, example=1.4, description="Petal length in cm")
    petal_width: float = Field(..., ge=0.1, le=20.0, example=0.2, description="Petal width in cm")


@router.post("/predict")
async def predict_iris(data: IrisInput):
    try:
        result = iris_model.predict(
            data.sepal_length, data.sepal_width,
            data.petal_length, data.petal_width
        )
        return {"success": True, "model": "Iris Classifier", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def iris_info():
    return {
        "model": "Random Forest Classifier",
        "dataset": "Scikit-learn Iris Dataset",
        "classes": ["Setosa", "Versicolor", "Virginica"],
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "algorithm": "Random Forest (100 trees)",
    }
