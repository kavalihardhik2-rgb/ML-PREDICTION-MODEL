"""Sentiment analysis router"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.models import sentiment as sentiment_model

router = APIRouter(prefix="/api/sentiment", tags=["Sentiment Analysis"])


class SentimentInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, example="This movie was absolutely fantastic!")


@router.post("/predict")
async def predict_sentiment(data: SentimentInput):
    try:
        result = sentiment_model.predict(data.text)
        return {"success": True, "model": "Sentiment Analyzer", "text": data.text, **result}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def sentiment_info():
    return {
        "model": "Naive Bayes + TF-IDF Pipeline",
        "dataset": "Curated review corpus (positive/negative/neutral)",
        "classes": ["positive", "negative", "neutral"],
        "algorithm": "MultinomialNB with TF-IDF (bigrams, 5000 features)",
    }
