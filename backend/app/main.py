"""
ML Prediction Platform — FastAPI Backend
Trains and serves 4 ML models via REST API.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import threading

from app.routers import iris, house, churn, sentiment
from app.models import iris as iris_model
from app.models import house as house_model
from app.models import churn as churn_model
from app.models import sentiment as sentiment_model


def train_all_models():
    """Train all models if not already saved (runs in background thread)."""
    print("\n[*] ML Prediction Platform -- Training Models...")
    print("=" * 50)

    print("[1] Training Iris Classifier...")
    acc = iris_model.train_and_save()
    print(f"    OK  Iris model ready  (accuracy: {acc:.2%})")

    print("[2] Training House Price Predictor...")
    r2 = house_model.train_and_save()
    print(f"    OK  House model ready (R2: {r2:.4f})")

    print("[3] Training Customer Churn Predictor...")
    acc2 = churn_model.train_and_save()
    print(f"    OK  Churn model ready (accuracy: {acc2:.2%})")

    print("[4] Training Sentiment Analyzer...")
    acc3 = sentiment_model.train_and_save()
    print(f"    OK  Sentiment model ready (accuracy: {acc3:.2%})")

    print("=" * 50)
    print("All models trained and ready!\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Train models on startup (in background so server starts fast)
    thread = threading.Thread(target=train_all_models, daemon=True)
    thread.start()
    thread.join()  # Wait for all models before accepting requests
    yield


app = FastAPI(
    title="ML Prediction Platform",
    description="A multi-model ML prediction API with Iris, House Price, Churn & Sentiment models.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins for local dev (tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(iris.router)
app.include_router(house.router)
app.include_router(churn.router)
app.include_router(sentiment.router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "online",
        "app": "ML Prediction Platform",
        "version": "1.0.0",
        "models": ["iris", "house", "churn", "sentiment"],
        "docs": "/docs",
    }


@app.get("/api/models/status", tags=["Health"])
async def models_status():
    import os
    saved = "c:/Users/kaval/OneDrive/Desktop/Personal Projects/ML project/backend/saved_models"
    statuses = {}
    for name in ["iris", "house", "churn", "sentiment"]:
        path = os.path.join(saved, f"{name}_model.pkl")
        statuses[name] = "loaded" if os.path.exists(path) else "not_found"
    return {"models": statuses}
