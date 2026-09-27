"""
House Price Predictor
Uses Gradient Boosting on the California Housing dataset.
"""
import os
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../saved_models/house_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "../../saved_models/house_scaler.pkl")

FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

FEATURE_LABELS = [
    "Median Income (×$10k)", "House Age (years)", "Avg Rooms",
    "Avg Bedrooms", "Population", "Avg Occupancy", "Latitude", "Longitude"
]


def train_and_save():
    """Train the house price regressor and persist it."""
    housing = fetch_california_housing()
    X, y = housing.data, housing.target  # y in units of $100k

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    reg = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    reg.fit(X_train, y_train)

    r2 = r2_score(y_test, reg.predict(X_test))
    print(f"[House] R² score: {r2:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(reg, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return r2


def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict(
    med_inc: float, house_age: float, ave_rooms: float, ave_bedrms: float,
    population: float, ave_occup: float, latitude: float, longitude: float,
):
    model, scaler = load_model()
    features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms,
                           population, ave_occup, latitude, longitude]])
    features_scaled = scaler.transform(features)
    price_100k = float(model.predict(features_scaled)[0])
    price_usd = max(price_100k * 100_000, 50_000)

    # Feature importance
    importances = model.feature_importances_
    importance_dict = {
        FEATURE_LABELS[i]: round(float(imp) * 100, 2)
        for i, imp in enumerate(importances)
    }

    return {
        "predicted_price": round(price_usd, 2),
        "predicted_price_formatted": f"${price_usd:,.0f}",
        "feature_importance": importance_dict,
    }
