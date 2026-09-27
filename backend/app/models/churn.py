"""
Customer Churn Predictor
Random Forest classifier trained on synthetic telecom churn data.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../saved_models/churn_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "../../saved_models/churn_scaler.pkl")


def _generate_synthetic_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    monthly_charges = rng.uniform(20, 120, n)
    total_charges = tenure * monthly_charges + rng.normal(0, 50, n)
    num_products = rng.integers(1, 5, n)
    has_internet = rng.integers(0, 2, n)
    contract_type = rng.integers(0, 3, n)   # 0=monthly, 1=1yr, 2=2yr
    tech_support = rng.integers(0, 2, n)
    payment_method = rng.integers(0, 4, n)

    # Churn logic
    churn_score = (
        -0.05 * tenure
        + 0.02 * monthly_charges
        + 0.5 * (contract_type == 0)
        - 0.3 * tech_support
        - 0.4 * num_products
        + rng.normal(0, 0.5, n)
    )
    churn = (churn_score > 0).astype(int)

    X = np.column_stack([
        tenure, monthly_charges, total_charges, num_products,
        has_internet, contract_type, tech_support, payment_method
    ])
    return X, churn


def train_and_save():
    X, y = _generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[Churn] Training accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return acc


def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


FEATURE_LABELS = [
    "Tenure (months)", "Monthly Charges ($)", "Total Charges ($)",
    "Num Products", "Has Internet", "Contract Type", "Tech Support", "Payment Method"
]

CONTRACT_MAP = {"Monthly": 0, "One Year": 1, "Two Year": 2}
PAYMENT_MAP = {"Electronic Check": 0, "Mailed Check": 1, "Bank Transfer": 2, "Credit Card": 3}


def predict(
    tenure: int,
    monthly_charges: float,
    total_charges: float,
    num_products: int,
    has_internet: int,
    contract_type: int,
    tech_support: int,
    payment_method: int,
):
    model, scaler = load_model()
    features = np.array([[tenure, monthly_charges, total_charges, num_products,
                           has_internet, contract_type, tech_support, payment_method]])
    features_scaled = scaler.transform(features)
    proba = model.predict_proba(features_scaled)[0]
    churn_prob = round(float(proba[1]) * 100, 2)

    risk_level = "Low" if churn_prob < 30 else ("Medium" if churn_prob < 65 else "High")
    importances = model.feature_importances_
    importance_dict = {
        FEATURE_LABELS[i]: round(float(imp) * 100, 2)
        for i, imp in enumerate(importances)
    }

    return {
        "churn_probability": churn_prob,
        "stay_probability": round(float(proba[0]) * 100, 2),
        "risk_level": risk_level,
        "prediction": "Will Churn" if churn_prob >= 50 else "Will Stay",
        "feature_importance": importance_dict,
    }
