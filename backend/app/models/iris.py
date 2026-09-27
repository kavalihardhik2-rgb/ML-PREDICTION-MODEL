"""
Iris Flower Classifier
Uses scikit-learn Random Forest on the classic Iris dataset.
"""
import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../saved_models/iris_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "../../saved_models/iris_scaler.pkl")

CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]


def train_and_save():
    """Train the Iris classifier and persist it."""
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[Iris] Training accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return acc


def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    model, scaler = load_model()
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)
    proba = model.predict_proba(features_scaled)[0]
    predicted_idx = int(np.argmax(proba))
    return {
        "species": CLASS_NAMES[predicted_idx],
        "confidence": round(float(proba[predicted_idx]) * 100, 2),
        "probabilities": {
            CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(proba)
        },
    }
