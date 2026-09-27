"""
Sentiment Analyzer
Naive Bayes + TF-IDF on synthetic movie review data.
"""
import os
import joblib
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../saved_models/sentiment_model.pkl")

POSITIVE_REVIEWS = [
    "This movie was absolutely fantastic and I loved every minute of it",
    "Amazing film with outstanding performances and a great story",
    "Brilliant cinematography and wonderful acting throughout",
    "I thoroughly enjoyed this movie from start to finish",
    "Excellent direction and a gripping story that kept me hooked",
    "One of the best films I have ever seen in my life",
    "Superb performances by the entire cast, truly memorable",
    "A masterpiece of modern cinema with breathtaking visuals",
    "This was a delightful experience and I highly recommend it",
    "Incredibly well-made film with beautiful storytelling",
    "The performances were top-notch and the plot was engaging",
    "Heartwarming and uplifting story that made me smile",
    "Exceptional film with powerful emotional moments",
    "A wonderful cinematic journey full of surprises",
    "Great movie with memorable characters and excellent dialogue",
    "I was blown away by the quality of this production",
    "Stunning visuals combined with an incredible soundtrack",
    "This film exceeded all my expectations in every way",
    "Perfectly crafted story with fantastic character development",
    "A joy to watch from beginning to end, absolutely loved it",
    "The product works great and arrived on time",
    "Excellent quality and very satisfied with my purchase",
    "Great value for money highly recommend this product",
    "Fast shipping and the item is exactly as described",
    "Outstanding customer service and a perfect product",
    "Very happy with this purchase exceeded expectations",
    "Works perfectly and the build quality is impressive",
    "Amazing product that does everything as advertised",
    "Love this product it has made my life so much easier",
    "Top quality item would definitely buy again",
]

NEGATIVE_REVIEWS = [
    "This movie was terrible and a complete waste of time",
    "Awful film with poor acting and a nonsensical plot",
    "I hated every single minute of this boring disaster",
    "Terrible direction and the story made absolutely no sense",
    "One of the worst films I have ever had to sit through",
    "The acting was atrocious and the script was painful to watch",
    "A dreadful film that fails on every single level",
    "Completely boring and disappointing from start to finish",
    "I cannot believe how bad this movie actually was",
    "Waste of money and two hours of my life I will never get back",
    "Poorly written characters with zero depth or development",
    "The plot was confusing and the ending was very disappointing",
    "Terrible special effects and the pacing was unbearably slow",
    "I fell asleep multiple times during this dreadful film",
    "No redeeming qualities whatsoever, avoid this movie at all costs",
    "Badly acted and poorly directed, an absolute train wreck",
    "The worst screenplay I have ever encountered in cinema",
    "Painfully slow with no interesting characters whatsoever",
    "A total mess from beginning to end, very unsatisfying",
    "Extremely disappointing and not worth watching at all",
    "The product broke after two days of use, terrible quality",
    "Very disappointed with this purchase, not as described",
    "Poor quality and extremely slow shipping experience",
    "Stopped working within a week and customer service was unhelpful",
    "Awful product that does not work as advertised at all",
    "Total waste of money would not recommend to anyone",
    "The item arrived damaged and looked nothing like the photos",
    "Very poor quality control and the product malfunctioned",
    "Terrible experience from start to finish with this product",
    "Do not buy this product it is a complete disappointment",
]

NEUTRAL_REVIEWS = [
    "The movie was okay, not great but not terrible either",
    "It was an average film with some good and bad moments",
    "Decent enough film but nothing particularly memorable about it",
    "The film had its moments but overall felt quite mediocre",
    "Neither good nor bad, just a very average viewing experience",
    "The product works as expected nothing special about it",
    "Average quality for the price, does what it is supposed to do",
    "It is okay but there are probably better options available",
    "Functional product but nothing exceptional about the quality",
    "Meets basic expectations but does not exceed them in any way",
]


def _build_dataset():
    texts = POSITIVE_REVIEWS + NEGATIVE_REVIEWS + NEUTRAL_REVIEWS
    labels = (
        ["positive"] * len(POSITIVE_REVIEWS)
        + ["negative"] * len(NEGATIVE_REVIEWS)
        + ["neutral"] * len(NEUTRAL_REVIEWS)
    )
    return texts, labels


def train_and_save():
    texts, labels = _build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf", MultinomialNB(alpha=0.5)),
    ])
    pipeline.fit(X_train, y_train)

    acc = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"[Sentiment] Training accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return acc


def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    return joblib.load(MODEL_PATH)


SENTIMENT_EMOJI = {"positive": "😊", "negative": "😞", "neutral": "😐"}
SENTIMENT_COLOR = {"positive": "#22c55e", "negative": "#ef4444", "neutral": "#f59e0b"}


def predict(text: str):
    if not text.strip():
        raise ValueError("Text cannot be empty")

    model = load_model()
    classes = model.classes_
    proba = model.predict_proba([text])[0]
    predicted = model.predict([text])[0]

    proba_dict = {cls: round(float(p) * 100, 2) for cls, p in zip(classes, proba)}
    confidence = round(float(max(proba)) * 100, 2)

    word_count = len(text.split())
    char_count = len(text)

    return {
        "sentiment": predicted,
        "confidence": confidence,
        "emoji": SENTIMENT_EMOJI[predicted],
        "color": SENTIMENT_COLOR[predicted],
        "probabilities": proba_dict,
        "word_count": word_count,
        "char_count": char_count,
    }
