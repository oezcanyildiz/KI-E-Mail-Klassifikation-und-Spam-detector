import os
from src.spam_train import train_spam_model

MODEL_DIR = "models"
LOGREG_FILE = os.path.join(MODEL_DIR, "spam_logreg.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "spam_vectorizer.pkl")


def test_train_spam_model_creates_files():
    train_spam_model()
    assert os.path.exists(LOGREG_FILE), "Spam Modell wurde nicht erzeugt"
    assert os.path.exists(VECTORIZER_FILE), "Spam Vectorizer wurde nicht erzeugt"


def test_train_spam_model_callable():
    assert callable(train_spam_model)
