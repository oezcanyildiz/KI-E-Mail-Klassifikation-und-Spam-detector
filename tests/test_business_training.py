import os
from src.train import train_pipeline


MODEL_FILE = "models/email_classifier_pipeline.pkl"


def test_train_business_model_creates_file():
    """
    Testet ob das Training des Business-Modells erfolgreich durchläuft
    und die Pipeline als .pkl Datei gespeichert wird.
    """
    # Training starten
    train_pipeline()

    # Prüfen: Datei muss existieren
    assert os.path.exists(MODEL_FILE), "Business-Modell wurde nicht gespeichert"


def test_train_business_model_callable():
    """
    Stellt sicher, dass die Trainingsfunktion aufrufbar ist.
    """
    assert callable(train_pipeline)
