import os
import csv
import joblib
from datetime import datetime

VECTORIZER_FILE = r"C:\Users\Student\Desktop\Abschluss Projekt\Projekt\models\spam_vectorizer.pkl"
SPAM_MODEL_FILE = r"C:\Users\Student\Desktop\Abschluss Projekt\Projekt\models\complement_nb.pkl"
SPAM_LOG_FILE = r"C:\Users\Student\Desktop\Abschluss Projekt\Projekt\data/spam_log_daten.csv"

# ---------------------------
# Initialisierung (Modelle einmalig laden)
# ---------------------------
def _load_spam_models():
    if not os.path.exists(VECTORIZER_FILE) or not os.path.exists(SPAM_MODEL_FILE):
        print("FEHLER: Spam-Modelle oder Vectorizer nicht gefunden.")
        return None, None
    
    print("Spam-Modell und Vectorizer werden geladen...")
    vectorizer = joblib.load(VECTORIZER_FILE)
    spam_model = joblib.load(SPAM_MODEL_FILE)
    return vectorizer, spam_model

GLOBAL_SPAM_VECTORIZER, GLOBAL_SPAM_MODEL = _load_spam_models()

# ---------------------------
# Log-Funktion
# ---------------------------
def save_spam_to_log(text, prediction, probability):
    os.makedirs("Projekt/data", exist_ok=True)
    file_exists = os.path.isfile(SPAM_LOG_FILE)

    with open(SPAM_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["timestamp", "prediction", "probability", "email_text"])

        writer.writerow([
            datetime.now().isoformat(),
            prediction,
            f"{probability:.4f}",
            text.replace("\n", " ")[:5000]
        ])


# ---------------------------
# Klassifikation (Öffentliche Funktion)
# ---------------------------
def classify_spam(text):
    if GLOBAL_SPAM_MODEL is None:
        return "error", 0.0 # Fehlerbehandlung

    X_vec = GLOBAL_SPAM_VECTORIZER.transform([text])
    proba = GLOBAL_SPAM_MODEL.predict_proba(X_vec)[0]
    classes = GLOBAL_SPAM_MODEL.classes_

    idx_spam = list(classes).index("spam")
    prob_spam = proba[idx_spam]

    label = "spam" if prob_spam > 0.75 else "ham"

    return label, prob_spam
