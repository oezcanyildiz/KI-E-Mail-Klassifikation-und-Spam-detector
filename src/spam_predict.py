import os
import csv
import joblib
from datetime import datetime

VECTORIZER_FILE = r"C:\Users\Student\Desktop\Abschluss Projekt\Projekt\models\spam_vectorizer.pkl"
SPAM_MODEL_FILE = r"C:\Users\Student\Desktop\Abschluss Projekt\Projekt\models\complement_nb.pkl"   
SPAM_LOG_FILE = r"C:\Users\Student\Desktop\Abschluss Projekt\Projekt\data/spam_log_daten.csv"


# ---------------------------
# Modelle laden
# ---------------------------
def load_models():
    vectorizer = joblib.load(VECTORIZER_FILE)
    spam_model = joblib.load(SPAM_MODEL_FILE)
    return vectorizer, spam_model


# ---------------------------
# Spam speichern
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
# Klassifikation
# ---------------------------
def classify(text):
    vectorizer, spam_model = load_models()

    X_vec = vectorizer.transform([text])
    proba = spam_model.predict_proba(X_vec)[0]
    classes = spam_model.classes_

    idx_spam = list(classes).index("spam")
    prob_spam = proba[idx_spam]

    label = "spam" if prob_spam > 0.5 else "ham"

    return label, prob_spam


# ---------------------------
# Main
# ---------------------------
def main():
    text = input("Bitte Email-Text eingeben:\n")

    label, prob_spam = classify(text)

    print("\n📊 Ergebnis:")
    print(f"→ Hauptklasse: {label}")

    if label == "spam":
        print("ℹ️  Nachricht wurde als Spam klassifiziert.")
        save_spam_to_log(text, label, prob_spam)
        print(f"💾 Spam gespeichert in: {SPAM_LOG_FILE}")
    else:
        print("✔️ Nachricht ist kein Spam. Wird zur nächsten Klassifizierung weitergeleitet.")


if __name__ == "__main__":
    main()
