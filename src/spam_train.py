import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Pfade
DATA_FILE = os.path.join("data", "spam_preprocessed.csv")
MODEL_DIR = "models"
VECTORIZER_FILE = os.path.join(MODEL_DIR, "spam_vectorizer.pkl")
LOGREG_FILE = os.path.join(MODEL_DIR, "spam_logreg.pkl")


def train_spam_model():
    print(f"Lade Datensatz: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)

    # Bereinigung
    df = df[["category", "clean_text"]].dropna()
    df["category"] = df["category"].astype(str)
    df["clean_text"] = df["clean_text"].astype(str)

    X = df["clean_text"]
    y = df["category"]

    print(f"Datensätze insgesamt: {len(df)}")
    print(df["category"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    # 1) TF-IDF VECTORISER TRAINIEREN
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        lowercase=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 2) Logistic Regression TRAINIEREN
    logreg = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        n_jobs=-1
    )

    logreg.fit(X_train_vec, y_train)
    y_pred = logreg.predict(X_test_vec)

    # 3) Bewertung
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred, digits=3))

    # 4) Modelle speichern
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    joblib.dump(logreg, LOGREG_FILE)

if __name__ == "__main__":
    train_spam_model()
