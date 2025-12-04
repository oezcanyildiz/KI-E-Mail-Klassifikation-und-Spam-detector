import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Pfade
DATA_FILE = r"C:\Users\Student\Desktop\Abschluss Projekt\Projekt\data\spam_preprocessed.csv"

# Bereits existierender Vectorizer aus Logistic Regression Training
VECTORIZER_FILE = r"C:\Users\Student\Desktop\Abschluss Projekt\Projekt\models\spam_vectorizer.pkl"

# Speicherorte für NB-Modelle
MODEL_DIR = r"C:\Users\Student\Desktop\Abschluss Projekt\Projekt\models"
MULTI_FILE = os.path.join(MODEL_DIR, "multinomial_nb.pkl")
BERN_FILE = os.path.join(MODEL_DIR, "bernoulli_nb.pkl")
COMP_FILE = os.path.join(MODEL_DIR, "complement_nb.pkl")


def train_spam_nb_models():
    print(f"Lade Datensatz: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)

    df = df[["category", "clean_text"]].dropna()
    df["category"] = df["category"].astype(str)
    df["clean_text"] = df["clean_text"].astype(str)

    X = df["clean_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ------------------------------------
    # 1) Vectorizer LADEN statt neu trainieren
    # ------------------------------------
    print(f"Lade bestehenden TF-IDF Vectorizer: {VECTORIZER_FILE}")
    vectorizer = joblib.load(VECTORIZER_FILE)

    # Vectorizer NICHT neu fitten → nur transformieren
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # ------------------------------------
    # 2) Alle drei Naive-Bayes-Modelle trainieren
    # ------------------------------------
    models = {
        "MultinomialNB": (MultinomialNB(), MULTI_FILE),
        "BernoulliNB": (BernoulliNB(), BERN_FILE),
        "ComplementNB": (ComplementNB(), COMP_FILE),
    }

    for name, (model, path) in models.items():
        print(f"\n🔧 Trainiere Modell: {name}")
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred, digits=3))

        print(f"💾 Speichere Modell → {path}")
        joblib.dump(model, path)

    print("\nAlle Naive-Bayes-Modelle erfolgreich trainiert und gespeichert.")


if __name__ == "__main__":
    train_spam_nb_models()
