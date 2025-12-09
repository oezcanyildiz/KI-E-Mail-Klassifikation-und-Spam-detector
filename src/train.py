import pandas as pd
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Import der Bereinigungsfunktion
try:
    from spam_preprocces import preprocess_text
except ImportError:
    import re
    def preprocess_text(text):
        if not isinstance(text, str): return ""
        return re.sub(r'[^a-z\s]', '', text.lower()).strip()

def load_data(path):
    """
    Lädt die fertig vorbereitete Datei mit den 3 Klassen.
    """
    print(f"--- Lade Trainingsdaten: {path} ---")
    
    if not os.path.exists(path):
        print(f"[FEHLER] Datei nicht gefunden: {path}")
        print("Bitte zuerst 'src/prepare_3class_data.py' ausführen!")
        sys.exit(1)
        
    # Wir erwarten Semikolon als Trenner, wie im Prepare-Skript definiert
    df = pd.read_csv(path, sep=';', encoding='utf-8')
    
    # Kurze Prüfung
    expected_cols = ['subject', 'body', 'label']
    if not all(col in df.columns for col in expected_cols):
        print(f"[FEHLER] Spalten fehlen. Erwartet: {expected_cols}, Gefunden: {df.columns}")
        sys.exit(1)
        
    print(f"[OK] {len(df)} Zeilen geladen.")
    return df

def train_pipeline():
    # 1. Pfad zur EINEN Datei
    data_path = "data/training_data_3classes.csv"
    
    # 2. Daten laden
    df = load_data(data_path)
    
    # 3. Vorbereitung (Feature Engineering)
    df['subject'] = df['subject'].fillna("")
    df['body'] = df['body'].fillna("")
    
    # Text kombinieren
    df['text_combined'] = df['subject'] + " " + df['body']
    
    # Preprocessing
    print("Führe Text-Bereinigung durch...")
    df['clean_text'] = df['text_combined'].apply(preprocess_text)
    
    # Leere Texte entfernen
    df = df[df['clean_text'].str.len() > 1]
    
    X = df['clean_text']
    y = df['label']
    
    print(f"Finale Datengrundlage: {len(df)} E-Mails")
    print("Verteilung der Klassen:")
    print(y.value_counts())

    # 4. Split (80% Training, 20% Test)
    # stratify ist wichtig, damit Support/Vertrieb/Buchhaltung gleichmäßig verteilt sind
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Modelle definieren
    models_to_test = [
        {
            "name": "Logistic Regression",
            "clf": LogisticRegression(class_weight='balanced', max_iter=1000)
        },
        {
            "name": "Random Forest",
            "clf": RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
        },
        {
            "name": "Support Vector Machine (SVM)",
            "clf": SVC(class_weight='balanced', probability=True, kernel='linear')
        },
        {
            "name": "Naive Bayes (Multinomial)",
            "clf": MultinomialNB()
        }
    ]

    best_score = 0
    best_pipeline = None
    best_name = ""

    print("\n--- Starte Modellvergleich (3 Klassen) ---")
    
    for model_info in models_to_test:
        name = model_info["name"]
        clf = model_info["clf"]
        
        print(f"\nTeste: {name}...")
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        score = accuracy_score(y_test, predictions)
        
        print(f"--> Genauigkeit: {score:.2%}")
        
        if score > best_score:
            best_score = score
            best_pipeline = pipeline
            best_name = name

    print("-" * 30)
    print(f"SIEGER: {best_name} mit {best_score:.2%}")
    print("-" * 30)
    print("Detaillierter Bericht für den Sieger:")
    print(classification_report(y_test, best_pipeline.predict(X_test)))
    
    # 6. Speichern
    if not os.path.exists("models"):
        os.makedirs("models")
        
    model_path = "models/email_classifier_pipeline.pkl"
    joblib.dump(best_pipeline, model_path)
    print(f"Modell gespeichert unter: {model_path}")

if __name__ == "__main__":
    train_pipeline()