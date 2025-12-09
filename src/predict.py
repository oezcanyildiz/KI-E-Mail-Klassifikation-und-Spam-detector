import joblib
import os
import sys

MODEL_FILE = os.path.join("models", "email_classifier_pipeline.pkl") 

try:
    from spam_preprocces import preprocess_text
except ImportError:
    import re
    def preprocess_text(text):
        if not isinstance(text, str): return ""
        return re.sub(r'[^a-z\s]', '', text.lower()).strip()


# ---------------------------
# Initialisierung (Modell einmalig laden)
# ---------------------------
def _load_categorizer_model():
    if not os.path.exists(MODEL_FILE):
        print(f"[FEHLER] Kategorisierungsmodell nicht gefunden unter: {MODEL_FILE}")
        sys.exit(1)
            
    print(f"Kategorisierungs-Pipeline wird geladen...")
    return joblib.load(MODEL_FILE)

GLOBAL_CATEGORIZER_MODEL = _load_categorizer_model()

# ---------------------------
# Klassifikation (Öffentliche Funktion)
# ---------------------------
def categorize(text):

    if GLOBAL_CATEGORIZER_MODEL is None:
        return "Error: Model not loaded"
        
    clean_text = preprocess_text(text)
    
    if not clean_text:
        return "Unbekannt"

    probabilities = GLOBAL_CATEGORIZER_MODEL.predict_proba([clean_text])[0]
    
    best_idx = probabilities.argmax()
    best_label = GLOBAL_CATEGORIZER_MODEL.classes_[best_idx]
    
    return best_label
