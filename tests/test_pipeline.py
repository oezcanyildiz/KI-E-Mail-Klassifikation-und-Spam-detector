from src.spam_predict import classify_spam
from src.predict import categorize

def test_pipeline_end_to_end():
    text = "Hiermit übersende ich die Rechnung für März"
    
    label, prob = classify_spam(text)
    
    if label == "ham":
        category = categorize(text)
        assert category in ["Buchhaltung", "Support", "Einkauf", "Vertrieb"]
