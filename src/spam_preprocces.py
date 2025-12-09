import pandas as pd
import re
import nltk
from nltk.corpus import stopwords 


INPUT_FILE = r"C:\Users\Student\Desktop\Abschluss Projekt\Projekt\data\spam_training.csv"
OUTPUT_FILE = r"C:\Users\Student\Desktop\Abschluss Projekt\Projekt\data\spam_preprocessed.csv"

# --- Initialisierung der Stoppwörter ---
try:
    GERMAN_STOP_WORDS = set(stopwords.words('german'))
except LookupError:
    GERMAN_STOP_WORDS = set(['der', 'die', 'das', 'und', 'ist', 'ich', 'sie', 'er']) 
    

def clean_text(text):
    text = str(text)
    text = text.lower()

    text = re.sub(r"http\S+", "", text) 
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß ]", " ", text)
    
    words = text.split()

    filtered_words = [word for word in words if word not in GERMAN_STOP_WORDS]

    text = " ".join(filtered_words)
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def preprocess():
    df = pd.read_csv(INPUT_FILE)

    df["clean_text"] = df["clean_text"].astype(str).apply(clean_text)

    df = df[df["clean_text"].str.len() > 3]

    df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    preprocess()
