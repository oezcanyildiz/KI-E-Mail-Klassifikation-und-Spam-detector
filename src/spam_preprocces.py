import pandas as pd
import re


INPUT_FILE = r"C:\Users\Student\Desktop\Abschluss Projekt\Projekt\data\spam_training.csv"
OUTPUT_FILE = r"C:\Users\Student\Desktop\Abschluss Projekt\Projekt\data\spam_preprocessed.csv"

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)         # URLs entfernen
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess():
    df = pd.read_csv(INPUT_FILE)

    df["clean_text"] = df["clean_text"].astype(str).apply(clean_text)

    df = df[df["clean_text"].str.len() > 3]

    df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    preprocess()
