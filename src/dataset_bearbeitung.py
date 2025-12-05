import pandas as pd
import csv

data_input = r"C:\Users\Student\Desktop\Abschluss Projekt\Projekt\data\dataset_kaggle_clean.csv"
data_output = "data/dataset_endgueltig.csv"

def bearbeitung_data():
    df = pd.read_csv(data_input, sep=';', quoting=1)

    mapping = {
        'Software': 'Support',
        'Hardware': 'Support',
        'Buchhaltung': 'Buchhaltung'
    }

    df['label'] = df['label'].map(mapping)

    # WICHTIG: beim Export quoting und sep setzen
    df.to_csv(
        data_output,
        index=False,
        sep=';',
        quoting=csv.QUOTE_ALL,
        encoding='utf-8'
    )
    print("fertig")

bearbeitung_data()
