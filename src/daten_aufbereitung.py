import pandas as pd
import os

def clean_and_merge_labels():
    # Liste deiner Quelldateien
    files = [
        "data/dataset_endgueltig.csv",
        "data/vetrieb_einkauf.csv",
        "data/vetrieb_einkauf_V2.csv",
        "data/mini_data.csv"
    ]
    
    dfs = []
    
    print("--- Starte Datenaufbereitung (3 Klassen) ---")
    
    for path in files:
        if not os.path.exists(path):
            print(f"[WARNUNG] Datei fehlt: {path}")
            continue
            
        try:
            # Daten laden
            df = pd.read_csv(path, sep=';', encoding='utf-8', on_bad_lines='skip')
            
            # Spalten bereinigen (Leerzeichen entfernen, Kleinschreibung)
            df.columns = df.columns.str.strip().str.lower()
            
            # Label-Spalte als String erzwingen
            if 'label' in df.columns:
                df['label'] = df['label'].astype(str).str.strip()
                
                # --- HIER PASSIERT DIE MAGIE ---
                # 1. Alles was 'einkauf' ist -> 'Buchhaltung'
                # 2. Schreibweisen vereinheitlichen (Support, Vertrieb, Buchhaltung)
                
                # Case-Insensitive Mapping
                df['label'] = df['label'].str.lower().replace({
                    'einkauf': 'Buchhaltung',
                    'buchhaltung': 'Buchhaltung',
                    'support': 'Support',
                    'vertrieb': 'Vertrieb'
                })
                
                # Falls durch die Kleinschreibung oben 'buchhaltung' entstand, 
                # wird es durch das replace-Dict (Zeile 38) korrekt großgeschrieben.
                # Um sicher zu gehen, dass 'buchhaltung' aus Zeile 36 auch groß wird:
                df.loc[df['label'] == 'buchhaltung', 'label'] = 'Buchhaltung'
                
                # Filtern: Wir behalten nur die 3 gewünschten Klassen
                valid_labels = ['Buchhaltung', 'Support', 'Vertrieb']
                df = df[df['label'].isin(valid_labels)]
                
            dfs.append(df)
            print(f"[OK] {path}: Verarbeitet.")
            
        except Exception as e:
            print(f"[FEHLER] bei {path}: {e}")

    # Zusammenfügen
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Duplikate entfernen
    full_df.drop_duplicates(subset=['subject', 'body'], inplace=True)
    
    # Speichern der neuen "Master-Datei"
    output_path = "data/training_data_3classes.csv"
    full_df.to_csv(output_path, sep=';', index=False, encoding='utf-8')
    
    print("-" * 30)
    print(f"Fertig! Neue Datei erstellt: {output_path}")
    print(f"Anzahl E-Mails gesamt: {len(full_df)}")
    print("Verteilung der Klassen:")
    print(full_df['label'].value_counts())
    print("-" * 30)
    print("TIPP: Benutze ab jetzt nur noch 'data/training_data_3classes.csv' in deiner train.py!")

if __name__ == "__main__":
    clean_and_merge_labels()