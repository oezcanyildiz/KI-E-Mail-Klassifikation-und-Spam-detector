1. Übersicht

Dieses Repository enthält den Prototyp der E-Mail-Klassifikations-Pipeline. Die Anwendung ist als sequenzielle Machine-Learning-Pipeline in Python implementiert und dient als Proof of Concept (PoC) zur automatisierten Vorsortierung eingehender E-Mails.

Das System führt zwei Hauptfunktionen aus:

    Stufe 1: Überprüfung auf Spam (Filterung).

    Stufe 2: Thematische Klassifikation (Kategorisierung) von legitimen E-Mails.

2. Systemanforderungen und Setup

Die Anwendung wurde für lokale Ausführung auf Standard-CPUs entwickelt und erfordert keine speziellen Server oder GPUs.

2.1 Voraussetzungen

    Python: Version 3.x

    Bibliotheken: Alle Abhängigkeiten sind in der Datei requirements.txt gelistet (z.B. scikit-learn, pandas, joblib).

2.2 Installation

    Klone das Repository lokal.

    Installiere die benötigten Python-Bibliotheken:
    Bash

    pip install -r requirements.txt

3. Nutzung der Inferenz-Pipeline (Klassifikation)

Die Klassifikation neuer E-Mails erfolgt über das Hauptskript src/main.py (CLI), nachdem die Modelle trainiert wurden.
3.1 Modell-Initialisierung

Stellen Sie sicher, dass die Modelle (spam_logreg.pkl und email_classifier_pipeline.pkl) im Ordner models/ vorhanden sind. Diese werden beim Start des Hauptskripts einmalig in den Speicher geladen, um eine schnelle Klassifikation zu gewährleisten.
3.2 Ausführung der Klassifikation (CLI)

Führen Sie das Hauptskript aus und geben Sie den zu analysierenden E-Mail-Text in die Konsole ein:
Bash

python src/main.py

Das Skript fordert zur Eingabe auf:

Bitte Email-Text eingeben:

3.3 Ergebnisse und Weiterleitungsempfehlung

Die Pipeline liefert ein sequenzielles Ergebnis:
A. Ergebnis bei Spam-Erkennung

Wenn die Spam-Wahrscheinlichkeit über 75% liegt:

📊 Ergebnis der Spam-Prüfung:
→ Hauptklasse: spam
🛑 Nachricht wurde als SPAM klassifiziert.

(Die Nachricht wird protokolliert und der Prozess beendet.)
B. Ergebnis bei Themenklassifikation

Wenn die E-Mail als 'Ham' erkannt wird:

📊 Ergebnis der Spam-Prüfung:
→ Hauptklasse: ham
✔️ Nachricht ist kein Spam. Leite zur Kategorisierung weiter...
🎉 Endergebnis: E-Mail ist 'Ham' und gehört zur Kategorie: **[Kategorie]**

(Das Ergebnis [Kategorie] dient als Weiterleitungsempfehlung an die zuständige Fachabteilung.)
4. Retraining und Erweiterung

Um die Modelle zu aktualisieren oder auf neue Kategorien zu erweitern, führen Sie das Trainingsskript aus:
Bash

python src/spam_train.py

    Dieses Skript liest die vorverarbeiteten Trainingsdaten ein.

    Es trainiert den TF-IDF Vectorizer neu.

    Es trainiert die Logistische Regression für die Klassifikation neu.

    Es speichert die neuen Modellartefakte automatisch im Ordner models/