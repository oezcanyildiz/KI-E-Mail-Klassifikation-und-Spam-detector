from src.spam_predict import classify_spam, save_spam_to_log
from src.predict import categorize
import sys

def main():
    text = input("Bitte Email-Text eingeben:\n")
    
    if not text.strip():
        print("Eingabe darf nicht leer sein.")
        return

    # 1. Spam-Prüfung (Modell 1)
    label, prob_spam = classify_spam(text)

    print("\n📊 Ergebnis der Spam-Prüfung:")
    print(f"→ Hauptklasse: {label}")
    print(f"→ Spam-Wahrscheinlichkeit: {prob_spam:.2%}")

    if label == "spam":
        # Fall 1: SPAM gefunden
        print("🛑 Nachricht wurde als **SPAM** klassifiziert.")
        save_spam_to_log(text, label, prob_spam)
        print("💾 Ereignis gespeichert.")
    else:
        # Fall 2: KEIN SPAM (HAM) -> Weiterleitung an Modell 2
        print("\n Nachricht ist kein Spam. Leite zur **Kategorisierung** weiter...")
        
        # 2. Kategorisierung (Modell 2)
        category_label = categorize(text) 
        
        print(f" Endergebnis: E-Mail ist 'Ham' und gehört zur Kategorie: **{category_label}**")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgramm beendet.")
        sys.exit(0)
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")