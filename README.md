# Chatbase Analytics Dashboard

Ein interaktives Dashboard zur Analyse von Chatbot-Protokollen aus Chatbase, entwickelt mit Streamlit und Python.

## Features

### 📊 Überblick & Support-Steuerung
- **KPIs**: Gesamtzahl Konversationen, Nachrichten, Ø Dauer.
- **Support-Heatmap**: Visualisierung der Stoßzeiten (Wochentag vs. Uhrzeit) zur Personalplanung.
- **Zeitverlauf**: Tägliche Entwicklung des Chat-Aufkommens.

### 💬 Themen & Inhalte
- **Exit-Analyse**: Wer beendet das Gespräch (User vs. Bot)? Was sind die häufigsten letzten Sätze?
- **Text-Mining**: Häufigste Phrasen (Bigrams/Trigrams) und Wordclouds.
- **Topic Clustering**: Automatische Gruppierung von Gesprächen nach Inhalt.

### 🧠 Qualität & Sentiment
- **Komplexitäts-Analyse**: Unterscheidung in kurze (Quick-Fix) vs. lange (Problem) Chats.
- **Sentiment-Analyse**: Stimmung der User-Nachrichten.

### 📂 Daten-Explorer
- Vollständige Durchsuchbarkeit aller Nachrichten und Metadaten.

## Installation

1. Repository klonen oder entpacken.
2. Python-Umgebung erstellen (empfohlen Python 3.9+).
3. Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

4. NLTK Daten werden beim ersten Start automatisch heruntergeladen.

## Nutzung

Starten Sie das Dashboard mit:

```bash
python -m streamlit run app.py
```

Das Dashboard öffnet sich automatisch im Browser (lokal unter `http://localhost:8501`).

### Datenformat
Das Tool erwartet einen CSV-Export von Chatbase im "Block-Format", der Metadaten und Nachrichtenverläufe enthält. Falls keine Datei hochgeladen wird, sucht das Tool nach einer Standard-CSV im Projektordner.

## Technologien
- **Frontend**: Streamlit
- **Visualisierung**: Plotly, Wordcloud, Matplotlib
- **Analyse**: Pandas, Scikit-learn, TextBlob-DE, NLTK
