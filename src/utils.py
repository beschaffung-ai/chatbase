import streamlit as st
import pandas as pd
from src.parser import ChatLogParser
from src.analyzer import ChatAnalyzer
import os
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

# Cache-Verzeichnis für persistente Datenspeicherung
CACHE_DIR = Path(".streamlit/user_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_user_cache_path(username: str) -> Path:
    """Gibt den Cache-Pfad für einen bestimmten Benutzer zurück."""
    safe_name = hashlib.md5(username.encode()).hexdigest()[:16]
    return CACHE_DIR / f"user_{safe_name}"

def save_uploaded_file(uploaded_file, username: str) -> str:
    """
    Speichert die hochgeladene Datei persistent für den Benutzer.
    Gibt den Pfad zur gespeicherten Datei zurück.
    """
    user_path = get_user_cache_path(username)
    user_path.mkdir(parents=True, exist_ok=True)
    
    # Speichere die CSV-Datei
    csv_path = user_path / "uploaded_data.csv"
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Speichere Metadaten
    meta_path = user_path / "metadata.pkl"
    metadata = {
        "filename": uploaded_file.name,
        "upload_time": datetime.now(),
        "size": uploaded_file.size
    }
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    
    return str(csv_path)

def get_cached_data(username: str):
    """
    Lädt gecachte Daten für einen Benutzer, falls vorhanden.
    Gibt (analyzer, metadata) zurück oder (None, None) wenn kein Cache existiert.
    """
    user_path = get_user_cache_path(username)
    csv_path = user_path / "uploaded_data.csv"
    meta_path = user_path / "metadata.pkl"
    analyzer_cache_path = user_path / "analyzer_cache.pkl"
    
    if not csv_path.exists():
        return None, None
    
    # Lade Metadaten
    metadata = None
    if meta_path.exists():
        try:
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
        except:
            pass
    
    # Prüfe ob Cache älter als 7 Tage ist
    if metadata and metadata.get("upload_time"):
        if datetime.now() - metadata["upload_time"] > timedelta(days=7):
            # Cache ist zu alt, lösche ihn
            clear_user_cache(username)
            return None, None
    
    # Versuche gecachten Analyzer zu laden
    if analyzer_cache_path.exists():
        try:
            with open(analyzer_cache_path, "rb") as f:
                analyzer = pickle.load(f)
            return analyzer, metadata
        except:
            # Cache beschädigt, parse neu
            pass
    
    # Parse die Daten neu und cache den Analyzer
    try:
        parser = ChatLogParser(str(csv_path))
        conv_df, msg_df = parser.parse()
        analyzer = ChatAnalyzer(conv_df, msg_df)
        
        # Cache den Analyzer für schnelleres Laden
        try:
            with open(analyzer_cache_path, "wb") as f:
                pickle.dump(analyzer, f)
        except:
            pass  # Pickle kann manchmal fehlschlagen, ignorieren
        
        return analyzer, metadata
    except Exception as e:
        st.error(f"Fehler beim Laden des Caches: {e}")
        return None, None

def clear_user_cache(username: str):
    """Löscht den Cache für einen bestimmten Benutzer."""
    user_path = get_user_cache_path(username)
    if user_path.exists():
        import shutil
        shutil.rmtree(user_path)

@st.cache_data
def load_data(uploaded_file=None, file_path=None, _username: str = None):
    """
    Loads and parses data. 
    Can handle either an uploaded file (Streamlit) or a local file path.
    Returns the Analyzer object.
    """
    path_to_use = file_path
    
    if uploaded_file is not None:
        # Wenn ein Username vorhanden ist, speichere die Datei persistent
        if _username:
            path_to_use = save_uploaded_file(uploaded_file, _username)
        else:
            # Fallback: temporäre Datei
            with open("temp_upload.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            path_to_use = "temp_upload.csv"
    elif file_path is None:
        return None

    if not os.path.exists(path_to_use):
        return None

    try:
        parser = ChatLogParser(path_to_use)
        conv_df, msg_df = parser.parse()
    except ValueError as e:
        # Show CSV preview for debugging
        try:
            preview_df = pd.read_csv(path_to_use, nrows=5, encoding='utf-8', on_bad_lines='skip')
            st.error(f"❌ Fehler beim Parsen: {str(e)}")
            st.warning("**CSV-Vorschau (erste 5 Zeilen):**")
            st.dataframe(preview_df)
            st.info(f"**Erkannte Spalten:** {list(preview_df.columns)}")
        except Exception as preview_error:
            st.error(f"❌ Fehler beim Parsen: {str(e)}")
            st.error(f"Vorschau-Fehler: {preview_error}")
        return None
    except Exception as e:
        st.error(f"❌ Unerwarteter Fehler: {type(e).__name__}: {str(e)}")
        return None
    
    analyzer = ChatAnalyzer(conv_df, msg_df)
    
    # Cache den Analyzer wenn Username vorhanden
    if _username:
        user_path = get_user_cache_path(_username)
        user_path.mkdir(parents=True, exist_ok=True)
        try:
            with open(user_path / "analyzer_cache.pkl", "wb") as f:
                pickle.dump(analyzer, f)
        except:
            pass
    
    # Cleanup temp file (nur wenn kein Username)
    if uploaded_file is not None and not _username and os.path.exists("temp_upload.csv"):
        os.remove("temp_upload.csv")
        
    return analyzer
