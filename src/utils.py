import streamlit as st
import pandas as pd
from src.parser import ChatLogParser
from src.analyzer import ChatAnalyzer
import os

@st.cache_data
def load_data(uploaded_file=None, file_path=None):
    """
    Loads and parses data. 
    Can handle either an uploaded file (Streamlit) or a local file path.
    Returns the Analyzer object.
    """
    path_to_use = file_path
    
    if uploaded_file is not None:
        # Save uploaded file temporarily or parse directly
        # Parser expects a path, so we save it.
        with open("temp_upload.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        path_to_use = "temp_upload.csv"
    elif file_path is None:
        return None

    if not os.path.exists(path_to_use):
        return None

    parser = ChatLogParser(path_to_use)
    conv_df, msg_df = parser.parse()
    
    analyzer = ChatAnalyzer(conv_df, msg_df)
    
    # Cleanup temp file
    if uploaded_file is not None and os.path.exists("temp_upload.csv"):
        os.remove("temp_upload.csv")
        
    return analyzer
