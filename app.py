import os
import re
import duckdb
import gdown
import pandas as pd
import streamlit as st

# URL foldera na Google Drive-u
FOLDER_URL = "https://drive.google.com/drive/folders/1q__8P3gY-JMzqD5cpt8avm_7VAY-fHWI?usp=sharing"

DB_PATH = "kola_sk.db"

# Preuzmi delove ako baza još ne postoji
if not os.path.exists(DB_PATH):
    st.info("⬇️ Preuzimam sve .part fajlove sa Google Drive foldera...")
    # preuzima sve fajlove iz foldera
    gdown.download_folder(id="1q__8P3gY-JMzqD5cpt8avm_7VAY-fHWI", quiet=False, use_cookies=False)


    st.success("✅ Svi delovi preuzeti! Mogao bi sada da se spoje.")
