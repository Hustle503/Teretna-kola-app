@echo off
:: ------------------------------------------------------------
:: Pokretanje Streamlit aplikacije "Brzi Data Studio"
:: Automatski aktivira Python okruženje i pokreće app.py
:: ------------------------------------------------------------

:: Provera da li je aktiviran virtuelni environment
if exist .venv (
    echo Aktiviram virtuelno okruzenje...
    call .venv\Scripts\activate
)

:: Pokretanje aplikacije
streamlit run app.py

:: Pauza da se terminal ne zatvori odmah
pause
