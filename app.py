import duckdb
import streamlit as st
import os

# -------------------------
# Podesavanja
# -------------------------
PARQUET_FOLDER = r"C:\Teretna kola\parquet"   # folder sa mesečnim parquet fajlovima
EXCEL_STANJE = r"C:\Teretna kola\stanje.xlsx"
EXCEL_STANICE = r"C:\Teretna kola\stanice.xlsx"

# -------------------------
# DuckDB konekcija
# -------------------------
@st.cache_resource
def get_connection():
    con = duckdb.connect()
    # registruj parquet folder kao tabelu
    con.execute(f"""
        CREATE OR REPLACE VIEW kola AS
        SELECT * FROM read_parquet('{PARQUET_FOLDER}/*.parquet')
    """)
    return con

con = get_connection()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(layout="wide")
st.title("🚂 Teretna kola SK — Parquet + DuckDB")

st.sidebar.title("⚙️ Podešavanja")

# koliko redova ima ukupno
try:
    total_rows = con.execute("SELECT COUNT(*) FROM kola").fetchone()[0]
    st.sidebar.write(f"📂 Ukupan broj redova: {total_rows:,}".replace(",", "."))
except Exception as e:
    st.sidebar.error(f"Greška pri učitavanju: {e}")
    st.stop()

# -------------------------
# Tabovi
# -------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Pregled", "📈 Izveštaji", "🔎 Ad-hoc SQL"
])

# --- Tab 1: Pregled ---
with tab1:
    st.subheader("Prvih 100 redova iz baze")
    df_head = con.execute("SELECT * FROM kola LIMIT 100").df()
    st.dataframe(df_head, use_container_width=True)

# --- Tab 2: Izveštaji ---
with tab2:
    st.subheader("Suma NetoTone po mesecu")
    try:
        df_month = con.execute("""
            SELECT DATE_TRUNC('month', CAST(DatumVreme AS TIMESTAMP)) AS mesec,
                   SUM(NetoTone) AS suma
            FROM kola
            GROUP BY 1
            ORDER BY 1
        """).df()
        st.line_chart(df_month.set_index("mesec")["suma"])
    except Exception as e:
        st.warning(f"Nema kolone DatumVreme ili NetoTone ({e})")

    st.subheader("Top 20 stanica po broju vagona")
    try:
        df_sta = con.execute("""
            SELECT Stanica, COUNT(*) AS broj
            FROM kola
            GROUP BY Stanica
            ORDER BY broj DESC
            LIMIT 20
        """).df()
        st.bar_chart(df_sta.set_index("Stanica")["broj"])
    except Exception as e:
        st.warning(f"Nema kolone Stanica ({e})")

# --- Tab 3: Ad-hoc SQL ---
with tab3:
    st.subheader("SQL upit nad Parquet podacima")
    query = st.text_area("Unesi SQL upit", 
        "SELECT Stanica, COUNT(*) AS n FROM kola GROUP BY Stanica ORDER BY n DESC LIMIT 10"
    )
    if st.button("Pokreni upit"):
        try:
            result = con.execute(query).df()
            st.success(f"✅ Rezultat: {len(result)} redova")
            st.dataframe(result, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Greška: {e}")
