import os
import glob
import time
import pandas as pd
import streamlit as st

# -------------------------
# Podesavanja
# -------------------------
PARQUET_FOLDER = r"C:\Teretna kola\parquet"   # folder sa mesečnim parquet fajlovima
EXCEL_STANJE = r"C:\Teretna kola\stanje.xlsx"
EXCEL_STANICE = r"C:\Teretna kola\stanice.xlsx"

# -------------------------
# Učitavanje podataka
# -------------------------
@st.cache_data(show_spinner=False)
def load_parquet_data(folder: str) -> pd.DataFrame:
    """Učitaj sve parquet fajlove iz foldera i spoji u jedan DataFrame"""
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        return pd.DataFrame()
    df_list = [pd.read_parquet(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    return df

@st.cache_data(show_spinner=False)
def load_excel_data(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_excel(path)
    return pd.DataFrame()

# Učitaj glavne podatke
df_kola = load_parquet_data(PARQUET_FOLDER)
df_stanje = load_excel_data(EXCEL_STANJE)
df_stanice = load_excel_data(EXCEL_STANICE)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(layout="wide")
st.title("🚂 Teretna kola SK — kontrolna tabla (Parquet verzija)")

st.sidebar.title("⚙️ Podešavanja")
st.sidebar.write(f"📂 Učitano redova: {len(df_kola):,}".replace(",", "."))

# -------------------------
# Tabovi
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Pregled", "📈 Izveštaji", "🗂 Excel baze", "🔎 Ad-hoc upiti"
])

# --- Tab 1: Pregled ---
with tab1:
    if df_kola.empty:
        st.warning("⚠️ Nema podataka u Parquet folderu.")
    else:
        st.metric("Ukupan broj redova", f"{len(df_kola):,}".replace(",", "."))
        st.dataframe(df_kola.head(100), use_container_width=True)

# --- Tab 2: Izveštaji ---
with tab2:
    if not df_kola.empty:
        st.subheader("Suma NetoTone po mesecu")
        if "DatumVreme" in df_kola.columns:
            df_kola["mesec"] = pd.to_datetime(df_kola["DatumVreme"]).dt.to_period("M")
            df_month = df_kola.groupby("mesec")["NetoTone"].sum().reset_index()
            st.line_chart(df_month.set_index("mesec")["NetoTone"])

        st.subheader("Top 20 stanica po broju vagona")
        if "Stanica" in df_kola.columns:
            df_sta = (
                df_kola.groupby("Stanica")
                .size()
                .reset_index(name="broj")
                .sort_values("broj", ascending=False)
                .head(20)
            )
            st.bar_chart(df_sta.set_index("Stanica")["broj"])

# --- Tab 3: Excel baze ---
with tab3:
    st.subheader("Stanje.xlsx")
    if df_stanje.empty:
        st.info("📂 Nema fajla stanje.xlsx")
    else:
        st.dataframe(df_stanje.head(50), use_container_width=True)

    st.subheader("Stanice.xlsx")
    if df_stanice.empty:
        st.info("📂 Nema fajla stanice.xlsx")
    else:
        st.dataframe(df_stanice.head(50), use_container_width=True)

# --- Tab 4: Ad-hoc upiti ---
with tab4:
    st.subheader("SQL upit nad Parquet podacima (DuckDB in-memory)")

    import duckdb
    con = duckdb.connect()
    con.register("kola", df_kola)

    query = st.text_area("Unesi SQL upit", "SELECT Stanica, COUNT(*) AS n FROM kola GROUP BY Stanica ORDER BY n DESC LIMIT 10")
    if st.button("Pokreni upit"):
        try:
            result = con.execute(query).df()
            st.success(f"✅ Rezultat: {len(result)} redova")
            st.dataframe(result, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Greška: {e}")
