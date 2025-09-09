import os
import re
import io
import time
import duckdb
import pandas as pd
import streamlit as st
import gdown

# =========================
# Konstante
# =========================
DB_PATH = "kola_sk.db"
FOLDER_ID = "1q__8P3gY-JMzqD5cpt8avm_7VAY-fHWI"

# =========================
# Preuzimanje fajlova sa Google Drive
# =========================
if not os.path.exists(DB_PATH):
    st.info("☁️ Preuzimam delove baze sa Google Drive...")

    try:
        gdown.download_folder(id="1q__8P3gY-JMzqD5cpt8avm_7VAY-fHWI", quiet=False, use_cookies=False)
    except Exception as e:
        st.error(f"❌ Greška pri preuzimanju sa Google Drive: {e}")

    # Spajanje .part fajlova
    part_files = sorted(
        [f for f in os.listdir(".") if re.match(r"kola_sk\.db\.part\d+", f)],
        key=lambda x: int(re.search(r"part(\d+)", x).group(1))
    )

    if part_files and len(part_files) == 48:
        with open(DB_PATH, "wb") as outfile:
            for fname in part_files:
                st.write(f"➡️ Dodajem {fname}")
                with open(fname, "rb") as infile:
                    outfile.write(infile.read())
        st.success(f"✅ Spojeno {len(part_files)} delova → {DB_PATH}")
    else:
        st.error(f"❌ Nađeno samo {len(part_files)} fajlova, a treba 48.")

# =========================
# Provera i inicijalizacija baze
# =========================
if os.path.exists(DB_PATH):
    st.success(f"✅ Baza {DB_PATH} je pronađena")
    st.write("📂 Veličina fajla:", os.path.getsize(DB_PATH), "bajta")

    try:
        con = duckdb.connect(DB_PATH)
        con.execute("""
            CREATE OR REPLACE VIEW kola_view AS
            SELECT * FROM kola
        """)
        con.close()
        st.success("✅ 'kola_view' je spreman za upotrebu")
    except Exception as e:
        st.error(f"❌ Problem sa bazom: {e}")
else:
    st.error(f"❌ Baza {DB_PATH} nije pronađena")


# =========================
# Funkcije za rad sa bazom
# =========================
def run_sql(db_file: str, sql: str) -> pd.DataFrame:
    con = duckdb.connect(db_file, read_only=True)
    try:
        df = con.execute(sql).fetchdf()
    finally:
        con.close()
    return df


def table_exists(schema: str, table: str) -> bool:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        result = con.execute(
            f"SELECT COUNT(*) FROM information_schema.tables "
            f"WHERE table_schema='{schema}' AND table_name='{table}'"
        ).fetchone()[0]
    finally:
        con.close()
    return result > 0


def create_or_replace_table_from_df(db_file: str, table_name: str, df: pd.DataFrame):
    con = duckdb.connect(db_file)
    try:
        con.register("df_tmp", df)
        con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM df_tmp')
        con.unregister("df_tmp")
    finally:
        con.close()

# =========================
# Test rada baze
# =========================
if os.path.exists(DB_PATH):
    try:
        df_test = run_sql(DB_PATH, "SELECT COUNT(*) AS broj_redova FROM kola_view")
        st.write("📊 Broj redova u kola_view:", df_test.iloc[0, 0])
    except Exception:
        df_test = run_sql(DB_PATH, "SELECT COUNT(*) AS broj_redova FROM kola")
        st.write("📊 Broj redova u kola:", df_test.iloc[0, 0])
else:
    st.error(f"❌ Baza {DB_PATH} nije pronađena")

# =========================
# Sidebar
# =========================
st.sidebar.title("⚙️ Podešavanja")
st.sidebar.caption("Glavna baza: kola_sk.db (auto download). Opciona lokalna baza za UNION: kola_sk_update.db")
st.sidebar.markdown("---")

st.sidebar.subheader("📂 Uvoz Excela → tabela 'stanje'")
uploaded_excel_stanje = st.sidebar.file_uploader("Izaberi Excel (.xlsx)", type=["xlsx"], key="stanje_up")
if uploaded_excel_stanje is not None:
    if st.sidebar.button("📥 Učitaj u bazu kao 'stanje'"):
        try:
            df_stanje = pd.read_excel(uploaded_excel_stanje)
            create_or_replace_table_from_df(DB_PATH, "stanje", df_stanje)
            st.sidebar.success(f"✅ 'stanje' učitano ({len(df_stanje)} redova).")
        except Exception as e:
            st.sidebar.error(f"❌ Greška: {e}")

st.sidebar.subheader("🗺️ Uvoz mape stanica → tabela 'stanice'")
uploaded_excel_stanice = st.sidebar.file_uploader("Izaberi Excel (.xlsx)", type=["xlsx"], key="stanice_up")
if uploaded_excel_stanice is not None:
    if st.sidebar.button("📥 Učitaj u bazu kao 'stanice'"):
        try:
            df_st = pd.read_excel(uploaded_excel_stanice)
            create_or_replace_table_from_df(DB_PATH, "stanice", df_st)
            st.sidebar.success(f"✅ 'stanice' učitano ({len(df_st)} redova).")
        except Exception as e:
            st.sidebar.error(f"❌ Greška: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("Sve tabele možete koristiti u SQL upitima. Glavni podaci su u 'kola_view'.")

# =========================
# Glavni naslov i tabovi
# =========================
st.title("🚃 Teretna kola SK — kontrolna tabla")

tab1, tab2, tab3 = st.tabs([
    "📊 Pregled", "📈 Izveštaji", "🔎 SQL upiti"
])

# Tab 1
with tab1:
    try:
        df_cnt = run_sql(DB_PATH, "SELECT COUNT(*) AS broj_redova FROM kola_view")
        st.metric("Ukupan broj redova", f"{int(df_cnt['broj_redova'][0]):,}".replace(",", "."))
    except Exception as e:
        st.error(f"Ne mogu da pročitam bazu: {e}")
        st.stop()

# Tab 2
with tab2:
    st.subheader("Suma NetoTone po mesecu")
    df_month = run_sql(DB_PATH, """
        SELECT date_trunc('month', DatumVreme) AS mesec,
               SUM(COALESCE("NetoTone", 0)) AS ukupno_tona
        FROM kola_view
        WHERE DatumVreme IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """)
    if not df_month.empty:
        st.line_chart(df_month.set_index("mesec")["ukupno_tona"])

# Tab 3
with tab3:
    st.subheader("Piši svoj SQL (koristi npr. kola_view, stanje, stanice)")
    default_sql = "SELECT * FROM kola_view LIMIT 100"
    user_sql = st.text_area("SQL:", height=160, value=default_sql)
    if st.button("▶️ Izvrši upit"):
        try:
            df_user = run_sql(DB_PATH, user_sql)
            st.success(f"OK — {len(df_user):,} redova".replace(",", "."))
            st.dataframe(df_user, use_container_width=True)
        except Exception as e:
            st.error(f"Greška u upitu: {e}")
