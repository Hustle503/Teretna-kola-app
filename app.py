import os
import re
import streamlit as st
import duckdb
import pandas as pd

# ======================================
#  Putanja do baze
# ======================================
DB_PATH = "kola_sk.db"

# ======================================
#  Spajanje .part fajlova u bazu
# ======================================
if not os.path.exists(DB_PATH):
    st.info("🔄 Spajam 48 .part fajlova u jednu bazu...")

    part_files = sorted(
        [f for f in os.listdir(".") if re.match(r"kola_sk\.db\.part\d+", f)],
        key=lambda x: int(re.search(r"part(\d+)", x).group(1))
    )

    if len(part_files) == 48:
        with open(DB_PATH, "wb") as outfile:
            for fname in part_files:
                st.write(f"➡️ Dodajem {fname}")
                with open(fname, "rb") as infile:
                    outfile.write(infile.read())
        st.success(f"✅ Spojeno {len(part_files)} delova → {DB_PATH}")
    else:
        st.error(f"❌ Nađeno {len(part_files)} delova, a očekivano je 48!")

# ======================================
#  Helper funkcije
# ======================================
def run_sql(sql: str) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()

def table_exists(schema: str, table: str) -> bool:
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        result = con.execute(
            f"""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema='{schema}' AND table_name='{table}'
            """
        ).fetchone()[0]
        return result > 0
    finally:
        con.close()

def ensure_kola_view():
    con = duckdb.connect(DB_PATH)
    try:
        has_glavna = table_exists("glavna", "kola")
        has_upd = table_exists("upd", "kola_update")

        if has_glavna and has_upd:
            con.execute("""
                CREATE OR REPLACE VIEW kola_view AS
                SELECT * FROM glavna.kola
                UNION ALL
                SELECT * FROM upd.kola_update
            """)
        elif has_glavna:
            con.execute("CREATE OR REPLACE VIEW kola_view AS SELECT * FROM glavna.kola")
        else:
            con.execute("""
                CREATE OR REPLACE VIEW kola_view AS
                SELECT * FROM kola
            """)
    finally:
        con.close()

# ======================================
#  Provera baze
# ======================================
if os.path.exists(DB_PATH):
    st.success(f"✅ Baza {DB_PATH} je pronađena")

    try:
        # Napravi view ako ne postoji
        ensure_kola_view()

        # Test upit
        df_test = run_sql("SELECT COUNT(*) AS broj_redova FROM kola_view")
        st.write("📊 Broj redova u `kola_view`:", df_test.iloc[0,0])

        # Pregled tabela
        tabele = run_sql("SHOW TABLES")
        st.write("📂 Dostupne tabele:", tabele)

    except Exception as e:
        st.error(f"❌ Ne mogu da pročitam bazu: {e}")
else:
    st.error("❌ Baza kola_sk.db nije pronađena!")

# ======================================
#  Dashboard (placeholder)
# ======================================
st.title("🚃 Teretna kola SK — kontrolna tabla")
st.write("📊 Pregled podataka i izveštaji...")
