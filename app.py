import os
import re
import duckdb
import pandas as pd
import streamlit as st

# =========================
#  Spajanje delova u kola_sk.db
# =========================
DB_PATH = "kola_sk.db"

if not os.path.exists(DB_PATH):
    st.info("🔄 Spajam 48 .part fajlova u jednu bazu...")

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
        st.error(f"❌ Nije pronađeno svih 48 fajlova (.part1 … .part48). Nađeno: {len(part_files)}")

# =========================
#  Funkcije za rad sa bazom
# =========================
def run_sql(sql: str) -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=False)
    try:
        df = con.execute(sql).fetchdf()
    finally:
        con.close()
    return df

def table_exists(schema: str, table: str) -> bool:
    con = duckdb.connect(DB_PATH, read_only=False)
    try:
        result = con.execute(
            f"""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema='{schema}' AND table_name='{table}'
            """
        ).fetchone()[0]
    finally:
        con.close()
    return result > 0

# =========================
#  Provera i inicijalizacija baze
# =========================
if os.path.exists(DB_PATH):
    st.success(f"✅ Baza {DB_PATH} je pronađena")
    st.write("📂 Veličina fajla:", os.path.getsize(DB_PATH), "bajta")

    try:
        con = duckdb.connect(DB_PATH, read_only=False)
        # napravi view ako postoji tabela kola
        con.execute("CREATE OR REPLACE VIEW kola_view AS SELECT * FROM kola")
        con.close()
        st.success("✅ 'kola_view' je spreman za upotrebu")
    except Exception as e:
        st.error(f"❌ Problem sa bazom: {e}")
else:
    st.error(f"❌ Baza {DB_PATH} nije pronađena")
