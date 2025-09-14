import os
import re
import time
import duckdb
import hashlib
import json
import pandas as pd
import polars as pl
import streamlit as st
import gdown

# =========================
# Putevi i folderi
# =========================
DB_PATH = "kola_sk.db"   # više nam i ne treba .db.part
NOVI_UNOS_FOLDER = "novi_unos"
NOVI_UNOS_FOLDER_ID = "1XQEUt3_TjM_lWahZHoZmlANExIwDwBW1"
HASH_FILE = "novi_unos_hash.json"

# =========================
# DuckDB konekcija
# =========================
if "con" not in st.session_state:
    st.session_state.con = duckdb.connect(DB_PATH)
con = st.session_state.con

# =========================
# Funkcija za SQL upite
# =========================
def run_sql(sql: str) -> pd.DataFrame:
    try:
        return con.execute(sql).fetchdf()
    except Exception as e:
        st.error(f"Greška u upitu: {e}")
        return pd.DataFrame()

# =========================
# Parsiranje TXT fajlova
# =========================
def parse_txt(path) -> pl.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rows.append({
                "Režim": line[0:2].strip(),
                "Vlasnik": line[2:4].strip(),
                "Serija": line[4:7].strip(),
                "Inv br": line[7:11].strip(),
                "KB": line[11:12].strip(),
                "Tip kola": line[12:15].strip(),
                "Voz br": line[15:20].strip(),
                "Stanica": line[20:25].strip(),
                "Status": line[25:27].strip(),
                "Datum": line[27:35].strip(),
                "Vreme": line[35:39].strip(),
                "Roba": line[41:47].strip(),
                "Reon": line[61:66].strip(),
                "tara": line[78:81].strip(),
                "NetoTone": line[83:86].strip(),
                "Broj vagona": line[0:12].strip(),
                "Broj kola": line[2:11].strip(),
                "source_file": os.path.basename(path),
            })

    df = pl.DataFrame(rows)
    df = df.with_columns([
        pl.when(pl.col("Vreme") == "2400").then(pl.lit("0000")).otherwise(pl.col("Vreme")).alias("Vreme"),
        (pl.col("Datum") + " " + pl.col("Vreme")).str.strptime(pl.Datetime, "%Y%m%d %H%M", strict=False).alias("DatumVreme"),
    ])
    df = df.with_columns([
        pl.col("tara").cast(pl.Int32, strict=False),
        pl.col("NetoTone").cast(pl.Int32, strict=False),
        pl.col("Inv br").cast(pl.Int32, strict=False),
        pl.lit(None).alias("broj_kola_bez_rezima_i_kb")
    ])
    return df

# =========================
# Hash funkcije
# =========================
def hash_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def load_hashes():
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return json.load(f)
    return {}

def save_hashes(hashes):
    with open(HASH_FILE, "w") as f:
        json.dump(hashes, f)

# =========================
# Učitavanje TXT fajlova samo ako su novi/izmenjeni
# =========================
def load_novi_unosi():
    os.makedirs(NOVI_UNOS_FOLDER, exist_ok=True)
    txt_files = [os.path.join(NOVI_UNOS_FOLDER, f) for f in os.listdir(NOVI_UNOS_FOLDER) if f.lower().endswith(".txt")]
    if not txt_files:
        return pl.DataFrame()

    old_hashes = load_hashes()
    new_hashes = {}
    changed_files = []

    for f in txt_files:
        h = hash_file(f)
        new_hashes[f] = h
        if f not in old_hashes or old_hashes[f] != h:
            changed_files.append(f)

    save_hashes(new_hashes)

    if not changed_files:
        st.info("ℹ️ Nema novih ili izmenjenih TXT fajlova.")
        return pl.DataFrame()

    st.info(f"⬇️ Učitavam {len(changed_files)} novih/izmenjenih fajlova...")
    dfs = [parse_txt(f) for f in changed_files]
    return pl.concat(dfs)

# =========================
# Učitavanje Parquet fajlova
# =========================
def load_parquet_files(folder: str) -> pl.DataFrame:
    parquet_files = sorted([f for f in os.listdir(folder) if f.startswith("combined_") and f.endswith(".parquet")])
    if not parquet_files:
        st.warning("⚠️ Nema combined_00X.parquet fajlova!")
        return pl.DataFrame()

    dfs = [pl.read_parquet(os.path.join(folder, f)) for f in parquet_files]
    df = pl.concat(dfs, rechunk=True)

    # Ako su sirovi u jednoj koloni "line"
    if df.shape[1] == 1:
        col = df.columns[0]
        rows = []
        for line in df[col].to_list():
            rows.append({
                "Režim": line[0:2].strip(),
                "Vlasnik": line[2:4].strip(),
                "Serija": line[4:7].strip(),
                "Inv br": line[7:11].strip(),
                "KB": line[11:12].strip(),
                "Tip kola": line[12:15].strip(),
                "Voz br": line[15:20].strip(),
                "Stanica": line[20:25].strip(),
                "Status": line[25:27].strip(),
                "Datum": line[27:35].strip(),
                "Vreme": line[35:39].strip(),
                "Roba": line[41:47].strip(),
                "Reon": line[61:66].strip(),
                "tara": line[78:81].strip(),
                "NetoTone": line[83:86].strip(),
                "Broj vagona": line[0:12].strip(),
                "Broj kola": line[2:11].strip(),
                "source_file": "parquet",
            })
        df = pl.DataFrame(rows)

    df = df.with_columns([
        pl.when(pl.col("Vreme") == "2400").then(pl.lit("0000")).otherwise(pl.col("Vreme")).alias("Vreme"),
        (pl.col("Datum") + " " + pl.col("Vreme")).str.strptime(pl.Datetime, "%Y%m%d %H%M", strict=False).alias("DatumVreme"),
        pl.col("tara").cast(pl.Int32, strict=False),
        pl.col("NetoTone").cast(pl.Int32, strict=False),
        pl.col("Inv br").cast(pl.Int32, strict=False),
    ])
    return df

# =========================
# Preuzimanje TXT fajlova (opciono, ne kritično)
# =========================
os.makedirs(NOVI_UNOS_FOLDER, exist_ok=True)
folder_url_txt = f"https://drive.google.com/drive/folders/{NOVI_UNOS_FOLDER_ID}"

try:
    import gdown
    st.info(f"⬇️ Pokušavam da preuzmem TXT fajlove iz foldera: {folder_url_txt}")
    gdown.download_folder(
        url=folder_url_txt,
        output=NOVI_UNOS_FOLDER,
        quiet=True,
        use_cookies=False
    )
    st.success("✅ Preuzimanje završeno ili fajlovi već postoje.")
except Exception as e:
    st.warning(f"⚠️ Nije uspelo preuzimanje TXT fajlova ({e}). "
               "Koristiću fajlove koji su već u folderu 'novi_unos'.")

# =========================
# Učitavanje Parquet fajlova → kola_sk
# =========================
df_parquet = load_parquet_files(".")
if df_parquet.height > 0:
    con.register("df_parquet", df_parquet.to_pandas())
    con.execute("DROP TABLE IF EXISTS kola_sk")
    con.execute("CREATE TABLE kola_sk AS SELECT * FROM df_parquet")
    con.unregister("df_parquet")

# =========================
# Učitavanje TXT fajlova → novi_unosi
# =========================
df_all = load_novi_unosi()
if df_all.height > 0:
    con.register("df_novi", df_all.to_pandas())
    con.execute("DROP TABLE IF EXISTS novi_unosi")
    con.execute("CREATE TABLE novi_unosi AS SELECT * FROM df_novi")
    con.unregister("df_novi")

# =========================
# Kreiranje view-a kola_sve
# =========================
con.execute("DROP VIEW IF EXISTS kola_sve")
tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
if "kola_sk" in tables and "novi_unosi" in tables:
    con.execute("""
        CREATE VIEW kola_sve AS
        SELECT * FROM kola_sk
        UNION ALL
        SELECT * FROM novi_unosi
    """)
elif "kola_sk" in tables:
    con.execute("CREATE VIEW kola_sve AS SELECT * FROM kola_sk")
elif "novi_unosi" in tables:
    con.execute("CREATE VIEW kola_sve AS SELECT * FROM novi_unosi")
else:
    st.warning("⚠️ Nema podataka za kreiranje view-a 'kola_sve'")

# =========================
# Default tabela/view
# =========================
DEFAULT_TABLE = "kola_sve"
table_name = DEFAULT_TABLE

# =========================
# Naslov + dalje tabovi (ostaju isti)
# =========================
st.title("🚃 Teretna kola SK — kontrolna tabla")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Pregled", "📈 Izveštaji", "🔎 SQL upiti", "🔬 Pregled podataka", "📌 Poslednji unosi", "🔍 Pretraga kola", "📊 Kola po stanicima", "🚂 Kretanje 4098 kola–TIP 0", "🚂 Kretanje 4098 kola–TIP 1", "📊 Kola po serijama"])

# ---------- Tab 1: Pregled ----------
with tab1:
    col_a, col_b, col_c, col_d = st.columns(4)
    try:
        df_cnt = run_sql(f'SELECT COUNT(*) AS broj_redova FROM "{table_name}"')
        col_a.metric("Ukupan broj redova", f"{int(df_cnt['broj_redova'][0]):,}".replace(",", "."))

        df_files = run_sql(f'SELECT COUNT(DISTINCT source_file) AS fajlova FROM "{table_name}"')
        col_b.metric("Učitanih fajlova", int(df_files["fajlova"][0]))

        df_range = run_sql(
            f'''
            SELECT
              MIN(DatumVreme) AS min_dt,
              MAX(DatumVreme) AS max_dt
            FROM "{table_name}"
            WHERE DatumVreme IS NOT NULL
            '''
        )
        min_dt = str(df_range["min_dt"][0]) if df_range["min_dt"][0] is not None else "—"
        max_dt = str(df_range["max_dt"][0]) if df_range["max_dt"][0] is not None else "—"
        col_c.metric("Najraniji datum", min_dt)
        col_d.metric("Najkasniji datum", max_dt)

        st.divider()
        st.subheader("Učitanih redova po fajlu (top 20)")
        df_by_file = run_sql(
            f'''
            SELECT source_file, COUNT(*) AS broj
            FROM "{table_name}"
            GROUP BY source_file
            ORDER BY broj DESC
            LIMIT 20
            '''
        )
        st.dataframe(df_by_file, use_container_width=True)

    except Exception as e:
        st.error(f"Ne mogu da pročitam bazu: {e}")
        st.stop()

# ---------- Tab 2: Izveštaji ----------
with tab2:
    st.subheader("Suma NetoTone po mesecu")
    q_month = f"""
        SELECT
          date_trunc('month', DatumVreme) AS mesec,
          SUM(COALESCE("NetoTone", 0)) AS ukupno_tona
        FROM "{table_name}"
        WHERE DatumVreme IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """
    df_month = run_sql(q_month)
    st.line_chart(df_month.set_index("mesec")["ukupno_tona"])

    st.subheader("Top 20 stanica po broju vagona")
    q_sta = f"""
        SELECT "Stanica", COUNT(*) AS broj
        FROM "{table_name}"
        GROUP BY "Stanica"
        ORDER BY broj DESC
        LIMIT 20
    """
    df_sta = run_sql(q_sta)
    st.bar_chart(df_sta.set_index("Stanica")["broj"])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Prosečna NetoTone po tipu kola")
        q_tip = f"""
            SELECT "Tip kola" AS tip, AVG(COALESCE("NetoTone", 0)) AS prosek_tona
            FROM "{table_name}"
            GROUP BY tip
            ORDER BY prosek_tona DESC
            LIMIT 20
        """
        df_tip = run_sql(q_tip)
        st.dataframe(df_tip, use_container_width=True)
    with c2:
        st.subheader("Prosečna tara po tipu kola")
        q_tara = f"""
            SELECT "Tip kola" AS tip, AVG(COALESCE("tara", 0)) AS prosek_tare
            FROM "{table_name}"
            GROUP BY tip
            ORDER BY prosek_tare DESC
            LIMIT 20
        """
        df_tara = run_sql(q_tara)
        st.dataframe(df_tara, use_container_width=True)

# ---------- Tab 3: SQL upiti ----------
with tab3:
    st.subheader("Piši svoj SQL")
    default_sql = f'SELECT * FROM "{table_name}" LIMIT 100'
    user_sql = st.text_area("SQL:", height=160, value=default_sql)
    colx, coly = st.columns([1, 3])
    run_btn = colx.button("▶️ Izvrši upit")
    if run_btn:
        t0 = time.time()
        try:
            df_user = run_sql(user_sql)
            elapsed = time.time() - t0
            st.success(f"OK ({elapsed:.2f}s) — {len(df_user):,} redova".replace(",", "."))
            st.dataframe(df_user, use_container_width=True)
            if len(df_user):
                csv = df_user.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Preuzmi CSV", data=csv, file_name="rezultat.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Greška u upitu: {e}")

# ---------- Tab 4: Pregled podataka ----------
with tab4:
    st.subheader("Brzi pregled")
    limit = st.slider("Broj redova (LIMIT)", 10, 2000, 200)
    cols = st.multiselect(
        "Kolone",
        [
            "Režim", "Vlasnik", "Serija", "Inv br", "KB", "Tip kola",
            "Voz br", "Stanica", "Status", "Datum", "Vreme", "Roba", "Reon",
            "tara", "NetoTone", "Broj vagona", "Broj kola", "source_file", "DatumVreme"
        ],
        default=["DatumVreme", "Stanica", "Tip kola", "NetoTone", "tara", "source_file"]
    )
    try:
        cols_sql = ", ".join([f'"{c}"' if c not in ("DatumVreme",) else c for c in cols])
        df_preview = run_sql(f'SELECT {cols_sql} FROM "{table_name}" LIMIT {int(limit)}')
        st.dataframe(df_preview, use_container_width=True)
    except Exception as e:
        st.error(f"Greška pri čitanju: {e}")

# ---------- Tab 5: Poslednji unosi ----------
with tab5:
    st.subheader("📌 Poslednji unos za 4098 kola iz Excel tabele")

    if st.button("🔎 Prikaži poslednje unose"):
        tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
        if "stanje" not in tables:
            st.warning("⚠️ Tabela 'stanje' ne postoji, poslednji unosi se ne mogu prikazati.")
        else:
            q_last = f"""
                WITH ranked AS (
                    SELECT 
                        s.SerijaIpodserija,
                        k.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY s.SerijaIpodserija
                            ORDER BY k.DatumVreme DESC
                        ) AS rn
                    FROM stanje s
                    JOIN "kola_sve" k
                      ON CAST(s.SerijaIpodserija AS TEXT) = REPLACE(k.broj_kola_bez_rezima_i_kb, ' ', '')
                )
                SELECT *
                FROM ranked
                WHERE rn = 1
            """
            df_last = run_sql(q_last)
            if df_last.empty:
                st.warning("⚠️ Nema pronađenih podataka.")
            else:
                st.success(f"✅ Pronađeno {len(df_last)} poslednjih unosa.")
                st.dataframe(df_last, use_container_width=True)
# ---------- Tab 6: Pretraga kola ----------
with tab6:
    st.subheader("🔍 Pretraga kola po broju i periodu")

    broj_kola_input = st.text_input("Unesi broj kola (ili deo broja)")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("📅 Od datuma")
    with col2:
        end_date = st.date_input("📅 Do datuma")

    if st.button("🔎 Pretraži"):
        try:
            q_search = f"""
                SELECT *
                FROM "{table_name}"
                WHERE "Broj kola" LIKE '%{broj_kola_input}%'
                  AND "DatumVreme" BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY "DatumVreme" DESC
            """
            df_search = run_sql(q_search)

            if df_search.empty:
                st.warning("⚠️ Nema podataka za zadate kriterijume.")
            else:
                st.success(f"✅ Pronađeno {len(df_search)} redova.")
                st.dataframe(df_search, use_container_width=True)

        except Exception as e:
            st.error(f"Greška u upitu: {e}")

with tab7:
    st.subheader("📊 Broj kola po stanicama")
    try:
        q_sta_count = f"""
            SELECT "Stanica", COUNT(*) AS broj
            FROM "{table_name}"
            GROUP BY "Stanica"
            ORDER BY broj DESC
            LIMIT 50
        """
        df_sta_count = run_sql(q_sta_count)
        st.bar_chart(df_sta_count.set_index("Stanica")["broj"])
        st.dataframe(df_sta_count, use_container_width=True)
    except Exception as e:
        st.error(f"Greška u Tab 7: {e}")

# ---------- Tab 8: Kretanje 4098 kola – TIP 0 ----------
with tab8:
    st.subheader("🚂 Kretanje 4098 kola – TIP 0")
    try:
        q_tip0 = f"""
            SELECT "Broj kola", "DatumVreme", "Stanica", "Status"
            FROM "{table_name}"
            WHERE "Tip kola" = '0'
            ORDER BY "Broj kola", "DatumVreme"
            LIMIT 500
        """
        df_tip0 = run_sql(q_tip0)
        st.dataframe(df_tip0, use_container_width=True)
    except Exception as e:
        st.error(f"Greška u Tab 8: {e}")

# ---------- Tab 9: Kretanje 4098 kola – TIP 1 ----------
with tab9:
    st.subheader("🚂 Kretanje 4098 kola – TIP 1")
    try:
        q_tip1 = f"""
            SELECT "Broj kola", "DatumVreme", "Stanica", "Status"
            FROM "{table_name}"
            WHERE "Tip kola" = '1'
            ORDER BY "Broj kola", "DatumVreme"
            LIMIT 500
        """
        df_tip1 = run_sql(q_tip1)
        st.dataframe(df_tip1, use_container_width=True)
    except Exception as e:
        st.error(f"Greška u Tab 9: {e}")

# ---------- Tab 10: Kola po serijama ----------
with tab10:
    st.subheader("📊 Broj kola po serijama")
    try:
        q_serije = f"""
            SELECT "Serija", COUNT(*) AS broj
            FROM "{table_name}"
            GROUP BY "Serija"
            ORDER BY broj DESC
            LIMIT 50
        """
        df_serije = run_sql(q_serije)
        st.bar_chart(df_serije.set_index("Serija")["broj"])
        st.dataframe(df_serije, use_container_width=True)
    except Exception as e:
        st.error(f"Greška u Tab 10: {e}")
