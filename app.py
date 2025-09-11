import os
import re
import io
import time
import duckdb
import shutil
import pandas as pd
import streamlit as st
import gdown
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import polars as pl

# =========================
# Putevi i folderi
# =========================
# =========================
# Putevi i folderi
# =========================
DB_PATH = "kola_sk.db"
FOLDER_ID = "1q__8P3gY-JMzqD5cpt8avm_7VAY-fHWI"
NOVI_UNOS_FOLDER = "novi_unos"   # lokalni folder gde dodaješ txt fajlove
NOVI_UNOS_FOLDER_ID = "1XQEUt3_TjM_lWahZHoZmlANExIwDwBW1"  # Google Drive ID za "novi unos"

# =========================
# Preuzimanje delova baze (.part fajlovi)
# =========================
folder_url_parts = f"https://drive.google.com/drive/folders/{FOLDER_ID}?usp=sharing"
st.info(f"⬇️ Preuzimam part fajlove iz foldera: {folder_url_parts}")

try:
    gdown.download_folder(
        url=folder_url_parts,
        output=".",   # skida u trenutni folder
        quiet=False,
        use_cookies=False
    )
    st.success("✅ Svi .part fajlovi preuzeti")
except Exception as e:
    st.warning(f"⚠️ Greška pri preuzimanju part fajlova: {e}. Ako su fajlovi već skinuti, nastavljam...")

# =========================
# Preuzimanje TXT fajlova (novi unos)
# =========================
os.makedirs(NOVI_UNOS_FOLDER, exist_ok=True)
folder_url = f"https://drive.google.com/drive/folders/{NOVI_UNOS_FOLDER_ID}?usp=sharing"
st.info(f"⬇️ Preuzimam TXT fajlove iz foldera: {folder_url}")

try:
    gdown.download_folder(
        url=folder_url,
        output=NOVI_UNOS_FOLDER,
        quiet=False,
        use_cookies=False
    )
    st.success("✅ TXT fajlovi preuzeti")
except Exception as e:
    st.warning(f"⚠️ Greška pri preuzimanju TXT fajlova: {e}. Ako su fajlovi već skinuti, nastavljam...")

# =========================
# Fallback download sa pydrive2 (ako gdown ne uspe)
# =========================
def download_folder(folder_id: str, dest: str):
    os.makedirs(dest, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url, output=dest, quiet=False, use_cookies=False)
   

# =========================
# Merge delova u jednu bazu
# =========================
def merge_parts():
    part_files = []
    for f in os.listdir("."):
        if re.match(r"(Copy of )?kola_sk\.db\.part\d+$", f):
            part_files.append(f)

    part_files = sorted(part_files, key=lambda x: int(re.search(r"part(\d+)", x).group(1)))

    if len(part_files) == 48:
        with open(DB_PATH, "wb") as outfile:
            for fname in part_files:
                with open(fname, "rb") as infile:
                    outfile.write(infile.read())
        print(f"✅ Spojeno {len(part_files)} delova → {DB_PATH}")
    else:
        print(f"❌ Nađeno {len(part_files)} fajlova, očekivano 48")
        print("📂 Fajlovi koje sam našao:", part_files)
# =========================
# Učitavanje novih TXT fajlova u tabelu novi_unosi
# =========================
def parse_txt(path) -> pd.DataFrame:
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
                "Broj kola": line[1:11].strip(),
                "Broj kola bez rezima i kb": line[2:11].strip(),
                "source_file": os.path.basename(path),
            })
    return pd.DataFrame(rows)

st.info("☁️ Osvežavam TXT fajlove sa Google Drive (novi unos)...")
folder_url = f"https://drive.google.com/drive/folders/{NOVI_UNOS_FOLDER_ID}"
gdown.download_folder(url=folder_url, output=NOVI_UNOS_FOLDER, quiet=False, use_cookies=False)

txt_files = [os.path.join(NOVI_UNOS_FOLDER, f) for f in os.listdir(NOVI_UNOS_FOLDER) if f.endswith(".txt")]

if txt_files:
    dfs = [parse_txt(f) for f in txt_files]
    df_all = pd.concat(dfs, ignore_index=True)

    con = duckdb.connect(DB_PATH)
    con.register("df_novi", df_all)
    con.execute("""CREATE OR REPLACE TABLE novi_unosi AS SELECT * FROM df_novi""")
    con.unregister("df_novi")
    con.close()

    st.success(f"✅ Učitan {len(df_all)} redova iz {len(txt_files)} TXT fajlova u tabelu 'novi_unosi'")
else:
    st.warning("⚠️ Nema pronađenih TXT fajlova u folderu 'novi_unos'.")


# =========================
# Spajanje baze i novih fajlova
# =========================
if os.path.exists(DB_PATH):
    con = duckdb.connect(DB_PATH)

    # 1️⃣ Učitaj TXT fajlove iz "novi_unos"
    dfs = []
    folder_path = "novi_unos"
    if os.path.exists(folder_path):
        for fname in os.listdir(folder_path):
            if fname.endswith(".txt"):
                fpath = os.path.join(folder_path, fname)
                df_txt = parse_txt(fpath)
                dfs.append(df_txt)

    if dfs:
        dfs = [parse_txt(f) for f in txt_files]
        df_all = pd.concat(dfs, ignore_index=True)

        # Ako ima novih fajlova → kreiraj/menjaj tabelu
        con = duckdb.connect(DB_PATH)
        con.register("df_novi", df_all)
        con.execute("""CREATE OR REPLACE TABLE novi_unosi AS SELECT * FROM df_novi""")
        con.unregister("df_novi")
        con.close()
    else:
        # Ako nema fajlova, napravi praznu tabelu sa strukturom
        con.execute("""
            CREATE OR REPLACE TABLE novi_unosi AS 
            SELECT * FROM kola WHERE FALSE
        """)
        st.warning("⚠️ Nema TXT fajlova u 'novi_unos' folderu → kreirana prazna tabela")

    # 2️⃣ Tek sada kreiraj VIEW
    con.execute("""
        CREATE OR REPLACE VIEW kola_sve AS
        SELECT * FROM kola
        UNION ALL
        SELECT * FROM novi_unosi
    """)
    con.close()

    st.success("✅ View 'kola_sve' je spreman za upotrebu")

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

def create_or_replace_table_from_df(db_file: str, table_name: str, df: pd.DataFrame):
    con = duckdb.connect(db_file)
    try:
        con.register("df_tmp", df)
        con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM df_tmp')
        con.unregister("df_tmp")
    finally:
        con.close()

# =========================
# Preuzimanje baze ako ne postoji
# =========================
if not os.path.exists(DB_PATH):
    st.info("☁️ Preuzimam delove baze sa Google Drive...")
    try:
        gdown.download_folder(id=FOLDER_ID, quiet=False, use_cookies=False)
    except Exception as e:
        st.warning(f"⚠️ gdown nije uspeo ({e}), prelazim na pydrive2...")
        try:
            download_with_pydrive2(FOLDER_ID)
        except Exception as ee:
            st.error(f"❌ Ni pydrive2 nije uspeo: {ee}")
    merge_parts()

# =========================
# Inicijalizacija baze
# =========================
if os.path.exists(DB_PATH):
    st.success(f"✅ Baza {DB_PATH} je pronađena")

    # učitaj TXT fajlove iz foldera novi_unos
    if os.path.exists(NOVI_UNOS_FOLDER):
        all_txt = []
        for fname in os.listdir(NOVI_UNOS_FOLDER):
            if fname.lower().endswith(".txt"):
                st.write(f"📥 Učitavam {fname}")
                df_txt = parse_txt(os.path.join(NOVI_UNOS_FOLDER, fname))
                all_txt.append(df_txt)

        if all_txt:
            df_all = pl.concat(all_txt).to_pandas()
            create_or_replace_table_from_df(DB_PATH, "novi_unosi", df_all)
            st.success(f"✅ Učitano {len(df_all)} redova iz TXT fajlova u 'novi_unosi'")
        else:
            st.warning("⚠️ Nema TXT fajlova u 'novi_unos' folderu")

    # kreiraj pogled kola_sve
    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE OR REPLACE VIEW kola_sve AS
        SELECT * FROM kola
        UNION ALL
        SELECT * FROM novi_unosi
    """)
    con.close()
    st.success("✅ 'kola_sve' je spreman (kola ∪ novi_unosi)")

    # test
    try:
        df_test = run_sql(DB_PATH, "SELECT COUNT(*) AS broj_redova FROM kola_sve")
        st.write("📊 Ukupan broj redova u kola_sve:", df_test.iloc[0, 0])
    except Exception as e:
        st.error(f"❌ Problem prilikom testiranja: {e}")
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
try:
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
        JOIN "{table_name}" k
          ON CAST(s.SerijaIpodserija AS TEXT) = REPLACE(k.broj_kola_bez_rezima_i_kb, ' ', '')
    )
    SELECT *
    FROM ranked
    WHERE rn = 1
    """

    df_last = run_sql(DB_PATH, q_last)

    if df_last.empty:
        st.warning("⚠️ Nema pronađenih podataka.")
    else:
        st.success(f"✅ Pronađeno {len(df_last)} poslednjih unosa.")
        st.dataframe(df_last, use_container_width=True)

except Exception as e:
    st.error(f"Greška u upitu: {e}")

# =========================
# Glavni naslov i tabovi
# =========================

# Uvek koristimo jednu glavnu tabelu
table_name = "kola_sve"
db_path = DB_PATH

st.title("🚃 Teretna kola SK — kontrolna tabla")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Pregled", "📈 Izveštaji", "🔎 SQL upiti", "🔬 Pregled podataka", "📌 Poslednji unosi", "🔍 Pretraga kola", "📊 Kola po stanicima", "🚂 Kretanje 4098 kola–TIP 0", "🚂 Kretanje 4098 kola–TIP 1", "📊 Kola po serijama"])

# ---------- Tab 1: Pregled ----------
with tab1:
    col_a, col_b, col_c, col_d = st.columns(4)
    try:
        df_cnt = run_sql(DB_PATH, f'SELECT COUNT(*) AS broj_redova FROM "{table_name}"')
        col_a.metric("Ukupan broj redova", f"{int(df_cnt['broj_redova'][0]):,}".replace(",", "."))

        df_files = run_sql(DB_PATH, f'SELECT COUNT(DISTINCT source_file) AS fajlova FROM "{table_name}"')
        col_b.metric("Učitanih fajlova", int(df_files["fajlova"][0]))

        df_range = run_sql(
            DB_PATH,
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
            DB_PATH,
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
    df_month = run_sql(DB_PATH, q_month)
    st.line_chart(df_month.set_index("mesec")["ukupno_tona"])

    st.subheader("Top 20 stanica po broju vagona")
    q_sta = f"""
        SELECT "Stanica", COUNT(*) AS broj
        FROM "{table_name}"
        GROUP BY "Stanica"
        ORDER BY broj DESC
        LIMIT 20
    """
    df_sta = run_sql(DB_PATH, q_sta)
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
        df_tip = run_sql(DB_PATH, q_tip)
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
        df_tara = run_sql(DB_PATH, q_tara)
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
            df_user = run_sql(DB_PATH, user_sql)
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
        df_preview = run_sql(DB_PATH, f'SELECT {cols_sql} FROM "{table_name}" LIMIT {int(limit)}')
        st.dataframe(df_preview, use_container_width=True)
    except Exception as e:
        st.error(f"Greška pri čitanju: {e}")

# ---------- Tab 5: Poslednji unosi ----------
with tab5:
    st.subheader("📌 Poslednji unos za 4098 kola iz Excel tabele")

    if st.button("🔎 Prikaži poslednje unose"):
        try:
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

            df_last = run_sql(DB_PATH, q_last)

            if df_last.empty:
                st.warning("⚠️ Nema pronađenih podataka.")
            else:
                st.success(f"✅ Pronađeno {len(df_last)} poslednjih unosa.")
                st.dataframe(df_last, use_container_width=True)

        except Exception as e:
            st.error(f"Greška u upitu: {e}")

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
            df_search = run_sql(DB_PATH, q_search)

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
        df_sta_count = run_sql(DB_PATH, q_sta_count)
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
        df_tip0 = run_sql(DB_PATH, q_tip0)
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
        df_tip1 = run_sql(DB_PATH, q_tip1)
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
        df_serije = run_sql(DB_PATH, q_serije)
        st.bar_chart(df_serije.set_index("Serija")["broj"])
        st.dataframe(df_serije, use_container_width=True)
    except Exception as e:
        st.error(f"Greška u Tab 10: {e}")
