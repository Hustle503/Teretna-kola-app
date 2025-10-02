import os
import time
import duckdb
import glob
import pandas as pd
import streamlit as st
import io
import polars as pl
import json
from datetime import date
from huggingface_hub import hf_hub_download, Repository, HfApi

# app.py — Zbirni, memorijski-štedljivi Streamlit app za "Teretna kola SK"
# Kombinuje funkcionalnosti iz oba fajla i koristi DuckDB za minimalno korišćenje RAM-a.
# Izvori: originalni skripti (parsiranje, HF/parquet, admin UI). 
# Potreban: python paketi: streamlit, duckdb, polars, pandas, pyarrow, huggingface_hub (opciono)

import os
import glob
import io
import json
import time
from datetime import datetime, date, timedelta

import streamlit as st
import duckdb
import polars as pl
import pandas as pd

# OPTIONAL: if you want HF parquet features
try:
    from huggingface_hub import hf_hub_download, HfApi
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# ---------- Konfiguracija ----------
st.set_page_config(layout="wide", page_title="🚂 Teretna kola SK - Unified")
DEFAULT_FOLDER = os.environ.get("TERETNA_FOLDER", r"C:\Teretna kola")
DB_FILE = os.environ.get("TERETNA_DB", os.path.join(DEFAULT_FOLDER, "kola_sk.db"))
STATE_FILE = os.path.join(DEFAULT_FOLDER, "loaded_files.json")
TABLE_NAME = "kola"

# Admin secrets (možeš ih staviti u streamlit secrets)
ADMIN_PASS = st.secrets.get("ADMIN_PASS", "tajna123")
HF_TOKEN = st.secrets.get("HF_TOKEN", None)
HF_REPO = st.secrets.get("HF_REPO", None)

# ---------- Pomoćne funkcije za memoriju ----------
# - Koristimo jednu @st.cache_resource DuckDB konekciju (jedan objekat po session)
# - Izbegavamo velike pandas.DataFrame trajne kopije; radimo većinu batch operacija direktno u DuckDB.
@st.cache_resource
def get_duckdb_connection(db_path=DB_FILE, read_only=False):
    # Konekcija ka disk-based DuckDB (manje RAM nego držati sve u pandas)
    return duckdb.connect(database=db_path, read_only=read_only)

# Brzi SQL helper (kešira rezultate kad je potrebno)
@st.cache_data
def run_sql(sql: str):
    con = get_duckdb_connection()
    return con.execute(sql).fetchdf()

# ---------- Učitaj mapu stanica (globalno, jednom) ----------
# Ako postoji stanice.xlsx, učitaj u dict radi mapiranja imena stanica.
STANICE_MAP = {}
try:
    if os.path.exists("stanice.xlsx"):
        tmp = pd.read_excel("stanice.xlsx")
        tmp["sifra"] = tmp["sifra"].astype(str).str.strip()
        STANICE_MAP = dict(zip(tmp["sifra"], tmp["naziv"]))
except Exception as e:
    st.warning(f"Ne mogu da učitam stanice.xlsx: {e}")

def add_station_names_pandas(df: pd.DataFrame) -> pd.DataFrame:
    # Koristimo kratkotrajnu pandas transformaciju — brišemo brzo posle upisa u DB
    if not STANICE_MAP:
        return df
    if "Stanica" in df.columns:
        df["Stanica"] = df["Stanica"].astype(str).str.strip()
        df.insert(df.columns.get_loc("Stanica")+1, "Naziv st.", df["Stanica"].map(STANICE_MAP))
    # mapiranje za Otp/Up stanice (ako postoje)
    for col, flag_col, name_col in [
        ("Otp st", "Otp. država", "Naziv otp st."),
        ("Up st", "Uputna država", "Naziv up st.")
    ]:
        if col in df.columns and flag_col in df.columns:
            mask = df[flag_col].astype(str).str.strip() == "72"
            df[name_col] = None
            df.loc[mask, name_col] = df.loc[mask, col].astype(str).str.strip().map(STANICE_MAP)
    return df

# ---------- Parse TXT (polars) ----------
def parse_txt(path: str) -> pl.DataFrame:
    """Parsira fixed-width txt u polars DataFrame — low-memory (streaming čitanje linija)."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # kraće polja (prilagodi po potrebi)
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
                "source_file": os.path.basename(path),
            })
    df = pl.DataFrame(rows)
    # kombinuje Datum+Vreme (tolerantno)
    df = df.with_columns([
        (pl.col("Datum") + " " + pl.col("Vreme")).str.strptime(pl.Datetime, "%Y%m%d %H%M", strict=False).alias("DatumVreme")
    ])
    return df

# ---------- State fajl (koji txt-ove smo već procesirali) ----------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()

def save_state(processed_files):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(list(processed_files), f, indent=2)

# ---------- Inicijalizacija baze (batch iz foldera) ----------
def init_database(folder: str, table_name: str = TABLE_NAME):
    files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    if not files:
        st.warning("⚠️ Nema txt fajlova u folderu.")
        return

    con = get_duckdb_connection()
    pbar = st.progress(0)
    status = st.empty()

    # Umesto držanja svih parsed DF-ova u RAM-u,
    # parsiramo po fajlu i za svaki fajl napišemo privremeni parquet,
    # pa ne držimo sve u Pythonu. DuckDB može da kreira table iz parquet fajlova.
    tmp_parquets = []
    for i, fpath in enumerate(files, start=1):
        status.text(f"Čitam {os.path.basename(fpath)} ({i}/{len(files)})")
        df_part = parse_txt(fpath)
        # minimalni post-proc — broj vagona kao int ako postoji
        try:
            df_part = df_part.with_columns([
                pl.col("DatumVreme"),
            ])
        except Exception:
            pass
        # upiši temp parquet (disk je jeftiniji od RAM-a)
        tmp_path = os.path.join("/tmp", f"part_{i}_{os.path.basename(fpath)}.parquet")
        df_part.write_parquet(tmp_path)
        tmp_parquets.append(tmp_path)
        pbar.progress(i/len(files))

    # Napravi tabelu iz svih parquetova u jednom koraku (DuckDB čita parquet direktno)
    # Ovo izbegava veliki presip u Python RAM-u.
    parquet_list = "','".join(tmp_parquets)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    # DuckDB direct read
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet(['{\"','\".join(tmp_parquets)}'])")
    # Dodaj ID_rb kolonu
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT *, ROW_NUMBER() OVER (ORDER BY DatumVreme ASC NULLS LAST) AS ID_rb
        FROM {table_name}
    """)
    # cleanup tmp parquet files (ako želiš)
    for p in tmp_parquets:
        try:
            os.remove(p)
        except:
            pass

    save_state(set(files))
    pbar.empty()
    status.text("✅ Učitavanje završeno")
    st.success(f"✅ Inicijalno učitano {len(files)} fajlova u tabelu '{table_name}'")

# ---------- Update baze (samo novi fajlovi) ----------
def update_database(folder: str, table_name: str = TABLE_NAME):
    processed = load_state()
    files = set(glob.glob(os.path.join(folder, "*.txt")))
    new_files = sorted(files - processed)
    if not new_files:
        st.info("ℹ️ Nema novih fajlova za unos.")
        return

    con = get_duckdb_connection()
    for f in new_files:
        df_new = parse_txt(f)
        # write temp parquet and insert using DuckDB (ne držimo veliki DF u RAM)
        tmp = os.path.join("/tmp", f"up_{os.path.basename(f)}.parquet")
        df_new.write_parquet(tmp)
        # if table exists -> insert from parquet; else create
        tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
        if table_name in tables:
            con.execute(f"INSERT INTO {table_name} SELECT * FROM read_parquet('{tmp}')")
        else:
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{tmp}')")
        try:
            os.remove(tmp)
        except:
            pass
        processed.add(f)
        st.write(f"➕ Ubačen: {os.path.basename(f)}")

    save_state(processed)
    # refresh ID_rb
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS
        SELECT *, ROW_NUMBER() OVER (ORDER BY DatumVreme ASC NULLS LAST) AS ID_rb
        FROM {table_name}
    """)
    st.success("✅ Update završen")

# ---------- Dodavanje pojedinačnog fajla putem Streamlit upload ----------
def add_uploaded_file(uploaded_file):
    if uploaded_file is None:
        st.warning("⚠️ Niste izabrali fajl.")
        return
    tmp_path = os.path.join("/tmp", uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ext = os.path.splitext(tmp_path)[1].lower()
    if ext == ".txt":
        df_new = parse_txt(tmp_path)
    elif ext in (".xls", ".xlsx"):
        # excel -> pandas -> polars minimally
        pdf = pd.read_excel(tmp_path)
        df_new = pl.from_pandas(pdf)
    else:
        st.error("Nepoznat tip fajla.")
        return

    # write parquet and insert via DuckDB
    pq = tmp_path + ".parquet"
    df_new.write_parquet(pq)
    con = get_duckdb_connection()
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    if TABLE_NAME in tables:
        con.execute(f"INSERT INTO {TABLE_NAME} SELECT * FROM read_parquet('{pq}')")
    else:
        con.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM read_parquet('{pq}')")
    # refresh ID_rb minimal
    con.execute(f"""
        CREATE OR REPLACE TABLE {TABLE_NAME} AS
        SELECT *, ROW_NUMBER() OVER (ORDER BY id ASC NULLS LAST) AS ID_rb
        FROM {TABLE_NAME}
    """)
    try:
        os.remove(pq)
        os.remove(tmp_path)
    except:
        pass
    st.success(f"✅ Fajl '{uploaded_file.name}' dodat u bazu.")

# ---------- OPTIONAL: HF parquet helper (smanjuje lokalnu memoriju čitanjem parquet fajlova direktno iz HF) ----------
def hf_get_parquet_path(filename: str) -> str:
    if not HF_AVAILABLE or not HF_TOKEN or not HF_REPO:
        raise RuntimeError("Hugging Face nije konfigurisan (HF_TOKEN/HF_REPO).")
    return hf_hub_download(repo_id=HF_REPO, filename=filename, repo_type="dataset", token=HF_TOKEN)

# ---------- UI: Sidebar / Login ----------
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

st.sidebar.title("⚙️ Podešavanja")
if not st.session_state.admin_logged_in:
    password = st.sidebar.text_input("🔑 Unesi lozinku:", type="password")
    if st.sidebar.button("🔓 Otključaj"):
        if password == ADMIN_PASS:
            st.session_state.admin_logged_in = True
            st.sidebar.success("✅ Ulogovan")
        else:
            st.sidebar.error("❌ Pogrešna lozinka")
else:
    if st.sidebar.button("🚪 Odjavi se"):
        st.session_state.admin_logged_in = False
        st.sidebar.info("🔒 Odjavljen")

# ---------- Admin area ----------
if st.session_state.admin_logged_in:
    tabs = st.tabs(["📂 Init/Update", "🔍 Duplikati", "📄 Upload", "📊 Učitani fajlovi"])
    # Init / Update
    with tabs[0]:
        st.subheader("📂 Inicijalizacija / Update baze")
        folder = st.text_input("Folder sa TXT fajlovima", value=DEFAULT_FOLDER)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Inicijalizuj bazu"):
                init_database(folder)
        with col2:
            if st.button("🔄 Update baze iz foldera"):
                update_database(folder)

        st.divider()
        st.subheader("➕ Dodaj pojedinačni fajl")
        upl = st.file_uploader("Izaberi TXT/XLSX fajl", type=["txt","xlsx","xls"])
        if st.button("📥 Dodaj fajl"):
            if upl:
                add_uploaded_file(upl)
            else:
                st.warning("Niste izabrali fajl.")

    # Duplikati (SQL bazirano)
    with tabs[1]:
        st.subheader("🔍 Duplikati u tabeli kola")
        godina = st.text_input("Godina (YYYY)", max_chars=4)
        mesec = st.text_input("Mesec (MM)", max_chars=2)
        def where_clause(g, m):
            if not g:
                return ""
            clause = f"WHERE EXTRACT(YEAR FROM DatumVreme)={g}"
            if m:
                clause += f" AND EXTRACT(MONTH FROM DatumVreme)={m}"
            return clause
        def dupes_sql(g, m):
            w = where_clause(g, m)
            return f"""
                WITH dupl AS (
                  SELECT *, ROW_NUMBER() OVER (
                       PARTITION BY "Režim","Vlasnik","Serija","Inv br","KB","Tip kola","Voz br",
                                    "Stanica","Status","Roba","Rid","UN broj","Reon",
                                    "tara","NetoTone","dužina vagona","broj osovina",
                                    "Otp. država","Otp st","Uputna država","Up st","Broj kola",
                                    "Redni broj kola", "Datum", "Vreme"
                       ORDER BY DatumVreme
                  ) AS rn, COUNT(*) OVER (PARTITION BY "Režim","Vlasnik","Serija","Inv br","KB","Tip kola","Voz br",
                                    "Stanica","Status","Roba","Rid","UN broj","Reon",
                                    "tara","NetoTone","dužina vagona","broj osovina",
                                    "Otp. država","Otp st","Uputna država","Up st","Broj kola",
                                    "Redni broj kola", "Datum", "Vreme") AS cnt
                  FROM "{TABLE_NAME}"
                  {w}
                )
                SELECT * FROM dupl WHERE cnt > 1 ORDER BY DatumVreme
            """
        if st.button("🔎 Proveri duplikate"):
            if not godina:
                st.warning("Unesi godinu.")
            else:
                df_dupes = run_sql(dupes_sql(godina, mesec))
                if df_dupes.empty:
                    st.success("✅ Duplikata nema")
                else:
                    st.warning(f"⚠️ Pronađeno {len(df_dupes)} redova sa duplikatima")
                    st.dataframe(df_dupes, use_container_width=True)
    # Upload fajlova / pregled
    with tabs[2]:
        st.subheader("📄 Upload fajlova (TXT/XLSX)")
        upf = st.file_uploader("Izaberi fajl za upload", type=["txt","xlsx","xls"])
        if st.button("⬆️ Upload fajla"):
            if upf:
                add_uploaded_file(upf)
            else:
                st.warning("Niste izabrali fajl.")
    with tabs[3]:
        st.subheader("📊 Učitani fajlovi (top 20)")
        try:
            df_by_file = run_sql(f"SELECT source_file, COUNT(*) AS broj FROM \"{TABLE_NAME}\" GROUP BY source_file ORDER BY broj DESC LIMIT 20")
            st.dataframe(df_by_file, use_container_width=True)
        except Exception as e:
            st.warning(f"Ne mogu da pročitam bazu: {e}")

# ---------- Glavni tabovi (pregled, poslednje stanje, sql, itd) ----------
tab_buttons = ["📊 Pregled", "📌 Poslednje stanje kola", "🔎 SQL upiti", "🔬 Pregled podataka"]
selected = st.sidebar.radio("Izaberi prikaz:", tab_buttons, index=0)

if selected == "📊 Pregled":
    st.subheader("📊 Pregled (LIMIT 50)")
    try:
        df_preview = run_sql(f'SELECT * FROM "{TABLE_NAME}" ORDER BY DatumVreme DESC LIMIT 50')
        st.dataframe(df_preview, use_container_width=True)
    except Exception as e:
        st.error(f"Greška: {e}")

if selected == "📌 Poslednje stanje kola":
    st.subheader("📌 Poslednje stanje kola")
    try:
        # Primer SQL za poslednje stanje (spoji sa tabelom stanje ako postoji)
        q = """
        WITH kola_clean AS (
          SELECT *, TRY_CAST(SUBSTR("Broj kola", 3) AS BIGINT) AS broj_clean FROM kola
        ), poslednje AS (
          SELECT s."Broj kola" AS broj_stanje, k.*, ROW_NUMBER() OVER (PARTITION BY s."Broj kola" ORDER BY k."DatumVreme" DESC) AS rn
          FROM "stanje" s
          LEFT JOIN kola_clean k ON TRY_CAST(s."Broj kola" AS BIGINT) = k.broj_clean
        )
        SELECT * FROM poslednje WHERE rn = 1
        """
        df_last = run_sql(q)
        st.dataframe(df_last, use_container_width=True)
        # export
        buf = io.BytesIO()
        df_last.to_excel(buf, index=False)
        st.download_button("⬇️ Preuzmi Excel", data=buf.getvalue(), file_name="poslednje_stanje.xlsx")
    except Exception as e:
        st.error(f"Greška: {e}")

if selected == "🔎 SQL upiti":
    st.subheader("🔎 Piši svoj SQL")
    default_sql = f'SELECT * FROM "{TABLE_NAME}" LIMIT 100'
    user_sql = st.text_area("SQL:", value=default_sql, height=160)
    if st.button("▶️ Izvrši upit"):
        t0 = time.time()
        try:
            df_res = run_sql(user_sql)
            elapsed = time.time() - t0
            st.success(f"OK ({elapsed:.2f}s) — {len(df_res)} redova")
            st.dataframe(df_res, use_container_width=True)
            if len(df_res):
                st.download_button("⬇️ Preuzmi CSV", data=df_res.to_csv(index=False).encode("utf-8"), file_name="rezultat.csv")
        except Exception as e:
            st.error(f"Greška u upitu: {e}")

if selected == "🔬 Pregled podataka":
    st.subheader("🔬 Pregled podataka")
    limit = st.slider("Broj redova", 10, 2000, 200)
    cols = st.multiselect("Kolone", ["DatumVreme","Stanica","Tip kola","Broj kola","NetoTone","tara"], default=["DatumVreme","Stanica","Broj kola"])
    try:
        col_sql = ", ".join([f'"{c}"' if not c.isidentifier() else c for c in cols]) if cols else "*"
        df_show = run_sql(f'SELECT {col_sql} FROM "{TABLE_NAME}" LIMIT {int(limit)}')
        st.dataframe(df_show, use_container_width=True)
    except Exception as e:
        st.error(f"Greška: {e}")

# ---------- Tab 5: Kola u inostranstvu ----------
stanice_df = pd.read_excel("stanice.xlsx")  # kolone: sifra, naziv
stanice_df["sifra"] = stanice_df["sifra"].astype(str).str.strip()
stanice_map = dict(zip(stanice_df["sifra"], stanice_df["naziv"]))


# 📌 Kola u inostranstvu
if selected_tab == "📌 Kola u inostranstvu":
    st.subheader("📌 Kola u inostranstvu")    

    # 🔹 Izbor tipa prikaza
    prikaz_tip = st.radio(
        "🔎 Izaberite prikaz:",
        ["Samo poslednje stanje", "Sva kretanja (istorija)"],
        index=0,
        horizontal=True
    )

    # ------------------------
    # Pomoćne funkcije
    # ------------------------
    def add_station_names(df):
        if "Stanica" in df.columns:
            df["Stanica"] = df["Stanica"].astype(str).str.strip()
            df.insert(
                df.columns.get_loc("Stanica") + 1,
                "Naziv st.",
                df["Stanica"].map(stanice_map)
            )
        return df

    def format_numeric(df):
        for col in ["tara", "NetoTone", "dužina vagona"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].apply(
                    lambda x: x / 10 if pd.notnull(x) and x > 100 else x
                )
                df[col] = df[col].apply(
                    lambda x: f"{x:.1f}".replace(".", ",") if pd.notnull(x) else None
                )
        return df

    # ------------------------
    # SQL upiti
    # ------------------------
    if prikaz_tip == "Samo poslednje stanje":
        q_foreign = """
        WITH poslednje_stanje AS (
            SELECT 
                s."Broj kola" AS broj_stanje,
                k."Broj kola" AS broj_kola_raw,
                k.*,
                ROW_NUMBER() OVER (
                    PARTITION BY s."Broj kola"
                    ORDER BY k."DatumVreme" DESC
                ) AS rn
            FROM stanje s
            LEFT JOIN (
                SELECT 
                    TRY_CAST(SUBSTR("Broj kola", 3, LENGTH("Broj kola") - 3) AS BIGINT) AS broj_clean,
                    *
                FROM kola
            ) k
                ON TRY_CAST(s."Broj kola" AS BIGINT) = k.broj_clean
        )
        SELECT *
        FROM poslednje_stanje
        WHERE rn = 1
          AND status IN (11, 14)
        ORDER BY broj_stanje;
        """
        try:
            df_foreign = run_sql(q_foreign)
            df_foreign = add_station_names(df_foreign)
            df_foreign = format_numeric(df_foreign)

            st.success(f"🌍 Pronađeno {len(df_foreign)} kola u inostranstvu (poslednje stanje).")
            st.dataframe(df_foreign, use_container_width=True)

            # Export
            if st.button("📥 Export u Excel", key="export_last_state"):
                file_name = "kola_u_inostranstvu_poslednje_stanje.xlsx"
                df_foreign.to_excel(file_name, index=False)
                st.success(f"✅ Podaci eksportovani u {file_name}")

        except Exception as e:
            st.error(f"❌ Greška pri učitavanju kola u inostranstvu: {e}")

    else:  # 🔹 Sva kretanja (istorija)
        st.markdown("<h4 style='text-align: center;'>🔎 Opcioni filteri</h4>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            broj_kola_input = st.text_input("🚃 Izaberi broj kola (opciono)", "")

        with col2:
            date_range = st.date_input("📅 Izaberi vremenski period (opciono)", [])
            start_date, end_date = None, None
            if isinstance(date_range, list) and len(date_range) == 2:
                start_date, end_date = date_range

        with col3:
            try:
                stanice = run_sql('SELECT DISTINCT "Stanica" FROM kola WHERE "Stanica" IS NOT NULL ORDER BY "Stanica"')
                stanice_list = stanice["Stanica"].dropna().astype(str).tolist()
            except:
                stanice_list = []
            granicni_prelaz = st.selectbox("🌍 Izaberi granični prelaz (opciono)", [""] + stanice_list)

        q_foreign = """
        SELECT 
            s."Broj kola" AS broj_stanje,
            k."Broj kola" AS broj_kola_raw,
            k.*,
            s.status AS status_stanje
        FROM stanje s
        LEFT JOIN (
            SELECT 
                TRY_CAST(SUBSTR("Broj kola", 3, LENGTH("Broj kola") - 3) AS BIGINT) AS broj_clean,
                *
            FROM kola
        ) k
            ON TRY_CAST(s."Broj kola" AS BIGINT) = k.broj_clean
        WHERE k.status IN (11, 14, 21, 24)
        ORDER BY broj_stanje, k."DatumVreme" DESC
        """
        try:
            df_foreign = run_sql(q_foreign)

            # --- Primena filtera ---
            if broj_kola_input:
                brojevi = [b.strip() for b in broj_kola_input.split(",") if b.strip()]
                df_foreign = df_foreign[df_foreign["Broj kola"].isin(brojevi)]

            if start_date and end_date:
                df_foreign = df_foreign[
                    (df_foreign["DatumVreme"] >= pd.to_datetime(start_date)) &
                    (df_foreign["DatumVreme"] <= pd.to_datetime(end_date))
                ]

            if granicni_prelaz:
                df_foreign = df_foreign[df_foreign["Stanica"] == granicni_prelaz]

            df_foreign = add_station_names(df_foreign)
            df_foreign = format_numeric(df_foreign)

            st.success(f"🌍 Pronađeno {len(df_foreign)} redova (istorija kretanja).")
            st.dataframe(df_foreign, use_container_width=True)

            # Export
            if st.button("📥 Export u Excel", key="export_history"):
                file_name = "kola_u_inostranstvu_istorija.xlsx"
                df_foreign.to_excel(file_name, index=False)
                st.success(f"✅ Podaci eksportovani u {file_name}")

        except Exception as e:
            st.error(f"❌ Greška pri učitavanju istorije kola u inostranstvu: {e}")
       # --- Dugme za prikaz zadržavanja ---
        if st.button("📊 Prikaži zadržavanje kola u inostranstvu") and not df_foreign.empty:
            try:
                # Osiguraj da su datumi datetime
                df_foreign["DatumVreme"] = pd.to_datetime(df_foreign["DatumVreme"], errors="coerce")

                # Sortiraj po kolima i datumu
                df_foreign = df_foreign.sort_values(["Broj kola", "DatumVreme"])

                # Lista rezultata
                retention_records = []

                # Grupisanje po kolima
                for broj_kola, grupa in df_foreign.groupby("Broj kola"):
                    grupa = grupa.sort_values("DatumVreme")
                    start_time = None

                    for _, row in grupa.iterrows():
                        # Početak kada kola izlaze iz Srbije (Otp. država = 72)
                        if str(row.get("Otp. država")) == "72":
                            start_time = row["DatumVreme"]

                        # Kraj kada kola ulaze u Srbiju (Uputna država = 72)
                        elif start_time is not None and str(row.get("Uputna država")) == "72":
                            end_time = row["DatumVreme"]
                            retention = (end_time - start_time).total_seconds() / 3600  # sati
                            retention_records.append({
                                "Broj kola": broj_kola,
                                "Datum izlaska": start_time,
                                "Datum ulaska": end_time,
                                "Zadržavanje [h]": round(retention, 2)
                            })
                            start_time = None  # reset posle para

                # Kreiraj DataFrame rezultata
                df_retention = pd.DataFrame(retention_records)

                if not df_retention.empty:
                    # Dodaj red sa prosekom
                    avg_retention = df_retention["Zadržavanje [h]"].mean()
                    df_retention.loc[len(df_retention)] = {
                        "Broj kola": "📊 PROSEK",
                        "Datum izlaska": None,
                        "Datum ulaska": None,
                        "Zadržavanje [h]": round(avg_retention, 2)
                    }

                    st.success(f"✅ Pronađeno {len(df_retention)-1} parova ulaska/izlaska.")
                    st.dataframe(df_retention, use_container_width=True)

                    # Dodaj export u Excel
                    excel_file = "zadrzavanje_inostranstvo.xlsx"
                    df_retention.to_excel(excel_file, index=False)
                    with open(excel_file, "rb") as f:
                        st.download_button("⬇️ Preuzmi Excel (zadržavanje)", f, file_name=excel_file)
                else:
                    st.info("ℹ️ Nema pronađenih parova za računanje zadržavanja.")

            except Exception as e:
                st.error(f"❌ Greška pri izračunavanju zadržavanja: {e}")
 

# ---------- Tab 6: Pretraga kola ----------
if selected_tab == "🔍 Pretraga kola":
    st.subheader("🔍 Pretraga kola po broju i periodu")

    # Unos broja kola (ili deo broja)
    broj_kola_input = st.text_input("🚋 Unesi broj kola (ili deo broja)")

    # Odabir perioda
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("📅 Od datuma")
    with col2:
        end_date = st.date_input("📅 Do datuma")

    # Dugme za pretragu
    if st.button("🔎 Pretraži"):
        try:
            q_search = f"""
                SELECT *
                FROM (
                    SELECT 
                        "Broj kola" AS broj_kola_raw,
                        TRY_CAST(SUBSTR("Broj kola", 3, LENGTH("Broj kola") - 3) AS BIGINT) AS broj_clean,
                        *
                    FROM "{TABLE_NAME}"
                )
                WHERE broj_clean::VARCHAR LIKE '%{broj_kola_input}%'
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
            st.error(f"❌ Greška u upitu: {e}")
# ---------- Tab 7: Kola po stanicima ----------
if selected_tab == "📊 Kola po stanicima":   
    st.subheader("📊 Kola po stanicima")

    try:
        q = """
        WITH kola_clean AS (
            SELECT 
                *,
                TRY_CAST(SUBSTR("Broj kola", 3, LENGTH("Broj kola") - 3) AS BIGINT) AS broj_clean
            FROM kola
        ),
        poslednji AS (
            SELECT 
                k.Serija,
                k.Stanica,
                st.Naziv AS NazivStanice,
                s.TIP,
                ROW_NUMBER() OVER (
                    PARTITION BY k.broj_clean
                    ORDER BY k.DatumVreme DESC
                ) AS rn
            FROM kola_clean k
            JOIN stanje s 
              ON k.broj_clean = TRY_CAST(s."Broj kola" AS BIGINT)
            LEFT JOIN stanice st
              ON TRIM(CAST(k.Stanica AS VARCHAR)) = TRIM(CAST(st.Sifra AS VARCHAR))
        )
        SELECT *
        FROM poslednji
        WHERE rn = 1
        """
        df_last = run_sql(q)

        # Pivot tabela po stanici
        df_pivot = (
            df_last.groupby(["Stanica", "NazivStanice", "TIP"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        if 0 not in df_pivot.columns:
            df_pivot[0] = 0
        if 1 not in df_pivot.columns:
            df_pivot[1] = 0

        df_pivot = df_pivot.rename(columns={0: "tip0", 1: "tip1"})
        df_pivot["Ukupno"] = df_pivot["tip0"] + df_pivot["tip1"]

        # Dodaj red Σ
        total_row = {
            "Stanica": "Σ",
            "NazivStanice": "Ukupno",
            "tip0": df_pivot["tip0"].sum(),
            "tip1": df_pivot["tip1"].sum(),
            "Ukupno": df_pivot["Ukupno"].sum()
        }
        df_pivot = pd.concat([df_pivot, pd.DataFrame([total_row])], ignore_index=True)

        # Dva dela ekrana
        left, right = st.columns([1, 1])

        with left:
            st.markdown("### 📋 Ukupan broj kola (po stanicama)")
            st.dataframe(df_pivot, use_container_width=True)

        with right:
            st.markdown("### 📍 Klikni / izaberi stanicu")

            station_list = df_pivot[df_pivot["Stanica"] != "Σ"]["NazivStanice"].tolist()
            selected_station = st.selectbox("", ["Nijedna"] + station_list)

            if selected_station != "Nijedna":
                st.markdown(f"### 🔎 Detalji za stanicu: **{selected_station}**")

                stanica_id = df_pivot.loc[df_pivot["NazivStanice"] == selected_station, "Stanica"].iloc[0]
                df_detail = (
                    df_last[df_last["Stanica"] == stanica_id]
                    .groupby(["Serija", "TIP"])
                    .size()
                    .unstack(fill_value=0)
                    .reset_index()
                )

                if 0 not in df_detail.columns:
                    df_detail[0] = 0
                if 1 not in df_detail.columns:
                    df_detail[1] = 0

                df_detail = df_detail.rename(columns={0: "tip0", 1: "tip1"})
                df_detail["Ukupno"] = df_detail["tip0"] + df_detail["tip1"]

                # Dodaj red Σ
                total = {
                    "Serija": "Σ",
                    "tip0": df_detail["tip0"].sum(),
                    "tip1": df_detail["tip1"].sum(),
                    "Ukupno": df_detail["Ukupno"].sum()
                }
                df_detail = pd.concat([df_detail, pd.DataFrame([total])], ignore_index=True)

                st.dataframe(df_detail, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Greška: {e}")

# ---------- Tab 8: Kretanje 4098 kola – TIP 0 ----------
if selected_tab == "🚂 Kretanje 4098 kola–TIP 0":  
    st.subheader("🚂 Kretanje 4098 kola – TIP 0")  

    try:
        q = """
        WITH kola_clean AS (
            SELECT *,
                   TRY_CAST(SUBSTR("Broj kola", 3, LENGTH("Broj kola") - 3) AS BIGINT) AS broj_clean
            FROM kola
        ),
        poslednji AS (
            SELECT 
                s."Broj kola",
                s.TIP,
                s.TelegBaza,
                s.PR,
                s.NR,
                k.Serija,
                k.Stanica,
                st.Naziv AS NazivStanice,
                k.DatumVreme,
                ROW_NUMBER() OVER (
                    PARTITION BY s."Broj kola"
                    ORDER BY k.DatumVreme DESC
                ) AS rn
            FROM stanje s
            LEFT JOIN kola_clean k 
              ON k.broj_clean = TRY_CAST(s."Broj kola" AS BIGINT)
            LEFT JOIN stanice st
              ON TRIM(CAST(k.Stanica AS VARCHAR)) = TRIM(CAST(st.Sifra AS VARCHAR))
            WHERE s.TIP = 0
        )
        SELECT 
            "Broj kola",
            Serija,
            Stanica,
            NazivStanice,
            TelegBaza,
            PR,
            NR,
            DatumVreme,
            datediff('day', DatumVreme, current_date) AS BrojDana
        FROM poslednji
        WHERE rn = 1 OR rn IS NULL
        ORDER BY BrojDana ASC
        """
        df_tip0 = run_sql(q)
         # --- PR & NR -> samo datum ---
        if "PR" in df_tip0.columns:
            df_tip0["PR"] = pd.to_datetime(df_tip0["PR"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "NR" in df_tip0.columns:
            df_tip0["NR"] = pd.to_datetime(df_tip0["NR"], errors="coerce").dt.strftime("%Y-%m-%d")

        if "BrojDana" in df_tip0.columns:
            df_tip0["BrojDana"] = df_tip0["BrojDana"].astype("Int64")

        # --- Filter po seriji ---
        series_options = ["Sve serije"] + sorted(df_tip0["Serija"].dropna().unique().tolist())
        selected_series = st.selectbox("🚆 Filtriraj po seriji kola", series_options, key="tip0_series")

        if selected_series != "Sve serije":
            df_tip0 = df_tip0[df_tip0["Serija"] == selected_series]

        # --- Filter po stanici ---
        station_options = ["Sve stanice"] + sorted(df_tip0["NazivStanice"].dropna().unique().tolist())
        selected_station = st.selectbox("📍 Filtriraj po stanici", station_options, key="tip0_station")

        if selected_station != "Sve stanice":
            df_tip0 = df_tip0[df_tip0["NazivStanice"] == selected_station]

        # --- Prikaz podataka ---
        st.dataframe(df_tip0, use_container_width=True)

        # --- Export CSV / Excel ---
        c1, c2 = st.columns(2)
        with c1:
            csv = df_tip0.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Preuzmi tabelu (CSV)", csv, "tip0_kretanje.csv", "text/csv")
        with c2:
            import io
            excel_bytes = io.BytesIO()
            with pd.ExcelWriter(excel_bytes, engine="openpyxl") as writer:
                df_tip0.to_excel(writer, sheet_name="TIP0", index=False)
            st.download_button(
                "⬇️ Preuzmi tabelu (Excel)",
                excel_bytes.getvalue(),
                "tip0_kretanje.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"❌ Greška: {e}")
# ---------- Tab 9: Kretanje 4098 kola – TIP 1 ----------
if selected_tab == "🚂 Kretanje 4098 kola–TIP 1":
    st.subheader("🚂 Kretanje 4098 kola – TIP 1")
    st.subheader("🚂 Kretanje 4098 kola – samo TIP 1")

    try:
        q = """
        WITH kola_clean AS (
            SELECT *,
                   TRY_CAST(SUBSTR("Broj kola", 3, LENGTH("Broj kola") - 3) AS BIGINT) AS broj_clean
            FROM kola
        ),
        poslednji AS (
            SELECT 
                s."Broj kola",
                s.TIP,
                s.TelegBaza,
                k.Serija,
                s.PR,
                s.NR,
                k.Stanica,
                st.Naziv AS NazivStanice,
                k.DatumVreme,
                ROW_NUMBER() OVER (
                    PARTITION BY s."Broj kola"
                    ORDER BY k.DatumVreme DESC
                ) AS rn
            FROM stanje s
            LEFT JOIN kola_clean k 
              ON k.broj_clean = TRY_CAST(s."Broj kola" AS BIGINT)
            LEFT JOIN stanice st
              ON TRIM(CAST(k.Stanica AS VARCHAR)) = TRIM(CAST(st.Sifra AS VARCHAR))
            WHERE s.TIP = 1
        )
        SELECT 
            "Broj kola",
            Serija,
            Stanica,
            NazivStanice,
            TelegBaza,
            PR,
            NR,
            DatumVreme,
            datediff('day', DatumVreme, current_date) AS BrojDana
        FROM poslednji
        WHERE rn = 1 OR rn IS NULL
        ORDER BY BrojDana DESC
        """
        df_tip1 = run_sql(q)
         # --- PR & NR -> samo datum ---
        if "PR" in df_tip0.columns:
            df_tip0["PR"] = pd.to_datetime(df_tip0["PR"], errors="coerce").dt.strftime("%Y-%m-%d")
        if "NR" in df_tip0.columns:
            df_tip0["NR"] = pd.to_datetime(df_tip0["NR"], errors="coerce").dt.strftime("%Y-%m-%d")

        if "BrojDana" in df_tip1.columns:
            df_tip1["BrojDana"] = df_tip1["BrojDana"].astype("Int64")

        # --- Filter po seriji ---
        series_options = ["Sve serije"] + sorted(df_tip1["Serija"].dropna().unique().tolist())
        selected_series = st.selectbox("🚆 Filtriraj po seriji kola", series_options, key="tip1_series")

        if selected_series != "Sve serije":
            df_tip1 = df_tip1[df_tip1["Serija"] == selected_series]

        # --- Filter po stanici ---
        station_options = ["Sve stanice"] + sorted(df_tip1["NazivStanice"].dropna().unique().tolist())
        selected_station = st.selectbox("📍 Filtriraj po stanici", station_options, key="tip1_station")

        if selected_station != "Sve stanice":
            df_tip1 = df_tip1[df_tip1["NazivStanice"] == selected_station]

        # --- Prikaz podataka ---
        st.dataframe(df_tip1, use_container_width=True)

        # --- Export CSV / Excel ---
        c1, c2 = st.columns(2)
        with c1:
            csv = df_tip1.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Preuzmi tabelu (CSV)", csv, "tip1_kretanje.csv", "text/csv")
        with c2:
            import io
            excel_bytes = io.BytesIO()
            with pd.ExcelWriter(excel_bytes, engine="openpyxl") as writer:
                df_tip1.to_excel(writer, sheet_name="TIP1", index=False)
            st.download_button(
                "⬇️ Preuzmi tabelu (Excel)",
                excel_bytes.getvalue(),
                "tip1_kretanje.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"❌ Greška: {e}")

# ---------- Tab 10: Kola po serijama ----------
if selected_tab == "📊 Kola po serijama":
    st.subheader("📊 Kola po serijama")
    st.subheader("📊 Pivot po seriji i stanicama")

    try:
        q = """
        WITH kola_clean AS (
            SELECT *,
                   TRY_CAST(SUBSTR("Broj kola", 3, LENGTH("Broj kola") - 3) AS BIGINT) AS broj_clean
            FROM kola
        ),
        poslednji AS (
            SELECT 
                k.Serija,
                k.Stanica,
                st.Naziv AS NazivStanice,
                s.TIP,
                ROW_NUMBER() OVER (
                    PARTITION BY k.broj_clean
                    ORDER BY k.DatumVreme DESC
                ) AS rn
            FROM kola_clean k
            JOIN stanje s 
              ON k.broj_clean = TRY_CAST(s."Broj kola" AS BIGINT)
            LEFT JOIN stanice st
              ON TRIM(CAST(k.Stanica AS VARCHAR)) = TRIM(CAST(st.Sifra AS VARCHAR))
        )
        SELECT *
        FROM poslednji
        WHERE rn = 1
        """
        df_last = run_sql(q)

        # Pivot tabela po seriji
        df_pivot = (
            df_last.groupby(["Serija", "TIP"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        if 0 not in df_pivot.columns:
            df_pivot[0] = 0
        if 1 not in df_pivot.columns:
            df_pivot[1] = 0

        df_pivot = df_pivot.rename(columns={0: "tip0", 1: "tip1"})
        df_pivot["Ukupno"] = df_pivot["tip0"] + df_pivot["tip1"]

        # Dodaj Σ red
        total_row = {
            "Serija": "Σ",
            "tip0": df_pivot["tip0"].sum(),
            "tip1": df_pivot["tip1"].sum(),
            "Ukupno": df_pivot["Ukupno"].sum()
        }
        df_pivot = pd.concat([df_pivot, pd.DataFrame([total_row])], ignore_index=True)

        # Dva dela ekrana
        left, right = st.columns([1, 1])

        with left:
            st.markdown("### 📋 Ukupan broj kola (po serijama)")
            st.dataframe(df_pivot, use_container_width=True)

            # Export CSV / Excel
            c1, c2 = st.columns(2)
            with c1:
                csv = df_pivot.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Preuzmi pivot (CSV)", csv, "kola_po_serijama.csv", "text/csv")
            with c2:
                import io
                excel_bytes = io.BytesIO()
                with pd.ExcelWriter(excel_bytes, engine="openpyxl") as writer:
                    df_pivot.to_excel(writer, sheet_name="Pivot", index=False)
                st.download_button(
                    "⬇️ Preuzmi pivot (Excel)",
                    excel_bytes.getvalue(),
                    "kola_po_serijama.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        with right:
            st.markdown("### 🚆 Klikni / izaberi seriju")
            series_list = df_pivot[df_pivot["Serija"] != "Σ"]["Serija"].tolist()
            selected_series = st.selectbox("", ["Nijedna"] + series_list)

            if selected_series != "Nijedna":
                st.markdown(f"### 🔎 Detalji za seriju: **{selected_series}**")
                df_detail = (
                    df_last[df_last["Serija"] == selected_series]
                    .groupby(["Stanica", "NazivStanice", "TIP"])
                    .size()
                    .unstack(fill_value=0)
                    .reset_index()
                )

                if 0 not in df_detail.columns:
                    df_detail[0] = 0
                if 1 not in df_detail.columns:
                    df_detail[1] = 0

                df_detail = df_detail.rename(columns={0: "tip0", 1: "tip1"})
                df_detail["Ukupno"] = df_detail["tip0"] + df_detail["tip1"]

                # Σ red
                total = {
                    "Stanica": "Σ",
                    "NazivStanice": "Ukupno",
                    "tip0": df_detail["tip0"].sum(),
                    "tip1": df_detail["tip1"].sum(),
                    "Ukupno": df_detail["Ukupno"].sum()
                }
                df_detail = pd.concat([df_detail, pd.DataFrame([total])], ignore_index=True)

                st.dataframe(df_detail, use_container_width=True)

                # Export CSV / Excel
                c3, c4 = st.columns(2)
                with c3:
                    csv_detail = df_detail.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        f"⬇️ Preuzmi detalje {selected_series} (CSV)",
                        csv_detail,
                        f"{selected_series}_detalji.csv",
                        "text/csv"
                    )
                with c4:
                    excel_bytes_detail = io.BytesIO()
                    with pd.ExcelWriter(excel_bytes_detail, engine="openpyxl") as writer:
                        df_detail.to_excel(writer, sheet_name="Detalji", index=False)
                    st.download_button(
                        f"⬇️ Preuzmi detalje {selected_series} (Excel)",
                        excel_bytes_detail.getvalue(),
                        f"{selected_series}_detalji.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    except Exception as e:
        st.error(f"Greška: {e}")

# ---------- TAB 11: Prosečna starost vagona po seriji ----------
elif selected_tab == "📊 Prosečna starost":
    st.subheader("📊 Prosečna starost vagona po seriji")
    try:
        # Računanje prosečne starosti po seriji (kolona 3)
        df_age = run_sql(f"""
            SELECT 
                "3" AS Serija,
                ROUND(AVG(EXTRACT(YEAR FROM CURRENT_DATE) - CAST("MIKP GOD" AS INTEGER)), 1) AS prosečna_starost
            FROM "stanje"
            WHERE "MIKP GOD" IS NOT NULL
            GROUP BY "3"
            ORDER BY prosečna_starost DESC
        """)

        if df_age.empty:
            st.info("ℹ️ Nema podataka o godini proizvodnje.")
        else:
            st.dataframe(df_age, use_container_width=True)

            # Vizuelizacija bar chart
            st.bar_chart(df_age.set_index("Serija")["prosečna_starost"])

    except Exception as e:
        st.error(f"❌ Greška pri računanju starosti: {e}")

# ---------- TAB 12: Provera grešaka po statusu ----------

if selected_tab == "🛑 Provera grešaka po statusu":
    st.header("🛑 Provera grešaka po statusu")

    # 🔌 Povezivanje na DuckDB
    con = duckdb.connect("C:\\Teretna kola\\kola_sk.db")  # prilagodi putanju

    # ✅ Kreiramo view bez razmaka za tabelu "stanje SK"
    con.execute("""
        CREATE OR REPLACE VIEW stanje_SK AS
        SELECT * FROM "stanje SK";
    """)

    # 📝 Opcionalni unos brojeva kola
    st.subheader("Opcionalno: unesite listu brojeva kola (odvojene zarezom)")
    brojevi_kola_input = st.text_area(
        "Brojevi kola:",
        value="",
        help="Ako unesete listu, proveriće samo za ta kola"
    )
    if brojevi_kola_input:
        brojevi_kola = [int(x.strip()) for x in brojevi_kola_input.split(",") if x.strip().isdigit()]
    else:
        brojevi_kola = []

    # 👇 Batch veličina — stavili smo je PRE nego što klikneš na dugme
    batch_size = st.number_input("Batch veličina (broj kola po grupi)", value=500, step=100)

    # 🚀 Dugme za proveru
    if st.button("🔍 Proveri greške po statusu"):

        # Ako nisu uneti brojevi, uzmi iz tabele stanje_SK
        if brojevi_kola:
            kola_list = brojevi_kola
        else:
            df_stanje = con.execute('SELECT "Broj kola" FROM stanje_SK LIMIT 4098').fetchdf()
            kola_list = df_stanje["Broj kola"].tolist()

        greske_df = pd.DataFrame()
        total_batches = (len(kola_list) + batch_size - 1) // batch_size

        # --- Kreiranje progress bar i status teksta ---
        progress_bar = st.progress(0)
        status_text = st.empty()

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(kola_list))
            batch = kola_list[start_idx:end_idx]

            # 🔍 SQL upit za batch
            sql = f"""
            WITH kola_clean AS (
                SELECT *,
                       TRY_CAST(SUBSTR("Broj kola", 3, LENGTH("Broj kola") - 3) AS BIGINT) AS BrojKolaClean
                FROM kola
            ),
            batch_filtered AS (
                SELECT *
                FROM kola_clean
                WHERE BrojKolaClean IN ({','.join(map(str, batch))})
            ),
            windowed AS (
                SELECT *,
                       LAG("Status") OVER (PARTITION BY BrojKolaClean ORDER BY "DatumVreme") AS PrevStatus
                FROM batch_filtered
            )
            SELECT *
            FROM windowed
            WHERE "Status" = PrevStatus
            ORDER BY BrojKolaClean, "DatumVreme"
            """

            df_batch_errors = con.execute(sql).fetchdf()
            greske_df = pd.concat([greske_df, df_batch_errors], ignore_index=True)

            # --- Update progress bar i status ---
            progress = end_idx / len(kola_list)
            progress_bar.progress(progress)
            status_text.text(f"Obrađeno kola: {end_idx}/{len(kola_list)} | Pronađene greške: {len(greske_df)}")

        # --- Prikaz rezultata ---
        if greske_df.empty:
            st.success("✅ Nema grešaka po statusu za izabrana kola.")
        else:
            st.warning(f"⚠️ Pronađeno ukupno {len(greske_df)} grešaka!")
            st.dataframe(greske_df, use_container_width=True)

            # Dugme za eksport u Excel
            excel_file = "greske_status.xlsx"
            if len(greske_df) > 1048576:
                st.error("⚠️ Previše grešaka za eksport u Excel (limit je 1.048.576 redova). Preporučujemo batch export.")
            else:
                greske_df.to_excel(excel_file, index=False)
                with open(excel_file, "rb") as f:
                    st.download_button(
                        label="📥 Preuzmi Excel",
                        data=f,
                        file_name=excel_file,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
# =============================
# Tab 13 – Kretanje vozova
# =============================
if selected_tab == "🚂 Kretanje vozova":
    st.subheader("🚂 Kretanje vozova sa sastavom")

    # 🔹 Konekcija
    con = duckdb.connect("C:\\Teretna kola\\kola_sk.db")

    # ✅ Provera da li tabela postoji
    tables = con.execute("SHOW TABLES").fetchdf()
    if "kola" not in tables["name"].tolist():
        st.error("❌ Tabela 'kola' ne postoji u bazi. Proveri import.")
        st.stop()

    # 🔹 Inicijalizacija state
    if "tab13_show_data" not in st.session_state:
        st.session_state.tab13_show_data = False
    if "tab13_filters" not in st.session_state:
        st.session_state.tab13_filters = None
    if "tab13_open_voznja" not in st.session_state:
        st.session_state.tab13_open_voznja = None

    # -----------------------
    # 🔹 Glavni filteri
    # -----------------------
    col1, col2, col3 = st.columns([1, 1.5, 1.5])

    # 📌 Način filtriranja
    with col1:
        st.markdown(
            "<h4 style='text-align: center; font-size:18px;'>📅 Izaberi period</h4>", 
            unsafe_allow_html=True
        )
        mode = st.radio("", ["Godina/Mesec", "Period datuma"], key="filter_mode")

        if mode == "Godina/Mesec":
            c1, c2 = st.columns(2)
            with c1:
                selected_year = st.selectbox("Godina", list(range(2015, 2026)))
            with c2:
                selected_month = st.selectbox("Mesec", list(range(1, 13)))
        elif mode == "Period datuma":
            c1, c2 = st.columns(2)
            with c1:
                start_date = st.date_input("📅 Početni datum", value=date(2025, 6, 15))
            with c2:
                end_date = st.date_input("📅 Krajnji datum", value=date(2025, 7, 16))

    # 🔎 Broj voza
    with col2:
        st.markdown(
            "<h4 style='text-align: center; font-size:18px;'>🚉 Broj voza-opciono (više unosa odvojiti zarezom)</h4>", 
            unsafe_allow_html=True
        )
        voz_input = st.text_input("", value="")

    # 🎛 Opcioni filteri
    with col3:
        st.markdown(
            "<h4 style='text-align: center; font-size:18px;margin-bottom:24px;'>🔎 Opcioni filteri</h4>", 
            unsafe_allow_html=True)
        with st.expander("Izaberi filter", expanded=False):
            statusi = con.execute('SELECT DISTINCT "Status" FROM kola ORDER BY "Status"').fetchdf()
            sel_status = st.multiselect("Status", statusi["Status"].dropna().tolist())

            stanice = con.execute('SELECT DISTINCT "Stanica" FROM kola ORDER BY "Stanica"').fetchdf()
            sel_stanica = st.multiselect("Stanica", stanice["Stanica"].dropna().tolist())

            otp_drz = con.execute('SELECT DISTINCT "Otp. država" FROM kola ORDER BY "Otp. država"').fetchdf()
            sel_otp_drz = st.multiselect("Otp. država", otp_drz["Otp. država"].dropna().tolist())

            otp_st = con.execute('SELECT DISTINCT "Otp st" FROM kola ORDER BY "Otp st"').fetchdf()
            sel_otp_st = st.multiselect("Otp st", otp_st["Otp st"].dropna().tolist())

            up_drz = con.execute('SELECT DISTINCT "Uputna država" FROM kola ORDER BY "Uputna država"').fetchdf()
            sel_up_drz = st.multiselect("Uputna država", up_drz["Uputna država"].dropna().tolist())

            up_st = con.execute('SELECT DISTINCT "Up st" FROM kola ORDER BY "Up st"').fetchdf()
            sel_up_st = st.multiselect("Up st", up_st["Up st"].dropna().tolist())

    # 🔹 Dugme za prikaz
    if st.button("📊 Prikaži podatke"):
        if mode == "Godina/Mesec":
            where_clause = f"""
                EXTRACT(year FROM "DatumVreme") = {selected_year}
                AND EXTRACT(month FROM "DatumVreme") = {selected_month}
            """
            title = f"📅 {selected_month}/{selected_year}"
        else:
            where_clause = f"""
                "DatumVreme" >= '{start_date}'
                AND "DatumVreme" <= '{end_date} 23:59:59'
            """
            title = f"📅 Od {start_date} do {end_date}"

        # 🚂 Filter vozova
        if voz_input.strip():
            voz_list = [v.strip() for v in voz_input.split(",") if v.strip()]
            voz_values = ",".join([f"'{v}'" for v in voz_list])
            where_clause += f""" AND "Voz br" IN ({voz_values}) """

        st.session_state.tab13_show_data = True
        st.session_state.tab13_filters = {"where": where_clause, "title": title}
        st.session_state.tab13_open_voznja = None  # reset otvorenog voza

    # -----------------------
    # 📌 Dinamički WHERE uslov
    # -----------------------
    if "tab13_filters" in st.session_state and st.session_state.tab13_filters:
        filters = st.session_state.tab13_filters
        where_parts = [filters["where"]]  # osnovni uslov
    else:
        filters = {"where": "1=1"}  # fallback, ako nije definisan
        where_parts = [filters["where"]]

    # opciono dodajemo filtere iz expander-a
    if sel_status:
        where_parts.append(f'"Status" IN ({",".join([f"\'{s}\'" for s in sel_status])})')

    if sel_stanica:
        where_parts.append(f'"Stanica" IN ({",".join([f"\'{s}\'" for s in sel_stanica])})')

    if sel_otp_drz:
        where_parts.append(f'"Otp. država" IN ({",".join([f"\'{s}\'" for s in sel_otp_drz])})')

    if sel_otp_st:
        where_parts.append(f'"Otp st" IN ({",".join([f"\'{s}\'" for s in sel_otp_st])})')

    if sel_up_drz:
        where_parts.append(f'"Uputna država" IN ({",".join([f"\'{s}\'" for s in sel_up_drz])})')

    if sel_up_st:
        where_parts.append(f'"Up st" IN ({",".join([f"\'{s}\'" for s in sel_up_st])})')

    # konačni WHERE
    final_where = " AND ".join(where_parts)
    # konačni WHERE
    final_where = " AND ".join(where_parts)
    # sačuvaj u session_state da SQL zna šta da koristi
    st.session_state.tab13_filters = {
        **filters,  # postojeći title, where, itd.
        "final_where": final_where }
    # -----------------------
    # 📊 Prikaz podataka
    # -----------------------
    if st.session_state.tab13_show_data and st.session_state.tab13_filters:
        filters = st.session_state.tab13_filters
        st.subheader(filters["title"])

        # --- SUMARNI upit ---
        sql_summary = f"""
        SELECT
            "Voz br",
            COUNT(DISTINCT "Broj kola") AS "Br. kola u vozu",
            ROUND(SUM(
                CASE 
                    WHEN CAST("Tara" AS DOUBLE) > 60 THEN CAST("Tara" AS DOUBLE)/10
                    ELSE CAST("Tara" AS DOUBLE)
                END
            ), 1) AS Tara,
            ROUND(SUM(
                CASE 
                    WHEN CAST("NetoTone" AS DOUBLE) > 60 THEN CAST("NetoTone" AS DOUBLE)/10
                    ELSE CAST("NetoTone" AS DOUBLE)
                END
            ), 1) AS Neto,
            ROUND(SUM(
                CASE 
                    WHEN CAST("dužina vagona" AS DOUBLE) > 60 THEN CAST("dužina vagona" AS DOUBLE)/10
                    ELSE CAST("dužina vagona" AS DOUBLE)
                END
            ), 1) AS "Dužina voza",
            "Stanica",
            "Status",
            "DatumVreme",

            CASE WHEN COUNT(DISTINCT "Otp. država") = 1
                THEN MAX("Otp. država") ELSE NULL END AS "Otp. država",
            CASE WHEN COUNT(DISTINCT "Otp st") = 1
                THEN MAX("Otp st") ELSE NULL END AS "Otp st",
            CASE WHEN COUNT(DISTINCT "Uputna država") = 1
                THEN MAX("Uputna država") ELSE NULL END AS "Uputna država",
        CASE WHEN COUNT(DISTINCT "Up st") = 1
             THEN MAX("Up st") ELSE NULL END AS "Up st"

        FROM kola
        WHERE {filters["final_where"]}
        GROUP BY "Voz br", "Stanica", "Status", "DatumVreme"
        ORDER BY "Voz br", "DatumVreme", CAST("Status" AS INT)
        """
        df_summary = con.execute(sql_summary).fetchdf()

        if df_summary.empty:
            st.warning("⚠️ Nema podataka za izabrani filter.")
        else:
            # ✅ Header red
            with st.container(border=True):
                cols = st.columns([0.8,1.3,1.4,1.2,1.2,1.6,2,1.4,2,1.4,1.4,1.6,1.6])
                headers = ["Sastav","Voz br","Br. kola u vozu","Tara","Neto",
                       "Dužina voza","Stanica","Status","DatumVreme",
                       "Otp. država","Otp st","Uputna država","Up st"]
                for c, h in zip(cols, headers):
                    c.markdown(f"**{h}**")

            # ✅ Redovi vožnji
            for i, row in df_summary.iterrows():
                row_id = f"{row['Voz br']}_{row['DatumVreme']}_{row['Status']}"
                with st.container(border=True):
                    cols = st.columns([0.8,1.3,1.4,1.2,1.2,1.6,2,1.4,2,1.4,1.4,1.6,1.6])

                    # ➕ dugme za prikaz kola
                    with cols[0]:
                        icon = "✖" if st.session_state.tab13_open_voznja == row_id else "➕"
                        if st.button(icon, key=f"btn_{i}"):
                            if st.session_state.tab13_open_voznja == row_id:
                                st.session_state.tab13_open_voznja = None
                            else:
                                st.session_state.tab13_open_voznja = row_id

                    # Glavni red
                    cols[1].write(row["Voz br"])
                    cols[2].write(f"{row['Br. kola u vozu']}")
                    cols[3].write(f"{row['Tara']:.1f}")
                    cols[4].write(f"{row['Neto']:.1f}")
                    cols[5].write(f"{row['Dužina voza']:.1f}")
                    cols[6].write(row["Stanica"])
                    cols[7].write(row["Status"])
                    cols[8].write(str(row["DatumVreme"]))
                    cols[9].write(row["Otp. država"] if row["Otp. država"] else "")
                    cols[10].write(row["Otp st"] if row["Otp st"] else "")
                    cols[11].write(row["Uputna država"] if row["Uputna država"] else "")
                    cols[12].write(row["Up st"] if row["Up st"] else "")

                # --- Ako nema jedinstvenih vrednosti → prikaži raspodelu ispod ---
                if (not row["Otp. država"]) or (not row["Otp st"]) or (not row["Uputna država"]) or (not row["Up st"]):
                    sql_detail_rel = f"""
                    SELECT
                        COUNT(DISTINCT "Broj kola") AS "Br. kola",
                        ROUND(SUM(
                            CASE 
                                WHEN TRY_CAST("Tara" AS DOUBLE) > 60 THEN TRY_CAST("Tara" AS DOUBLE)/10
                                ELSE TRY_CAST("Tara" AS DOUBLE)
                            END
                        ), 1) AS Tara,
                        ROUND(SUM(
                            CASE 
                                WHEN TRY_CAST("NetoTone" AS DOUBLE) > 60 THEN TRY_CAST("NetoTone" AS DOUBLE)/10
                                ELSE TRY_CAST("NetoTone" AS DOUBLE)
                            END
                        ), 1) AS Neto,
                        ROUND(SUM(
                            CASE 
                                WHEN TRY_CAST("dužina vagona" AS DOUBLE) > 60 THEN TRY_CAST("dužina vagona" AS DOUBLE)/10
                                ELSE TRY_CAST("dužina vagona" AS DOUBLE)
                            END
                        ), 1) AS "Dužina voza",
                        "Otp. država",
                        "Otp st",
                        "Uputna država",
                        "Up st"
                    FROM kola
                    WHERE {filters["final_where"]}
                        AND "Voz br" = '{row["Voz br"]}'
                        AND "Stanica" = '{row["Stanica"]}'
                        AND "Status" = '{row["Status"]}'
                        AND "DatumVreme" = '{row["DatumVreme"]}'
                    GROUP BY "Otp. država","Otp st","Uputna država","Up st"
                    """
                    df_detail_rel = con.execute(sql_detail_rel).fetchdf()

                    # prikaži kao nastavak glavne tabele
                    for j, rel in df_detail_rel.iterrows():
                        with st.container(border=True):
                            cols = st.columns([0.7,1.4,1.4,1.2,1.2,1.6,2,1.4,2,1.4,1.4,1.6,1.6])
                            cols[0].write("")  # prazno polje
                            cols[1].write("")  # nema Voz br
                            cols[2].write(f"{rel['Br. kola']}")
                            cols[3].write(f"{rel['Tara']:.1f}")
                            cols[4].write(f"{rel['Neto']:.1f}")
                            cols[5].write(f"{rel['Dužina voza']:.1f}")
                            cols[6].write("")  
                            cols[7].write("")  
                            cols[8].write("")  
                            cols[9].write(rel["Otp. država"])
                            cols[10].write(rel["Otp st"])
                            cols[11].write(rel["Uputna država"])
                            cols[12].write(rel["Up st"])

                # --- Sastav kola (klik na +) ---
                if st.session_state.tab13_open_voznja == row_id:
                    st.markdown(f"📋 **Sastav voza {row['Voz br']} ({row['DatumVreme']}) – Status {row['Status']}**")

                    sql_detail_kola = """
                        SELECT
                            ROW_NUMBER() OVER () AS "Redni broj kola u vozu",
                            "Broj kola",
                            ROUND(
                                CASE 
                                    WHEN CAST("Tara" AS DOUBLE) > 60 THEN CAST("Tara" AS DOUBLE)/10
                                    ELSE CAST("Tara" AS DOUBLE)
                                END, 1
                            ) AS Tara,
                            ROUND(
                                CASE 
                                    WHEN CAST("NetoTone" AS DOUBLE) > 60 THEN CAST("NetoTone" AS DOUBLE)/10
                                    ELSE CAST("NetoTone" AS DOUBLE)
                                END, 1
                            ) AS Neto,
                            ROUND(
                                CASE 
                                    WHEN CAST("dužina vagona" AS DOUBLE) > 60 THEN CAST("dužina vagona" AS DOUBLE)/10
                                    ELSE CAST("dužina vagona" AS DOUBLE)
                                END, 1
                            ) AS "Dužina",
                            "Otp. država",
                            "Otp st",
                            "Uputna država",
                            "Up st"
                        FROM kola
                        WHERE "Voz br" = ?
                            AND "DatumVreme" = ?
                            AND "Status" = ?
                    """
                    df_kola_voza = con.execute(
                        sql_detail_kola,
                        [row["Voz br"], row["DatumVreme"], row["Status"]]
                    ).fetchdf()

                    if df_kola_voza.empty:
                        st.info("ℹ️ Nema kola za ovaj voz.")
                    else:
                        st.dataframe(df_kola_voza, use_container_width=True, hide_index=True)

                        # Excel export
                        buffer = io.BytesIO()
                        df_kola_voza.to_excel(buffer, index=False, engine="openpyxl")
                        buffer.seek(0)
                        st.download_button(
                            label="📥 Preuzmi kola u Excelu",
                            data=buffer,
                            file_name=f"sastav_{row['Voz br']}_{row['DatumVreme']}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
# =============================
# Tab 14 – Km prazno/tovareno
# =============================

if selected_tab == "📏 Km prazno/tovareno":
    st.subheader("📏 Izračunavanje kilometara prazno/tovareno")
    
    # --- Ulaz ---
    kola_input = st.text_input("Unesi broj(eve) kola (odvoji zarezom)", key="tab14_kola")
    today = date.today()
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("📅 Početni datum", value=date(today.year, today.month, 1))
    with c2:
        end_date = st.date_input("📅 Krajnji datum", value=date.today())


    if st.button("🔍 Izračunaj km", key="tab14_btn"):
        if not kola_input:
            st.warning("⚠️ Unesi broj kola!")
        else:
            kola_list = [k.strip() for k in kola_input.split(",")]

            # --- Uzimamo kretanja iz baze ---
            sql = f"""
                SELECT *
                FROM {TABLE_NAME}
                WHERE "Broj kola" IN ({','.join([f"'{k}'" for k in kola_list])})
                  AND DatumVreme BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY "Broj kola", DatumVreme
            """
            df = run_sql(sql)
            if df.empty:
                st.warning("⚠️ Nema podataka za traženi period.")
            else:
                # --- Prazno / Tovareno ---
                df["Stanje"] = df["NetoTone"].astype(float).apply(lambda x: "Prazno" if x <= 5 else "Tovareno")

                # --- Filtriramo samo domaća kretanja (72) ---
                df = df[(df["Otp. država"] == "72") & (df["Uputna država"] == "72")]

                # --- Preskakanje duplikata parova ---
                df["Par"] = df["Otp st"].astype(str) + "-" + df["Up st"].astype(str)
                df = df.loc[df["Par"].shift() != df["Par"]]

                # --- Dodavanje rastojanja ---
                rast = pd.read_excel("rastojanja_medju_stanicama.xlsx")
                # Pre merge-a – usklađivanje tipova
                df["Otp st"] = df["Otp st"].astype(str).str.strip()
                df["Up st"]  = df["Up st"].astype(str).str.strip()

                rast["Од (шифра)"] = rast["Од (шифра)"].astype(str).str.strip()
                rast["До (шифра)"] = rast["До (шифра)"].astype(str).str.strip()

                merged = df.merge(
                    rast,
                    left_on=["Otp st", "Up st"],
                    right_on=["Од (шифра)", "До (шифра)"],
                    how="left"
                )

                # --- Grupisanje po stanju ---
                summary = merged.groupby("Stanje")[["Тарифски километри", "Дужина (km)"]].sum().reset_index()

                # --- Prikaz ---
                st.subheader("📊 Detaljno kretanje")
                st.dataframe(merged, use_container_width=True)

                st.subheader("📈 Sažetak po stanju")

                # ovo je tvoja početna tabela sa sažetkom
                df_summary = summary.copy()

                # Dodavanje reda "Ukupno"
                df_total = df_summary.select_dtypes(include="number").sum().to_frame().T
                df_total.index = ["Ukupno"]

                # Dodaj tekstualne kolone (ako postoje) sa praznim vrednostima
                for col in df_summary.columns:
                    if col not in df_total.columns:
                        df_total[col] = ""

                # Redosled kolona kao u originalu
                df_total = df_total[df_summary.columns]

                # Spajanje originalne tabele + red ukupno
                df_summary = pd.concat([df_summary, df_total], ignore_index=False)

                # Formatiranje na jednu decimalu
                df_summary = df_summary.map(lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x)

                # Prikaz u Streamlit
                st.dataframe(df_summary, use_container_width=True)
                # --- Export ---
                csv = merged.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Preuzmi CSV", csv, "km_prazno_tovareno.csv", "text/csv")
# =============================
# Tab 15 – Revizija
# =============================

import pandas as pd
import streamlit as st

# --- Inicijalizacija session_state ---
if "revizija_df" not in st.session_state:
    st.session_state["revizija_df"] = None
if "revizija_filtered_df" not in st.session_state:
    st.session_state["revizija_filtered_df"] = None
if "revizija_prikazano" not in st.session_state:
    st.session_state["revizija_prikazano"] = False
if "revizija_dana_input" not in st.session_state:
    st.session_state["revizija_dana_input"] = 30
if "revizija_danas" not in st.session_state:
    st.session_state["revizija_danas"] = pd.to_datetime("today").normalize()

if selected_tab == "🔧 Revizija":
    st.subheader("🔧 Revizija")

    # -----------------------
    # Filteri u 3 kolone
    # -----------------------
    col1, col2, col3 = st.columns([1, 1.5, 1.5])

    # TIP filter
    with col1:
        st.markdown("<h4 style='text-align: center; font-size:18px;'>🚦 Izaberi TIP</h4>", unsafe_allow_html=True)
        tip_options = ["TIP 0 (istekla)", "TIP 1 (važeća)", "Sva kola"]
        tip_filter = st.selectbox("", tip_options, index=2)

    # Broj kola
    with col2:
        st.markdown("<h4 style='text-align: center; font-size:18px;'>🚃 Broj kola (opciono)</h4>", unsafe_allow_html=True)
        broj_kola_input = st.text_input("", value="", key="rev_broj_kola")

    # Opcioni filteri
    with col3:
        st.markdown("<h4 style='text-align: center; font-size:18px; margin-bottom:24px;'>🔎 Opcioni filteri</h4>", unsafe_allow_html=True)
        with st.expander("Izaberi", expanded=False):
            try:
                # Učitavanje Excel fajlova samo jednom
                df_stanje = pd.read_excel("Stanje SK.xlsx")
                df_stanje["Broj kola"] = pd.to_numeric(df_stanje["Broj kola"], errors="coerce").astype('Int64')
                df_stanje = df_stanje.rename(columns={"3": "Serija"})
                df_stanje = df_stanje[["Broj kola", "Serija", "PR", "NR", "TelegBaza", "Napomena"]]

                df_opravke = pd.read_excel("Redovne opravke.xlsx")
                df_opravke["Broj kola"] = pd.to_numeric(df_opravke["Broj kola"], errors="coerce").astype('Int64')
                df_opravke.columns = df_opravke.columns.str.strip()
                df_opravke["Datum revizije"] = pd.to_datetime(df_opravke["Datum revizije"], errors="coerce")
                df_opravke["Datum naredne revizije"] = pd.to_datetime(df_opravke["Datum naredne revizije"], errors="coerce")

                # Filteri multiselect
                sel_serija = st.multiselect("Serija", sorted(df_stanje["Serija"].dropna().unique().tolist()))
                sel_radionica = st.multiselect(
                    "Radionica",
                    df_opravke["Radionica"].dropna().unique().tolist() if "Radionica" in df_opravke.columns else []
                )
                sel_vrsta = st.multiselect(
                    "Vrsta",
                    df_opravke["Vrsta"].dropna().unique().tolist() if "Vrsta" in df_opravke.columns else []
                )
                sel_datum = st.date_input("Datum revizije", [])

            except FileNotFoundError as e:
                st.error(f"Fajl nije pronađen: {e}")
                df_stanje = pd.DataFrame()
                df_opravke = pd.DataFrame()
                sel_serija, sel_radionica, sel_vrsta, sel_datum = [], [], [], []

    # -----------------------
    # Dugme za prikaz podataka
    # -----------------------
    if st.button("📌 Prikaži reviziju"):
        try:
            if df_stanje.empty or df_opravke.empty:
                st.warning("Excel fajlovi nisu učitani.")
            else:
                # Uzmi poslednji unos po Broju kola
                df_opravke_latest = df_opravke.sort_values("Datum revizije").groupby("Broj kola").tail(1)

                # Merge tabele
                df = df_stanje.merge(df_opravke_latest, on="Broj kola", how="left")

                # Datum za TIP
                df["Datum_za_tip"] = df["Datum naredne revizije"].fillna(df["NR"])
                danas = pd.to_datetime("today").normalize()

                df["TIP"] = None
                df.loc[df["Datum_za_tip"].notna() & (df["Datum_za_tip"] < danas), "TIP"] = "TIP 0 (istekla)"
                df.loc[df["Datum_za_tip"].notna() & (df["Datum_za_tip"] >= danas), "TIP"] = "TIP 1 (važeća)"
                df.loc[df["Datum_za_tip"].isna(), "TIP"] = "Nepoznato"

                # Primena filtera
                if broj_kola_input:
                    brojevi = [b.strip() for b in broj_kola_input.split(",") if b.strip()]
                    df = df[df["Broj kola"].isin(brojevi)]

                if sel_serija:
                    df = df[df["Serija"].isin(sel_serija)]
                if sel_radionica and "Radionica" in df.columns:
                    df = df[df["Radionica"].isin(sel_radionica)]
                if sel_vrsta and "Vrsta" in df.columns:
                    df = df[df["Vrsta"].isin(sel_vrsta)]
                if sel_datum:
                    if isinstance(sel_datum, list) and len(sel_datum) == 2:
                        df = df[(df["Datum revizije"] >= pd.to_datetime(sel_datum[0])) &
                                (df["Datum revizije"] <= pd.to_datetime(sel_datum[1]))]
                    else:
                        df = df[df["Datum revizije"] == pd.to_datetime(sel_datum)]

                if tip_filter != "Sva kola":
                    df = df[df["TIP"] == tip_filter]

                # Sortiranje i upis u session_state
                df = df.sort_values("Datum_za_tip")
                st.session_state["revizija_df"] = df
                st.session_state["revizija_prikazano"] = True

                st.success(f"✅ Pronađeno {len(df)} kola.")
                st.dataframe(df, use_container_width=True)

                # Dugme za preuzimanje
                excel_file = "revizija_prikaz.xlsx"
                df.to_excel(excel_file, index=False)
                with open(excel_file, "rb") as f:
                    st.download_button("⬇️ Preuzmi Excel", f, file_name=excel_file)

        except Exception as e:
            st.error(f"Greška pri učitavanju podataka: {e}")

    # --- Filter dana do isteka
    if st.session_state["revizija_prikazano"] and st.session_state["revizija_df"] is not None:
        df = st.session_state["revizija_df"]
        dana = st.number_input(
            "📆 Spisak kola kojima ističe revizija u narednih X dana",
            min_value=1, max_value=365,
            value=st.session_state["revizija_dana_input"], step=1
        )
        st.session_state["revizija_dana_input"] = dana

        mask = (df["Datum_za_tip"].notna()) & (
            (df["Datum_za_tip"] - st.session_state["revizija_danas"]).dt.days <= dana
        ) & ((df["Datum_za_tip"] - st.session_state["revizija_danas"]).dt.days >= 0)

        filtered_df = df[mask]
        st.session_state["revizija_filtered_df"] = filtered_df

        st.info(f"📌 Pronađeno {len(filtered_df)} kola kojima revizija ističe u narednih {dana} dana.")
        st.dataframe(filtered_df, use_container_width=True)

        if not filtered_df.empty:
            excel_file = "revizija_istek.xlsx"
            filtered_df.to_excel(excel_file, index=False)
            with open(excel_file, "rb") as f:
                st.download_button("⬇️ Preuzmi Excel (istek)", f, file_name=excel_file)
