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

st.set_page_config(layout="wide", page_title="🚂 Teretna kola SK")


# -------------------- CONFIG --------------------
st.set_page_config(layout="wide")
st.title("🚂 Teretna kola SK")

HF_TOKEN = st.secrets["HF_TOKEN"]
HF_REPO = st.secrets["HF_REPO"]
ADMIN_PASS = st.secrets.get("ADMIN_PASS", "tajna123")
DEFAULT_FOLDER = "/tmp"
TABLE_NAME = "kola"

def get_last_state_sql(con):
    sql = """
    WITH kola_clean AS (
        SELECT *,
               TRY_CAST(SUBSTR("Broj kola", 3) AS BIGINT) AS broj_clean
        FROM kola
    ),
    poslednje AS (
        SELECT s."Broj kola" AS broj_kola,
               k.*,
               ROW_NUMBER() OVER (PARTITION BY s."Broj kola" ORDER BY k."DatumVreme" DESC) AS rn
        FROM stanje s
        LEFT JOIN kola_clean k
        ON TRY_CAST(s."Broj kola" AS BIGINT) = k.broj_clean
    )
    SELECT *
    FROM poslednje
    WHERE rn = 1
    """
    return con.execute(sql).fetchdf()

# -------------------- HF PREUZIMANJE PARQUET --------------------
@st.cache_data(show_spinner=True)
def get_parquet_file(filename: str) -> str:
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        repo_type="dataset",
        token=HF_TOKEN
    )
    return path

# -------------------- DUCKDB KONEKCIJA --------------------
@st.cache_resource
def get_duckdb_connection(parquet_files: list):
    con = duckdb.connect(database=":memory:")
    for f in parquet_files:
        path = get_parquet_file(f)
        table_name = os.path.splitext(os.path.basename(f))[0]
        con.execute(f"CREATE VIEW {table_name} AS SELECT * FROM '{path}'")
    return con

# -------------------- INIT DB IZ PARQUET --------------------
PARQUET_FILES = ["kola.parquet", "rastojanja.parquet", "stanice.parquet",
                 "stanje.parquet", "stanje_SK.parquet"]

con = get_duckdb_connection(PARQUET_FILES)

# -------------------- SQL HELPER --------------------
@st.cache_data
def run_sql(sql: str) -> pd.DataFrame:
    return con.execute(sql).fetchdf()

# -------------------- ADMIN LOGIN --------------------
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

st.sidebar.title("⚙️ Podešavanja")
if not st.session_state.admin_logged_in:
    password = st.sidebar.text_input("🔑 Unesi lozinku:", type="password")
    if st.sidebar.button("🔓 Otključaj"):
        if password == ADMIN_PASS:
            st.session_state.admin_logged_in = True
            st.sidebar.success("✅ Uspešno ste se prijavili!")
        else:
            st.sidebar.error("❌ Pogrešna lozinka.")
else:
    if st.sidebar.button("🚪 Odjavi se"):
        st.session_state.admin_logged_in = False
        st.sidebar.warning("🔒 Odjavljeni ste.")

# -------------------- HF PUSH --------------------
def push_file_to_hf(local_path, commit_message="Update baza"):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=os.path.basename(local_path),
        repo_id=HF_REPO,
        token=HF_TOKEN,
        repo_type="dataset"
    )
    st.success(f"✅ Poslat na Hugging Face: {os.path.basename(local_path)}")
# ---------- Funkcija za parsiranje TXT fajla ----------
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
                "Rid": line[47:48].strip(),
                "UN broj": line[48:52].strip(),
                "Reon": line[61:66].strip(),
                "tara": line[78:81].strip(),
                "NetoTone": line[83:86].strip(),
                "dužina vagona": line[88:91].strip(),
                "broj osovina": line[98:100].strip(),
                "Otp. država": line[123:125].strip(),
                "Otp st": line[125:130].strip(),
                "Uputna država": line[130:132].strip(),
                "Up st": line[132:137].strip(),
                "Broj kola": line[0:12].strip(),
                "Redni broj kola": line[57:59].strip(),
                "source_file": os.path.basename(path),
            })

    df = pl.DataFrame(rows)

    # Ispravka vremena 2400 → 0000 i pomeranje datuma
    df = df.with_columns([
        pl.when(pl.col("Vreme") == "2400")
          .then(pl.lit("0000"))
          .otherwise(pl.col("Vreme"))
          .alias("Vreme"),
        pl.when(pl.col("Vreme") == "2400")
          .then(
              (pl.col("Datum").str.strptime(pl.Date, "%Y%m%d", strict=False) + pl.duration(days=1))
              .dt.strftime("%Y%m%d")
          )
          .otherwise(pl.col("Datum"))
          .alias("Datum"),
    ])

    # Kombinovana kolona DatumVreme
    df = df.with_columns([
        (pl.col("Datum") + " " + pl.col("Vreme"))
            .str.strptime(pl.Datetime, "%Y%m%d %H%M", strict=False)
            .alias("DatumVreme"),
        pl.col("Datum").str.strptime(pl.Date, "%Y%m%d", strict=False).is_not_null().alias("Datum_validan")
    ])

    # Brojevi u float
    df = df.with_columns([
        (pl.col("tara").str.slice(0, 2) + "." + pl.col("tara").str.slice(2)).cast(pl.Float64, strict=False).alias("tara"),
        (pl.col("NetoTone").str.slice(0, 2) + "." + pl.col("NetoTone").str.slice(2)).cast(pl.Float64, strict=False).alias("NetoTone"),
        (pl.col("dužina vagona").str.slice(0, 2) + "." + pl.col("dužina vagona").str.slice(2)).cast(pl.Float64, strict=False).alias("dužina vagona"),
        pl.col("broj osovina").cast(pl.Int32, strict=False).alias("broj osovina"),
])

    return df
# ---------- GLOBALNO UCITAVANJE MAPE STANICA ----------
STANICE_MAP = {}
try:
    stanice_df = pd.read_excel("stanice.xlsx")  # fajl sa kolonama: sifra, naziv
    STANICE_MAP = dict(zip(stanice_df["sifra"].astype(str).str.strip(),
                           stanice_df["naziv"].astype(str).str.strip()))
except Exception as e:
    st.warning(f"⚠️ Nije moguće učitati mapu stanica: {e}")

# ---------- Funkcija za dodavanje naziva stanica ----------
def add_station_names(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje kolone 'Naziv st.', 'Naziv otp st' i 'Naziv up st.' koristeći globalnu mapu STANICE_MAP."""
    if not STANICE_MAP:
        return df

    # Naziv st. (za kolonu Stanica)
    if "Stanica" in df.columns:
        df["Stanica"] = df["Stanica"].astype(str).str.strip()
        df.insert(
            df.columns.get_loc("Stanica") + 1,
            "Naziv st.",
            df["Stanica"].map(STANICE_MAP)
        )

    # Normalizacija kolona
    for col in ["Otp. država", "Uputna država", "Otp st", "Up st"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lstrip("0").str.replace(".0","")

    # Naziv otp st (samo ako je Otp. država == 72)
    if "Otp st" in df.columns and "Otp. država" in df.columns:
        mask = df["Otp. država"] == "72"
        df["Naziv otp st."] = None
        df.loc[mask, "Naziv otp st."] = df.loc[mask, "Otp st"].map(STANICE_MAP)

    # Naziv up st (samo ako je Uputna država == 72)
    if "Up st" in df.columns and "Uputna država" in df.columns:
        mask = df["Uputna država"] == "72"
        df["Naziv up st."] = None
        df.loc[mask, "Naziv up st."] = df.loc[mask, "Up st"].map(STANICE_MAP)

    return df
# ---------- Inicijalizacija baze ----------
def init_database(folder: str, TABLE_NAME: str = "kola"):
    files = glob.glob(os.path.join(folder, "*.txt"))
    if not files:
        st.warning(f"⚠️ Nema txt fajlova u folderu: {folder}")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    all_dfs = []
    for i, f in enumerate(files, start=1):
        status_text.text(f"📄 Učitavam fajl {i}/{len(files)}: {os.path.basename(f)}")
        df_part = parse_txt(f)   # pretpostavljam da već imaš parse_txt funkciju
        all_dfs.append(df_part)

        progress_bar.progress(i / len(files))

    df = pl.concat(all_dfs)

    # ID kolona
    df = df.with_columns(pl.arange(0, df.height).alias("id"))

    # Numeričke kolone
    df = df.with_columns([
        pl.col("tara").cast(pl.Float64).alias("tara"),
        pl.col("NetoTone").cast(pl.Float64).alias("NetoTone"),
        pl.col("dužina vagona").cast(pl.Float64).alias("dužina vagona"),
        pl.col("broj osovina").cast(pl.Int32).alias("broj osovina")
    ])

    # Broj vagona
    df = df.with_columns([
        pl.col("Broj kola").str.slice(2, 9).str.replace(" ", "").cast(pl.Int64, strict=False).alias("Broj vagona")
    ])

    # Konverzija u pandas radi dodavanja naziva stanica
    df_pd = df.to_pandas()
    df_pd = add_station_names(df_pd)

    con = get_duckdb_connection()
    con.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    con.register("df_pd", df_pd)
    con.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM df_pd")
    con.unregister("df_pd")

    # ID_rb po datumu
    con.execute(f"""
        CREATE OR REPLACE TABLE {TABLE_NAME} AS
        SELECT *,
               ROW_NUMBER() OVER (ORDER BY DatumVreme ASC NULLS LAST, id ASC) AS ID_rb
        FROM {TABLE_NAME}
    """)

    save_state(set(files))
    progress_bar.empty()
    status_text.text("✅ Učitavanje završeno")
    st.success(f"✅ Inicijalno učitano {len(df)} redova iz {len(files)} fajlova")

# ---------- Update baze ----------
def update_database(folder: str, TABLE_NAME: str = "kola"):
    processed = load_state()
    files = set(glob.glob(os.path.join(folder, "*.txt")))
    new_files = files - processed
    if not new_files:
        st.info("ℹ️ Nema novih fajlova za unos.")
        return

    con = get_duckdb_connection()
    for f in sorted(new_files):
        df_new = parse_txt(f)

        # Nastavljanje ID-a
        max_id = run_sql(f"SELECT MAX(id) AS max_id FROM {TABLE_NAME}").iloc[0, 0]
        max_id = 0 if max_id is None else max_id
        df_new = df_new.with_columns(pl.arange(max_id + 1, max_id + 1 + df_new.height).alias("id"))

        # Numeričke kolone
        df_new = df_new.with_columns([
            pl.col("tara").cast(pl.Float64).alias("tara"),
            pl.col("NetoTone").cast(pl.Float64).alias("NetoTone"),
            pl.col("dužina vagona").cast(pl.Float64).alias("dužina vagona"),
            pl.col("broj osovina").cast(pl.Int32).alias("broj osovina")
        ])

        # Broj vagona
        df_new = df_new.with_columns([
            pl.col("Broj kola").str.slice(2, 9).str.replace(" ", "").cast(pl.Int64, strict=False).alias("Broj vagona")
        ])

        # Pandas i dodavanje naziva stanica
        df_new_pd = df_new.to_pandas()
        df_new_pd = add_station_names(df_new_pd)

        # Normalizacija kolona
        existing_cols = [c[1] for c in con.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()]
        for col in existing_cols:
            if col not in df_new_pd.columns:
                df_new_pd[col] = None
        df_new_pd = df_new_pd[existing_cols]

        con.register("df_new_pd", df_new_pd)
        con.execute(f"INSERT INTO {TABLE_NAME} SELECT * FROM df_new_pd")
        con.unregister("df_new_pd")

        processed.add(f)
        st.write(f"➕ Ubačeno {len(df_new_pd)} redova iz {os.path.basename(f)}")

    save_state(processed)

    # Osvežavanje ID_rb
    con.execute(f"""
        CREATE OR REPLACE TABLE {TABLE_NAME} AS
        SELECT *,
               ROW_NUMBER() OVER (ORDER BY DatumVreme ASC NULLS LAST, id ASC) AS ID_rb
        FROM {TABLE_NAME}
    """)

    st.success("✅ Update baze završen (ID_rb osvežen).")
# ---------- Dodavanje pojedinačnog fajla ----------
def add_txt_file_streamlit(uploaded_file, TABLE_NAME: str = TABLE_NAME):
    if uploaded_file is None:
        st.warning("⚠️ Niste izabrali fajl.")
        return

    tmp_path = os.path.join(DEFAULT_FOLDER, uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    df_new = parse_txt(tmp_path)

    # Numeričke kolone
    df_new = df_new.with_columns([
        (pl.col("tara").cast(pl.Float64, strict=False) / 10).alias("tara"),
        (pl.col("NetoTone").cast(pl.Float64, strict=False) / 10).alias("NetoTone"),
        (pl.col("dužina vagona").cast(pl.Float64, strict=False) / 10).alias("dužina vagona"),
        pl.col("broj osovina").cast(pl.Int32, strict=False).alias("broj osovina"),
    ])

    # Broj vagona
    df_new = df_new.with_columns([
        pl.col("Broj kola").str.slice(2).cast(pl.Int64, strict=False).alias("Broj vagona")
    ])

    con = get_duckdb_connection()
    # Dobavi poslednji ID
    try:
        max_id = run_sql(f"SELECT MAX(id) AS max_id FROM {TABLE_NAME}").iloc[0, 0]
        if max_id is None:
            max_id = 0
    except:
        max_id = 0
    df_new = df_new.with_columns(pl.arange(max_id + 1, max_id + 1 + df_new.height).alias("id"))

    # Dodavanje naziva stanica
    df_new_pd = df_new.to_pandas()
    add_station_names(df_new_pd)
    df_new = pl.from_pandas(df_new_pd)

    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    if TABLE_NAME in tables:
        existing_cols = [c[1] for c in con.execute(f"PRAGMA table_info({TABLE_NAME})").fetchall()]
        for col in existing_cols:
            if col not in df_new.columns:
                df_new = df_new.with_columns(pl.lit(None).alias(col))
        df_new = df_new.select(existing_cols)
        con.register("df_new", df_new)
        con.execute(f"INSERT INTO {TABLE_NAME} SELECT * FROM df_new")
        con.unregister("df_new")
    else:
        con.register("df_new", df_new)
        con.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM df_new")
        con.unregister("df_new")

    # Osvežavanje ID_rb
    con.execute(f"""
        CREATE OR REPLACE TABLE {TABLE_NAME} AS
        SELECT *,
               ROW_NUMBER() OVER (ORDER BY id ASC) AS ID_rb
        FROM {TABLE_NAME}
    """)

    st.success(f"✅ Fajl '{uploaded_file.name}' dodat u bazu ({len(df_new)} redova)")
 

# -------------------- ADMIN LOGIN --------------------
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

st.sidebar.title("⚙️ Podešavanja")

if not st.session_state.admin_logged_in:
    password = st.sidebar.text_input("🔑 Unesi lozinku:", type="password", key="admin_password")
    if st.sidebar.button("🔓 Otključaj", key="unlock_btn"):
        if password == ADMIN_PASS:
            st.session_state.admin_logged_in = True
            st.sidebar.success("✅ Uspešno ste se prijavili!")
        else:
            st.sidebar.error("❌ Pogrešna lozinka.")
    st.sidebar.warning("🔒 Podešavanja su zaključana.")
else:
    if st.sidebar.button("🚪 Odjavi se", key="logout_btn"):
        st.session_state.admin_logged_in = False
        st.sidebar.warning("🔒 Odjavljeni ste.")
# -------------------- AKO JE ADMIN ULOGOVAN --------------------
if st.session_state.admin_logged_in:
    admin_tabs = st.tabs([
        "📂 Inicijalizacija / Update baze",
        "🔍 Duplikati",
        "📄 Upload Excel",
        "📊 Pregled učitanih fajlova"
    ])

    # ================= TAB 1 =================
    with admin_tabs[0]:
        st.subheader("📂 Inicijalizacija / Update baze")
        folder_path = st.text_input("Folder sa TXT fajlovima", value=DEFAULT_FOLDER)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Inicijalizuj bazu"):
                init_database(folder_path)
                st.success("✅ Inicijalizacija završena")
        with col2:
            if st.button("🔄 Update baze iz foldera"):
                update_database(folder_path)
                st.success("✅ Update završen")

        st.divider()
        st.subheader("➕ Dodaj pojedinačni TXT fajl")
        uploaded_file = st.file_uploader("Izaberite TXT fajl", type=["txt"])
        if st.button("📥 Dodaj fajl"):
            if uploaded_file is not None:
                add_txt_file_streamlit(uploaded_file)
            else:
                st.warning("⚠️ Niste izabrali fajl.")

    # ================= TAB 2 =================


    # --- Konekcija na DuckDB ---
    con = duckdb.connect(database='kola_sk.db', read_only=False)

    st.subheader("🔍 Duplikati u tabeli kola")

    # --- Filteri ---
    filter_godina = st.text_input("Godina (YYYY)", max_chars=4, key="dupl_godina")
    filter_mesec = st.text_input("Mesec (MM)", max_chars=2, key="dupl_mesec", help="Opcionalno")

    def get_where_clause(godina, mesec=None):
        clause = f"WHERE EXTRACT(YEAR FROM DatumVreme)={godina}"
        if mesec:
            clause += f" AND EXTRACT(MONTH FROM DatumVreme)={mesec}"
        return clause

    def get_dupes_sql(godina, mesec=None):
        where_clause = get_where_clause(godina, mesec)
        sql = f"""
            WITH dupl AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY 
                           "Režim","Vlasnik","Serija","Inv br","KB","Tip kola","Voz br",
                           "Stanica","Status","Roba","Rid","UN broj","Reon",
                           "tara","NetoTone","dužina vagona","broj osovina",
                           "Otp. država","Otp st","Uputna država","Up st","Broj kola",
                           "Redni broj kola", "Datum", "Vreme"
                           ORDER BY "DatumVreme"
                       ) AS rn,
                       COUNT(*) OVER (
                           PARTITION BY 
                           "Režim","Vlasnik","Serija","Inv br","KB","Tip kola","Voz br",
                           "Stanica","Status","Roba","Rid","UN broj","Reon",
                           "tara","NetoTone","dužina vagona","broj osovina",
                           "Otp. država","Otp st","Uputna država","Up st","Broj kola",
                           "Redni broj kola", "Datum", "Vreme"
                       ) AS cnt
                FROM kola
                {where_clause}
            )
            SELECT *
            FROM dupl
            WHERE cnt > 1
            ORDER BY "Režim","Vlasnik","Serija","Inv br","DatumVreme"
        """
        return sql

    def run_sql(sql):
        return con.execute(sql).fetchdf()

    # --- Provera duplikata ---
    if st.button("🔍 Proveri duplikate"):
        if not filter_godina:
            st.warning("⚠️ Unesite godinu.")
        else:
            dupes = run_sql(get_dupes_sql(filter_godina, filter_mesec))
            if dupes.empty:
                st.success("✅ Duplikata nema")
            else:
                # Brojanje originala i duplikata
                original_count = sum(dupes['rn'] == 1)
                duplicate_count = sum(dupes['rn'] > 1)

                st.warning(f"⚠️ Pronađeno {duplicate_count} duplikata!")
                st.info(f"ℹ️ Originala: {original_count}, Duplikata:  {duplicate_count}")

                st.dataframe(dupes, use_container_width=True)
                st.session_state.dupes = dupes

    # --- Brisanje duplikata uz čuvanje originala ---
    if "dupes" in st.session_state and not st.session_state.dupes.empty:
        if st.button("🗑️ Potvrdi brisanje duplikata (originali ostaju)"):
            delete_sql = f"""
                DELETE FROM kola
                WHERE ("Režim","Vlasnik","Serija","Inv br","KB","Tip kola","Voz br",
                   "Stanica","Status","Roba","Rid","UN broj","Reon",
                   "tara","NetoTone","dužina vagona","broj osovina",
                   "Otp. država","Otp st","Uputna država","Up st","Broj kola",
                   "Redni broj kola", "Datum", "Vreme")
                IN (
                    SELECT "Režim","Vlasnik","Serija","Inv br","KB","Tip kola","Voz br",
                       "Stanica","Status","Roba","Rid","UN broj","Reon",
                       "tara","NetoTone","dužina vagona","broj osovina",
                       "Otp. država","Otp st","Uputna država","Up st","Broj kola",
                       "Redni broj kola", "Datum", "Vreme"
                    FROM (
                        SELECT *,
                               ROW_NUMBER() OVER (
                                   PARTITION BY 
                               "Režim","Vlasnik","Serija","Inv br","KB","Tip kola","Voz br",
                               "Stanica","Status","Roba","Rid","UN broj","Reon",
                               "tara","NetoTone","dužina vagona","broj osovina",
                               "Otp. država","Otp st","Uputna država","Up st","Broj kola",
                               "Redni broj kola", "Datum", "Vreme"
                                   ORDER BY "DatumVreme"
                               ) AS rn
                        FROM kola
                        {get_where_clause(filter_godina, filter_mesec)}
                    ) t
                    WHERE rn > 1
                )
            """
            con.execute(delete_sql)
            st.success(f"✅ Obrišano {sum(st.session_state.dupes['rn']>1)} duplikata, originali su sačuvani.")
            st.session_state.dupes = pd.DataFrame()




    # ================= TAB 3 =================
    with admin_tabs[2]:
        st.subheader("📄 Upload Excel tabele")
        uploaded_excel = st.file_uploader("Izaberi Excel fajl", type=["xlsx"], key="excel_upload")
        if st.button("⬆️ Upload / Update Excel tabele"):
            if uploaded_excel is not None:
                try:
                    df_excel = pd.read_excel(uploaded_excel)
                    con = get_duckdb_connection()
                    ime_tabele = uploaded_excel.name.rsplit(".", 1)[0]

                    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
                    if ime_tabele in tables:
                        con.execute(f'DROP TABLE IF EXISTS "{ime_tabele}"')
                        st.info(f"ℹ️ Postojeća tabela '{ime_tabele}' obrisana – kreiramo novu.")

                    con.register("df_excel", df_excel)
                    con.execute(f'CREATE TABLE "{ime_tabele}" AS SELECT * FROM df_excel')
                    con.unregister("df_excel")
                    st.success(f"✅ Kreirana nova tabela '{ime_tabele}' ({len(df_excel)} redova).")
                except Exception as e:
                    st.error(f"❌ Greška pri učitavanju Excel-a: {e}")
            else:
                st.warning("⚠️ Niste izabrali Excel fajl.")

    # ================= TAB 4 =================
    with admin_tabs[3]:
        st.subheader("📊 Učitanih redova po fajlu (top 20)")
        try:
            df_by_file = run_sql(
                f'''
                SELECT source_file, COUNT(*) AS broj
                FROM "{TABLE_NAME}"
                GROUP BY source_file
                ORDER BY broj DESC
                LIMIT 20
                '''
            )
            st.dataframe(df_by_file, use_container_width=True)
        except Exception as e:
            st.warning(f"Ne mogu da pročitam bazu: {e}")


# --- Broj redova i učitanih fajlova ---

st.sidebar.title("🚂 Teretna kola SK — izveštaji")
st.set_page_config(layout="wide")

# --- Custom CSS ---
st.markdown(
    """
    <style>
    div[data-testid="stSidebar"] div[role="radiogroup"] > label {
        font-size: 36px;
        padding-top: 16px;
        padding-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

tab_buttons = [
    "📊 Pregled",
    "📌 Poslednje stanje kola",
    "🔎 SQL upiti",
    "🔬 Pregled podataka",
    "📌 Kola u inostranstvu",
    "🔍 Pretraga kola",
    "📊 Kola po stanicima",
    "🚂 Kretanje kola–TIP 1",
    "🚂 Kretanje kola–TIP 0",
    "📊 Kola po serijama",
    "📊 Prosečna starost",
    "🛑 Provera grešaka po statusu",  # <-- Tab 12
    "🚂 Kretanje vozova",            # <-- Tab 13
    "📏 Km prazno/tovareno",          # <-- Tab 14 (NOVO)
    "🔧 Revizija"
]
selected_tab = st.sidebar.radio("Izaberi prikaz:", tab_buttons, index=0)

# ---------- TAB 1: Pregled ----------
if selected_tab == "📊 Pregled":
    st.subheader("📊 Pregled tabele u bazi")

    try:
        df_preview = run_sql(
            f'''
            SELECT * 
            FROM "{TABLE_NAME}" 
            ORDER BY DatumVreme DESC 
            LIMIT 50
            '''
        )
        st.dataframe(df_preview, use_container_width=True)

    except Exception as e:
        st.error(f"Greška pri čitanju baze: {e}")


# 📌 Poslednje stanje kola
if selected_tab == "📌 Poslednje stanje kola":
    st.subheader("📌 Poslednje stanje kola")  

    # 🔹 Učitaj uvek sveže stanje iz DuckDB (osvežava se posle ubacivanja novih fajlova)
    try:
        q_last = """
        WITH kola_clean AS (
            SELECT 
                *,
                TRY_CAST(SUBSTR("Broj kola", 3, LENGTH("Broj kola") - 3) AS BIGINT) AS broj_clean
            FROM kola
        ),
        poslednje AS (
            SELECT DISTINCT ON (s."Broj kola")
                s."Broj kola" AS broj_kola,
                k.*
            FROM stanje s
            LEFT JOIN kola_clean k
            ON TRY_CAST(s."Broj kola" AS BIGINT) = k.broj_clean
            ORDER BY s."Broj kola", k."DatumVreme" DESC
        )
        SELECT *
        FROM poslednje;
         df_last = run_sql(q_last)

        # 🔹 Sačuvaj i u sesiju i u DuckDB
        st.session_state.df_last = df_last
        save_last_state(df_last)
        

        st.success(f"✅ Pronađeno {len(df_last)} poslednjih unosa za kola iz Excel tabele.")
        st.dataframe(df_last, use_container_width=True)

        # Eksport Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df_last.to_excel(writer, index=False, sheet_name="Poslednje stanje")
        st.download_button(
            "⬇️ Preuzmi kao Excel", 
            data=excel_buffer.getvalue(),
            file_name="poslednje_stanje.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Greška u upitu: {e}")        
# ---------- Tab 3: SQL upiti ----------
if selected_tab == "🔎 SQL upiti":
    st.subheader("🔎 SQL upiti")
    st.subheader("Piši svoj SQL")

    default_sql = f'SELECT * FROM "{TABLE_NAME}" LIMIT 100'
    user_sql = st.text_area("SQL:", height=160, value=default_sql)

    colx, coly = st.columns([1, 3])
    run_btn = colx.button("▶️ Izvrši upit")

    def is_modifying_query(sql: str):
        keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE"]
        sql_upper = sql.strip().upper()
        return any(k in sql_upper for k in keywords)

    if run_btn:
        t0 = time.time()
        try:
            # Provera da li upit menja bazu
            if is_modifying_query(user_sql):
                password = st.text_input(
                    "⚠️ Ovom komandom menjate bazu podataka. Unesite šifru za potvrdu:",
                    type="password"
                )
                # Ako lozinka nije uneta ili je netačna, prekida izvršavanje
                if not password:
                    st.warning("❌ Unesite šifru da biste izvršili upit.")
                    st.stop()
                elif password != "IVNIZEVA":  # Ovde stavi željenu šifru
                    st.error("❌ Neispravna šifra! Upit neće biti izvršen.")
                    st.stop()

            # Ako je upit SELECT ili je lozinka tačna
            df_user = run_sql(user_sql)
            elapsed = time.time() - t0

            st.success(f"OK ({elapsed:.2f}s) — {len(df_user):,} redova".replace(",", "."))
            st.dataframe(df_user, use_container_width=True)

            if len(df_user):
                csv = df_user.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Preuzmi CSV",
                    data=csv,
                    file_name="rezultat.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Greška u upitu: {e}")
# ---------- Tab 4: Pregled podataka ----------
if selected_tab == "🔬 Pregled podataka":
    st.subheader("🔬 Pregled podataka")

    limit = st.slider("Broj redova (LIMIT)", 10, 2000, 200)

    # Sve dostupne kolone iz kola
    sve_kolone = [
        "Režim", "Vlasnik", "Serija", "Inv br", "KB",
        "Tip kola", "Voz br", "Stanica", "Status",
        "Datum", "Vreme", "Roba", "Rid", "UN broj",
        "Reon", "tara", "NetoTone", "dužina vagona",
        "broj osovina", "Otp. država", "Otp st",
        "Uputna država", "Up st", "Broj kola", "Broj vagona",
        "Redni broj kola", "source_file", "DatumVreme"
    ]

    # Dodaj kolone iz Stanje SK po izboru
    sve_kolone.extend(["PR", "NR"])  # primer dodatnih kolona

    cols = st.multiselect(
        "Kolone za prikaz",
        sve_kolone,
        default=["DatumVreme", "Voz br", "Status", "Stanica", "Broj kola", "Tip kola", "NetoTone", "tara"]
    )

    # Opcioni filteri
    godina_filter = st.selectbox("Godina", ["Sve"] + list(range(2015, 2026)))
    mesec_filter = st.selectbox("Mesec", ["Sve"] + list(range(1, 13)))
    broj_kola_filter = st.text_input("Broj kola (opciono)")

    if cols:
        try:
            # Dinamički WHERE uslovi
            where_clauses = []
            if godina_filter != "Sve":
                where_clauses.append(f'EXTRACT(YEAR FROM k."DatumVreme") = {godina_filter}')
            if mesec_filter != "Sve":
                where_clauses.append(f'EXTRACT(MONTH FROM k."DatumVreme") = {mesec_filter}')
            if broj_kola_filter:
                where_clauses.append(f'k."Broj kola" = \'{broj_kola_filter}\'')

            where_sql = f'WHERE {" AND ".join(where_clauses)}' if where_clauses else ""

            # Priprema kolona za SQL
            cols_sql = ", ".join([f'k."{c}"' if c not in ("PR", "NR") else f's."{c}"' for c in cols])

            # SQL upit sa LEFT JOIN na Stanje SK
            q = f'''
            SELECT {cols_sql}
            FROM "{TABLE_NAME}" k
            LEFT JOIN "Stanje SK" s
              ON k."Broj vagona" = s."Broj kola"
            {where_sql}
            LIMIT {int(limit)}
            '''

            df_preview = run_sql(q)
            st.dataframe(df_preview, use_container_width=True)

        except Exception as e:
            st.error(f"Greška pri čitanju: {e}")
    else:
        st.info("👉 Izaberi bar jednu kolonu za prikaz")

# ---------- Tab 5: Kola u inostranstvu ----------
# 📌 Kola u inostranstvu
if selected_tab == "📌 Kola u inostranstvu":
    st.subheader("🌍 Kola u inostranstvu")

    prikaz_tip = st.radio(
        "🔎 Izaberite prikaz:",
        ["Samo poslednje stanje", "Sva kretanja (istorija)"],
        index=0,
        horizontal=True,
        key="prikaz_tip_foreign"
    )

    if prikaz_tip == "Samo poslednje stanje":
        q_last = """
        WITH poslednje_stanje_kola AS (
                SELECT DISTINCT ON ("Broj kola")
                    k.*
                FROM kola k
                ORDER BY "Broj kola", "DatumVreme" DESC
            )
            SELECT s."Broj kola" AS broj_stanje,
                    k.*
            FROM "Stanje SK" s
            LEFT JOIN poslednje_stanje_kola k
               ON TRY_CAST(s."Broj kola" AS BIGINT) = TRY_CAST(k."Broj vagona" AS BIGINT)
            WHERE k."Status" IN (21, 24)
        """

        try:
            df_foreign = run_sql(q_last)

            st.success(f"🌍 Pronađeno {len(df_foreign)} kola u inostranstvu (poslednje stanje).")
            st.dataframe(df_foreign, use_container_width=True)

            # Export u Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df_foreign.to_excel(writer, index=False, sheet_name="Poslednje stanje")
            excel_buffer.seek(0)

            st.download_button(
                "📥 Preuzmi Excel",
                data=excel_buffer,
                file_name="kola_u_inostranstvu_poslednje_stanje.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"❌ Greška pri učitavanju kola u inostranstvu: {e}")
    # ------------------------
    # 🔹 SVA KRETANJA (ISTORIJA)
    # ------------------------
    else:
        st.markdown("<h4 style='text-align: center;'>🔎 Opcioni filteri</h4>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            broj_kola_input = st.text_input("🚃 Broj kola (opciono)", "")

        with col2:
            date_range = st.date_input("📅 Vremenski period (opciono)", [])
            start_date, end_date = None, None
            if isinstance(date_range, list) and len(date_range) == 2:
                start_date, end_date = date_range

        with col3:
            try:
                stanice = run_sql("""
                    SELECT DISTINCT "Naziv st"
                    FROM kola
                    WHERE "Naziv st" IS NOT NULL
                      AND "Naziv st" LIKE 'DRŽAVNA GRANICA%'
                    ORDER BY "Naziv st"
                """)
                stanice_list = stanice["Stanica"].dropna().astype(str).tolist()
            except:
                stanice_list = []
            granicni_prelaz = st.selectbox("🌍 Granični prelaz (opciono)", [""] + stanice_list)

        q_history = """
        SELECT 
            s."Broj kola",
            k."Broj vagona",
            k.*,
            s."status"
        FROM "Stanje SK" s
        LEFT JOIN kola k
          ON TRY_CAST(s."Broj kola" AS BIGINT) = TRY_CAST(k."Broj vagona" AS BIGINT)
        WHERE k.status IN (11, 14, 21, 24)
        ORDER BY s."Broj kola", k."DatumVreme" DESC;
        """
        try:
            df_foreign = run_sql(q_history)

            # --- Filtriranje ---
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

            st.success(f"🌍 Pronađeno {len(df_foreign)} redova (istorija kretanja).")
            st.dataframe(df_foreign, use_container_width=True)

            # Export u Excel
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df_foreign.to_excel(writer, index=False, sheet_name="Poslednje stanje")
            excel_buffer.seek(0)

            st.download_button(
                "📥 Preuzmi Excel",
                data=excel_buffer,
                file_name="kola_u_inostranstvu_poslednje_stanje.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"❌ Greška pri učitavanju kola u inostranstvu: {e}")
        # ------------------------
        # 🔹 Zadržavanje u inostranstvu
        # ------------------------
        if st.button("📊 Prikaži zadržavanje kola") and not df_foreign.empty:
            try:
                df_foreign["DatumVreme"] = pd.to_datetime(df_foreign["DatumVreme"], errors="coerce")
                df_foreign = df_foreign.sort_values(["Broj vagona", "DatumVreme"])

                retention_records = []

                for broj_vagona, grupa in df_foreign.groupby("Broj vagona"):
                    start_time = None
                    for _, row in grupa.iterrows():
                        if str(row.get("Otp. država")) == "72":
                            start_time = row["DatumVreme"]
                        elif start_time is not None and str(row.get("Uputna država")) == "72":
                            end_time = row["DatumVreme"]
                            retention = (end_time - start_time).total_seconds() / 3600
                            retention_records.append({
                                "Broj kola": broj_vagona,
                                "Datum izlaska": start_time,
                                "Datum ulaska": end_time,
                                "Zadržavanje [h]": round(retention, 2)
                            })
                            start_time = None

                df_retention = pd.DataFrame(retention_records)

                if not df_retention.empty:
                    avg_retention = df_retention["Zadržavanje [h]"].mean()
                    df_retention.loc[len(df_retention)] = {
                        "Broj kola": "📊 PROSEK",
                        "Datum izlaska": None,
                        "Datum ulaska": None,
                        "Zadržavanje [h]": round(avg_retention, 2)
                    }

                    st.success(f"✅ Pronađeno {len(df_retention)-1} parova ulaska/izlaska.")
                    st.dataframe(df_retention, use_container_width=True)

                    # Export u Excel
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                        df_foreign.to_excel(writer, index=False, sheet_name="Poslednje stanje")
                    excel_buffer.seek(0)

                    st.download_button(
                        "📥 Preuzmi Excel",
                        data=excel_buffer,
                        file_name="kola_u_inostranstvu_poslednje_stanje.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
       
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
        # --- Učitavanje stanja i opravki ---
        df_stanje = pd.read_excel("stanje SK.xlsx")
        df_opravke = pd.read_excel("Redovne opravke.xlsx")

        # Preimenuj da se ne sudara sa kolonom iz baze
        df_stanje = df_stanje[["Broj kola", "3", "PR", "NR", "TelegBaza", "Napomena"]]
        df_stanje = df_stanje.rename(columns={"3": "SerijaExcel"})

        # Poslednji unos opravke po kolu
        df_opravke["Datum revizije"] = pd.to_datetime(df_opravke["Datum revizije"], errors="coerce")
        df_opravke["Datum naredne revizije"] = pd.to_datetime(df_opravke["Datum naredne revizije"], errors="coerce")
        df_opravke = df_opravke.sort_values("Datum revizije").groupby("Broj kola").tail(1)

        # Spoji stanje i opravke
        df_stanje = df_stanje.merge(df_opravke, on="Broj kola", how="left")

        # --- Računanje TIP-a ---
        df_stanje["Datum_za_tip"] = df_stanje["Datum naredne revizije"].fillna(df_stanje["NR"])
        danas = pd.to_datetime("today").normalize()

        # Kreiramo novu pomoćnu kolonu TIP_num
        df_stanje["TIP_num"] = pd.NA
        df_stanje.loc[df_stanje["Datum_za_tip"].notna() & (df_stanje["Datum_za_tip"] < danas), "TIP_num"] = 0  # TIP 0 (istekla)
        df_stanje.loc[df_stanje["Datum_za_tip"].notna() & (df_stanje["Datum_za_tip"] >= danas), "TIP_num"] = 1  # TIP 1 (važeća)
        df_stanje["TIP_num"] = df_stanje["TIP_num"].astype("Int64")

        # Opcionalno dodajemo i tekstualnu kolonu za prikaz
        df_stanje["TIP"] = df_stanje["TIP_num"].map({0: "TIP 0", 1: "TIP 1"})
        df_stanje["TIP"] = df_stanje["TIP"].fillna("Nepoznato")

        # Registruj stanje u DuckDB
        con = get_duckdb_connection()
        con.register("stanje", df_stanje)

        # --- SQL upit za poslednje stanje po kolima ---
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
                s.TIP_num AS TIP_num,
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

        # Pivot tabela po stanici koristeći TIP_num
        df_pivot = (
            df_last.groupby(["Stanica", "NazivStanice", "TIP_num"])
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
                    .groupby(["Serija", "TIP_num"])
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

# ---------- Tab 9: Kretanje TIP 1 kola ----------
if selected_tab == "🚂 Kretanje kola–TIP 1":  
    st.subheader("🚂 Kretanje kola – TIP 1")  

    try:
        # --- Nova logika za određivanje TIP-a ---
        df_stanje = pd.read_excel("stanje SK.xlsx")
        df_opravke = pd.read_excel("Redovne opravke.xlsx")

        # Odabir i preimenovanje kolona iz stanja SK
        df_stanje = df_stanje[["Broj kola", "3", "PR", "NR", "TelegBaza", "Napomena"]]
        df_stanje = df_stanje.rename(columns={"3": "Serija"})

        # Poslednji unos po kolu iz Redovne opravke
        df_opravke["Datum revizije"] = pd.to_datetime(df_opravke["Datum revizije"], errors="coerce")
        df_opravke["Datum naredne revizije"] = pd.to_datetime(df_opravke["Datum naredne revizije"], errors="coerce")
        df_opravke = df_opravke.sort_values("Datum revizije").groupby("Broj kola").tail(1)

        # Merge tabela
        df_stanje = df_stanje.merge(df_opravke, on="Broj kola", how="left")

        # Datum za određivanje TIP-a
        df_stanje["Datum_za_tip"] = df_stanje["Datum naredne revizije"].fillna(df_stanje["NR"])
        danas = pd.to_datetime("today").normalize()

        df_stanje["TIP"] = None
        df_stanje.loc[df_stanje["Datum_za_tip"].notna() & (df_stanje["Datum_za_tip"] < danas), "TIP"] = "TIP 0 (istekla)"
        df_stanje.loc[df_stanje["Datum_za_tip"].notna() & (df_stanje["Datum_za_tip"] >= danas), "TIP"] = "TIP 1 (važeća)"
        df_stanje.loc[df_stanje["Datum_za_tip"].isna(), "TIP"] = "Nepoznato"

        # Ograničimo samo na TIP 0
        df_stanje_tip0 = df_stanje[df_stanje["TIP"] == "TIP 0 (istekla)"]

        # Registrujemo tabelu u DuckDB
        con = get_duckdb_connection()
        con.register("stanje", df_stanje_tip0)

        # --- Originalni SQL upit za kretanja ---
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
        # --- Dodavanje podataka iz Redovne opravke ---
        df_opravke = pd.read_excel("Redovne opravke.xlsx")
        df_opravke["Datum revizije"] = pd.to_datetime(df_opravke["Datum revizije"], errors="coerce")
        df_opravke["Datum naredne revizije"] = pd.to_datetime(df_opravke["Datum naredne revizije"], errors="coerce")

        # Uzimamo samo poslednju opravku po kolu
        df_opravke = df_opravke.sort_values("Datum revizije").groupby("Broj kola").tail(1)

        #Merge sa rezultatima iz DuckDB
        df_tip0 = df_tip0.merge(
            df_opravke[["Broj kola", "Datum revizije", "Datum naredne revizije"]],
            on="Broj kola",
            how="left"
        )

        # --- Formatiranje datuma ---
        for col in ["PR", "NR", "Datum revizije", "Datum naredne revizije"]:
            if col in df_tip0.columns:
                df_tip0[col] = pd.to_datetime(df_tip0[col], errors="coerce").dt.strftime("%Y-%m-%d")

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
# ---------- Tab 8: Kretanje 4098 kola – TIP 0 ----------
if selected_tab == "🚂 Kretanje kola–TIP 0":
    st.subheader("🚂 Kretanje kola – TIP 0")

    try:
        # --- Učitavanje podataka ---
        df_stanje = pd.read_excel("stanje SK.xlsx")
        df_opravke = pd.read_excel("Redovne opravke.xlsx")

        # Odabir i preimenovanje kolona iz stanja SK
        df_stanje = df_stanje[["Broj kola", "3", "PR", "NR", "TelegBaza", "Napomena"]]
        df_stanje = df_stanje.rename(columns={"3": "SerijaExcel"})  # samo da ne sudara sa onom iz baze

        # Poslednji unos po kolu iz Redovne opravke
        df_opravke["Datum revizije"] = pd.to_datetime(df_opravke["Datum revizije"], errors="coerce")
        df_opravke["Datum naredne revizije"] = pd.to_datetime(df_opravke["Datum naredne revizije"], errors="coerce")
        df_opravke = df_opravke.sort_values("Datum revizije").groupby("Broj kola").tail(1)

        # Merge sa opravkama
        df_stanje = df_stanje.merge(df_opravke, on="Broj kola", how="left")

        # Datum za određivanje TIP-a
        df_stanje["Datum_za_tip"] = df_stanje["Datum naredne revizije"].fillna(df_stanje["NR"])
        danas = pd.to_datetime("today").normalize()

        df_stanje["TIP"] = None
        df_stanje.loc[df_stanje["Datum_za_tip"].notna() & (df_stanje["Datum_za_tip"] >= danas), "TIP"] = "TIP 1 (važeća)"
        df_stanje.loc[df_stanje["Datum_za_tip"].isna(), "TIP"] = "Nepoznato"

        # Ograničimo samo na TIP 1
        df_stanje_tip1 = df_stanje[df_stanje["TIP"] == "TIP 1 (važeća)"]

        # Registrujemo tabelu u DuckDB
        con = get_duckdb_connection()
        con.register("stanje", df_stanje_tip1)

        # --- SQL upit za poslednje kretanje kola ---
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
        df_tip1 = run_sql(q)

        # Dodaj datume iz opravke
        df_tip1 = df_tip1.merge(
            df_opravke[["Broj kola", "Datum revizije", "Datum naredne revizije"]],
            on="Broj kola",
            how="left"
        )

        # --- Formatiranje datuma ---
        for col in ["PR", "NR", "Datum revizije", "Datum naredne revizije", "DatumVreme"]:
            if col in df_tip1.columns:
                df_tip1[col] = pd.to_datetime(df_tip1[col], errors="coerce").dt.strftime("%Y-%m-%d")

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

    try:
        # --- Učitavanje stanja i opravki ---
        df_stanje = pd.read_excel("stanje SK.xlsx")
        df_opravke = pd.read_excel("Redovne opravke.xlsx")

        # Preimenuj da se ne sudara sa kolonom iz baze
        df_stanje = df_stanje[["Broj kola", "3", "PR", "NR", "TelegBaza", "Napomena"]]
        df_stanje = df_stanje.rename(columns={"3": "SerijaExcel"})

        # Poslednji unos opravke po kolu
        df_opravke["Datum revizije"] = pd.to_datetime(df_opravke["Datum revizije"], errors="coerce")
        df_opravke["Datum naredne revizije"] = pd.to_datetime(df_opravke["Datum naredne revizije"], errors="coerce")
        df_opravke = df_opravke.sort_values("Datum revizije").groupby("Broj kola").tail(1)

        # Spoji stanje i opravke
        df_stanje = df_stanje.merge(df_opravke, on="Broj kola", how="left")

        # --- Računanje TIP-a ---
        df_stanje["Datum_za_tip"] = df_stanje["Datum naredne revizije"].fillna(df_stanje["NR"])
        danas = pd.to_datetime("today").normalize()

        df_stanje["TIP"] = "Nepoznato"
        df_stanje.loc[df_stanje["Datum_za_tip"].notna() & (df_stanje["Datum_za_tip"] < danas), "TIP"] = "TIP 0"
        df_stanje.loc[df_stanje["Datum_za_tip"].notna() & (df_stanje["Datum_za_tip"] >= danas), "TIP"] = "TIP 1"

        # Pomoćna numerička kolona za pivot (0 ili 1)
        df_stanje["TIP_num"] = None
        df_stanje.loc[df_stanje["TIP"] == "TIP 0", "TIP_num"] = 0
        df_stanje.loc[df_stanje["TIP"] == "TIP 1", "TIP_num"] = 1

        # Registruj stanje u DuckDB
        con = get_duckdb_connection()
        con.register("stanje", df_stanje)

        # --- SQL upit za poslednje kretanje kola ---
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

        # --- Pivot tabela po seriji, obuhvata sva kola ---
        df_all = df_stanje[["SerijaExcel", "TIP_num"]].copy()
        df_all = df_all.rename(columns={"SerijaExcel": "Serija", "TIP_num": "TIP"})

        df_pivot = (
            df_all.groupby(["Serija", "TIP"])
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

        # --- Prikaz dva dela ekrana ---
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

if "revizija_df" not in st.session_state:
    st.session_state["revizija_df"] = None
if "revizija_filtered_df" not in st.session_state:
    st.session_state["revizija_filtered_df"] = None
if "revizija_tip" not in st.session_state:
    st.session_state["revizija_tip"] = None
if "revizija_danas" not in st.session_state:
    st.session_state["revizija_danas"] = pd.to_datetime("today").normalize()
if "revizija_prikazano" not in st.session_state:
    st.session_state["revizija_prikazano"] = False
if "revizija_dana_input" not in st.session_state:
    st.session_state["revizija_dana_input"] = 30

if selected_tab == "🔧 Revizija":
    st.subheader("🔧 Revizija")

    # -----------------------
    # 🔹 Filteri u 3 kolone
    # -----------------------
    col1, col2, col3 = st.columns([1, 1.5, 1.5])

    # 🚦 TIP filter
    with col1:
        st.markdown("<h4 style='text-align: center; font-size:18px;'>🚦 Izaberi TIP</h4>", unsafe_allow_html=True)
        tip_options = ["TIP 0 (istekla)", "TIP 1 (važeća)", "Sva kola"]
        tip_filter = st.selectbox("", tip_options, index=2)

    # 🚃 Broj(evi) kola
    with col2:
        st.markdown("<h4 style='text-align: center; font-size:18px;'>🚃 Broj kola (opciono)</h4>", unsafe_allow_html=True)
        broj_kola_input = st.text_input("", value="", key="rev_broj_kola")

    # 🔎 Opcioni filteri
    with col3:
        st.markdown(
            "<h4 style='text-align: center; font-size:18px; margin-bottom:24px;'>🔎 Opcioni filteri</h4>",
            unsafe_allow_html=True
        )
        with st.expander("Izaberi", expanded=False):
            try:
                stanje = pd.read_excel("stanje SK.xlsx")
                stanje["3"] = stanje["3"].astype(str).str.zfill(3)  # obezbeđuje da ima 3 cifre
                serije = sorted(stanje["3"].dropna().unique().tolist())
                sel_serija = st.multiselect("Serija", serije)
            except:
                sel_serija = []

            try:
                radionice = pd.read_excel("Redovne opravke.xlsx")
                sel_radionica = st.multiselect("Radionica", radionice["Radionica"].dropna().unique().tolist())
            except:
                sel_radionica = []

            try:
                vrste = pd.read_excel("Redovne opravke.xlsx")
                sel_vrsta = st.multiselect("Vrsta", vrste["Vrsta"].dropna().unique().tolist())
            except:
                sel_vrsta = []

            try:
                datumi = pd.read_excel("Redovne opravke.xlsx")
                sel_datum = st.date_input("Datum revizije", [])
            except:
                sel_datum = []

    # -----------------------
    # Dugme za prikaz podataka
    # -----------------------
    if st.button("📌 Prikaži reviziju"):
        try:
            # --- Učitavanje Excel tabela ---
            df_stanje = pd.read_excel("stanje SK.xlsx")
            df_opravke = pd.read_excel("Redovne opravke.xlsx")

            # Odabir i preimenovanje kolona iz stanja SK
            df_stanje = df_stanje[["Broj kola", "3", "PR", "NR", "TelegBaza", "Napomena"]]
            df_stanje = df_stanje.rename(columns={"3": "Serija"})

            # Poslednji unos po kolu iz Redovne opravke
            df_opravke["Datum revizije"] = pd.to_datetime(df_opravke["Datum revizije"], errors="coerce")
            df_opravke["Datum naredne revizije"] = pd.to_datetime(df_opravke["Datum naredne revizije"], errors="coerce")
            df_opravke = df_opravke.sort_values("Datum revizije").groupby("Broj kola").tail(1)

            # Merge tabela
            df = df_stanje.merge(df_opravke, on="Broj kola", how="left")

            # Datum za određivanje TIP-a
            df["Datum_za_tip"] = df["Datum naredne revizije"].fillna(df["NR"])
            danas = pd.to_datetime("today").normalize()

            df["TIP"] = None
            df.loc[df["Datum_za_tip"].notna() & (df["Datum_za_tip"] < danas), "TIP"] = "TIP 0 (istekla)"
            df.loc[df["Datum_za_tip"].notna() & (df["Datum_za_tip"] >= danas), "TIP"] = "TIP 1 (važeća)"
            df.loc[df["Datum_za_tip"].isna(), "TIP"] = "Nepoznato"

            # --- Primena filtera ---
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

            # 🔹 Primena filtera po TIP-u
            if tip_filter != "Sva kola":
                df = df[df["TIP"] == tip_filter]

            # Sortiranje
            df = df.sort_values("Datum_za_tip")

            # Upis u session_state
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

    # --- Filter dana do isteka ---
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
