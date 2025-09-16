import os
import time
import duckdb
import glob
import pandas as pd
import streamlit as st
import io

import streamlit as st
import duckdb
import pandas as pd
import polars as pl
import os
import glob
import json

# ---------- Konstante ----------
DB_FILE = r"C:\Teretna kola\kola_sk.db"
TABLE_NAME = "kola"
STATE_FILE = "loaded_files.json"
DEFAULT_FOLDER = r"C:\Teretna kola"

# ---------- Helper funkcije ----------
@st.cache_data(show_spinner=False)
def run_sql(sql: str) -> pd.DataFrame:
    """Izvrši SQL nad glavnom DuckDB bazom."""
    con = duckdb.connect(DB_FILE)
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_state(processed_files):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(list(processed_files), f, indent=2)

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
                "Reon": line[61:66].strip(),
                "tara": line[78:81].strip(),
                "NetoTone": line[83:86].strip(),
                "Broj vagona": line[0:12].strip(),
                "Broj kola": line[2:11].strip(),
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

    # Brojevi u int
    df = df.with_columns([
        pl.col("tara").cast(pl.Int32, strict=False),
        pl.col("NetoTone").cast(pl.Int32, strict=False),
        pl.col("Inv br").cast(pl.Int32, strict=False),
    ])

    return df

# ---------- Inicijalizacija baze ----------
def init_database(folder: str, table_name: str = TABLE_NAME):
    files = glob.glob(os.path.join(folder, "*.txt"))
    if not files:
        st.warning(f"⚠️ Nema txt fajlova u folderu: {folder}")
        return

    all_dfs = [parse_txt(f) for f in files]
    df = pl.concat(all_dfs)

    con = duckdb.connect(DB_FILE)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.register("df", df)
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    con.unregister("df")
    con.close()

    save_state(set(files))
    st.success(f"✅ Inicijalno učitano {len(df)} redova iz {len(files)} fajlova")

# ---------- Update baze ----------
def update_database(folder: str, table_name: str = TABLE_NAME):
    processed = load_state()
    files = set(glob.glob(os.path.join(folder, "*.txt")))
    new_files = files - processed
    if not new_files:
        st.info("ℹ️ Nema novih fajlova za unos.")
        return

    con = duckdb.connect(DB_FILE)
    for f in sorted(new_files):
        df_new = parse_txt(f)
        con.register("df_new", df_new)
        con.execute(f"INSERT INTO {table_name} SELECT * FROM df_new")
        con.unregister("df_new")
        processed.add(f)
        st.write(f"➕ Ubačeno {len(df_new)} redova iz {os.path.basename(f)}")
    con.close()
    save_state(processed)
    st.success("✅ Update baze završen.")

# ---------- Dodavanje pojedinačnog fajla ----------
def add_txt_file_streamlit(uploaded_file, table_name: str = TABLE_NAME):
    if uploaded_file is None:
        st.warning("⚠️ Niste izabrali fajl.")
        return

    # Sačuvaj privremeno fajl da ga polars/duckdb može učitati
    tmp_path = os.path.join(DEFAULT_FOLDER, uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    df = parse_txt(tmp_path)
    con = duckdb.connect(DB_FILE)
    con.register("df_new", df)
    con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df_new")
    con.execute(f"INSERT INTO {table_name} SELECT * FROM df_new")
    con.unregister("df_new")
    con.close()

    st.success(f"✅ Fajl '{uploaded_file.name}' dodat u bazu ({len(df)} redova)")

# ---------- Streamlit UI ----------
st.title("🚃 Teretna kola — DuckDB")

# --- Sidebar ---
st.sidebar.title("⚙️ Podešavanja")
folder_path = st.sidebar.text_input("Folder sa TXT fajlovima", value=DEFAULT_FOLDER)

# --- Inicijalizacija / update baze ---
if st.sidebar.button("🚀 Inicijalizuj bazu"):
    init_database(folder_path)

if st.sidebar.button("➕ Update baze iz foldera"):
    update_database(folder_path)

# --- Dodavanje pojedinačnog fajla ---
st.subheader("➕ Dodaj TXT fajl u bazu")
uploaded_file = st.file_uploader("Izaberite TXT fajl", type=["txt"])
if st.button("Dodaj fajl"):
    add_txt_file_streamlit(uploaded_file)

# --- Pregled tabele ---
st.subheader("📊 Pregled tabele u bazi")
try:
    df_preview = run_sql(f'SELECT * FROM {TABLE_NAME} LIMIT 20')
    st.dataframe(df_preview, use_container_width=True)
except Exception as e:
    st.error(f"Greška pri čitanju baze: {e}")

# --- Broj redova i učitanih fajlova ---
st.subheader("ℹ️ Status baze")
try:
    total_rows = run_sql(f"SELECT COUNT(*) AS broj_redova FROM {TABLE_NAME}").iloc[0,0]
    loaded_files = len(load_state())
    st.write(f"Ukupan broj redova: {total_rows}")
    st.write(f"Učitanih fajlova: {loaded_files}")
except Exception as e:
    st.warning(f"Ne mogu da pročitam bazu: {e}")


tabs = st.tabs([
    "📊 Pregled", "📈 Izveštaji", "🔎 SQL upiti", "🔬 Pregled podataka",
    "📌 Poslednji unosi", "🔍 Pretraga kola", "📊 Kola po stanicima",
    "🚂 Kretanje 4098 kola–TIP 0", "🚂 Kretanje 4098 kola–TIP 1", "📊 Kola po serijama"
])

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = tabs

# ---------- Tab 1: Pregled ----------
with tab1:
    col_a, col_b, col_c, col_d = st.columns(4)
    try:
        df_cnt = run_sql(f'SELECT COUNT(*) AS broj_redova FROM main.{TABLE_NAME}')
        col_a.metric("Ukupan broj redova", f"{int(df_cnt['broj_redova'][0]):,}".replace(",", "."))

        df_files = run_sql(f'SELECT COUNT(DISTINCT source_file) AS fajlova FROM main.{TABLE_NAME}')
        col_b.metric("Učitanih fajlova", int(df_files["fajlova"][0]))

        df_range = run_sql(f'SELECT MIN(DatumVreme) AS min_dt, MAX(DatumVreme) AS max_dt FROM main.{TABLE_NAME}')
        min_dt = str(df_range["min_dt"][0]) if df_range["min_dt"][0] is not None else "—"
        max_dt = str(df_range["max_dt"][0]) if df_range["max_dt"][0] is not None else "—"
        col_c.metric("Najraniji datum", min_dt)
        col_d.metric("Najkasniji datum", max_dt)

        st.divider()
        st.subheader("Učitanih redova po fajlu (top 20)")
        df_by_file = run_sql(f'''
            SELECT source_file, COUNT(*) AS broj
            main.{TABLE_NAME}
            GROUP BY source_file
            ORDER BY broj DESC
            LIMIT 20
        ''')
        st.dataframe(df_by_file, use_container_width=True)
    except Exception as e:
        st.error(f"Ne mogu da pročitam bazu: {e}")

# ---------- Tab 2: Izveštaji ----------
with tab2:
    try:
        st.subheader("Suma NetoTone po mesecu")
        df_month = run_sql(f'''
            SELECT date_trunc('month', DatumVreme) AS mesec,
                   SUM(COALESCE("NetoTone",0)) AS ukupno_tona
            main.{TABLE_NAME}
            WHERE DatumVreme IS NOT NULL
            GROUP BY 1 ORDER BY 1
        ''')
        st.line_chart(df_month.set_index("mesec")["ukupno_tona"])

        st.subheader("Top 20 stanica po broju vagona")
        df_sta = run_sql(f'''
            SELECT "Stanica", COUNT(*) AS broj
            FROM "{TABLE_NAME}"
            GROUP BY "Stanica"
            ORDER BY broj DESC
            LIMIT 20
        ''')
        st.bar_chart(df_sta.set_index("Stanica")["broj"])
    except Exception as e:
        st.error(f"Greška u izveštajima: {e}")

# ---------- Tab 3: SQL upiti ----------
with tab3:
    st.subheader("Piši svoj SQL")
    default_sql = f'SELECT * FROM "{TABLE_NAME}" LIMIT 100'
    user_sql = st.text_area("SQL:", height=160, value=default_sql)
    run_btn = st.button("▶️ Izvrši upit", key="sql_btn")
    if run_btn:
        try:
            t0 = time.time()
            df_user = run_sql(user_sql)
            st.success(f"OK ({time.time()-t0:.2f}s) — {len(df_user):,} redova".replace(",", "."))
            st.dataframe(df_user, use_container_width=True)
            if len(df_user):
                st.download_button("⬇️ Preuzmi CSV", df_user.to_csv(index=False).encode("utf-8"), "rezultat.csv")
        except Exception as e:
            st.error(f"Greška u upitu: {e}")

# ---------- Tab 4: Pregled podataka ----------
with tab4:
    st.subheader("Brzi pregled")
    limit = st.slider("Broj redova (LIMIT)", 10, 2000, 200)
    cols = st.multiselect(
        "Kolone",
        ["Režim","Vlasnik","Serija","Inv br","KB","Tip kola","Voz br","Stanica",
         "Status","Datum","Vreme","Roba","Reon","tara","NetoTone","Broj vagona",
         "Broj kola","source_file","DatumVreme"],
        default=["DatumVreme","Stanica","Tip kola","NetoTone","tara","source_file"]
    )
    if cols:
        cols_sql = ", ".join([f'"{c}"' for c in cols])
        try:
            df_preview = run_sql(f'SELECT {cols_sql} FROM "{TABLE_NAME}" LIMIT {limit}')
            st.dataframe(df_preview, use_container_width=True)
        except Exception as e:
            st.error(f"Greška pri čitanju: {e}")

# ---------- Tab 5: Poslednji unosi ----------
with tab5:
    st.subheader("📌 Poslednji unos za 4098 kola iz Excel tabele")
    if st.button("🔎 Prikaži poslednje unose", key="tab5_btn"):
        try:
            df_last = run_sql(f'''
                SELECT s.SerijaIpodserija, k.*
                FROM "{excel_table}" s
                JOIN "{TABLE_NAME}" k
                  ON CAST(s.SerijaIpodserija AS TEXT) = REPLACE(k.broj_kola_bez_rezima_i_kb,' ','')
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY s.SerijaIpodserija ORDER BY k.DatumVreme DESC
                ) = 1
            ''')
            st.success(f"✅ Pronađeno {len(df_last)} poslednjih unosa.")
            st.dataframe(df_last, use_container_width=True)
        except Exception as e:
            st.error(f"Greška u upitu: {e}")

# ---------- Tab 6: Pretraga kola ----------
with tab6:
    st.subheader("🔍 Pretraga kola po broju i periodu")
    broj_kola_input = st.text_input("Unesi broj kola (ili deo broja)", key="pretraga_input")
    start_date = st.date_input("📅 Od datuma", key="start_date")
    end_date = st.date_input("📅 Do datuma", key="end_date")
    if st.button("🔎 Pretraži", key="pretraga_btn"):
        try:
            df_search = run_sql(f'''
                SELECT *
                FROM "{TABLE_NAME}"
                WHERE "Broj kola" LIKE '%{broj_kola_input}%'
                  AND "DatumVreme" BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY "DatumVreme" DESC
            ''')
            if df_search.empty:
                st.warning("⚠️ Nema podataka za zadate kriterijume.")
            else:
                st.success(f"✅ Pronađeno {len(df_search)} redova.")
                st.dataframe(df_search, use_container_width=True)
        except Exception as e:
            st.error(f"Greška u upitu: {e}")

# ---------- Tab 7: Kola po stanicima ----------
with tab7:
    st.subheader("📊 Kola po stanicima")
    try:
        df_last = run_sql(f'''
            WITH poslednji AS (
                SELECT k.Serija, k.Stanica, s.TIP, ROW_NUMBER() OVER (
                    PARTITION BY k.broj_kola_bez_rezima_i_kb ORDER BY k.DatumVreme DESC
                ) AS rn
                FROM "{TABLE_NAME}" k
                JOIN "{excel_table}" s
                  ON k.broj_kola_bez_rezima_i_kb = s.SerijaIpodserija
            )
            SELECT *
            FROM poslednji WHERE rn=1
        ''')
        st.dataframe(df_last, use_container_width=True)
    except Exception as e:
        st.error(f"Greška: {e}")


# ---------- Tab 8: Kretanje 4098 kola – TIP 0 ----------
with tab8:
    st.subheader("🚂 Kretanje 4098 kola – samo TIP 0")
    try:
        # Proveri da li postoje potrebne tabele
        tables = run_sql("SELECT TABLE_NAME FROM duckdb_tables()")
        needed_tables = [TABLE_NAME, excel_table, "stanice"]
        missing = [t for t in needed_tables if t not in tables["TABLE_NAME"].tolist()]
        if missing:
            st.warning(f"⚠️ Nedostaju tabele: {', '.join(missing)}. Ne mogu da prikažem tab.")
        else:
            df_tip0 = run_sql(f'''
                WITH poslednji AS (
                    SELECT 
                        s.SerijaIpodserija,
                        s.TIP,
                        s.TelegBaza,
                        s.PR,
                        s.NR,
                        k.Serija,
                        k.Stanica,
                        st.Naziv AS NazivStanice,
                        k.DatumVreme,
                        ROW_NUMBER() OVER (
                            PARTITION BY s.SerijaIpodserija
                            ORDER BY k.DatumVreme DESC
                        ) AS rn
                    FROM "{excel_table}" s
                    LEFT JOIN "{TABLE_NAME}" k
                      ON TRIM(k.broj_kola_bez_rezima_i_kb) = TRIM(s.SerijaIpodserija)
                    LEFT JOIN stanice st
                      ON k.Stanica = st.Sifra
                    WHERE s.TIP = 0
                )
                SELECT *
                FROM poslednji
                WHERE rn = 1 OR rn IS NULL
                ORDER BY DatumVreme ASC
            ''')

            if "DatumVreme" in df_tip0.columns:
                df_tip0["BrojDana"] = (pd.Timestamp.now() - pd.to_datetime(df_tip0["DatumVreme"], errors='coerce')).dt.days

            # Filter po seriji i stanici
            series_options = ["Sve serije"] + sorted(df_tip0["Serija"].dropna().unique().tolist())
            selected_series = st.selectbox("🚆 Filtriraj po seriji kola (TIP 0)", series_options, key="tip0_series")
            if selected_series != "Sve serije":
                df_tip0 = df_tip0[df_tip0["Serija"] == selected_series]

            station_options = ["Sve stanice"] + sorted(df_tip0["NazivStanice"].dropna().unique().tolist())
            selected_station = st.selectbox("📍 Filtriraj po stanici (TIP 0)", station_options, key="tip0_station")
            if selected_station != "Sve stanice":
                df_tip0 = df_tip0[df_tip0["NazivStanice"] == selected_station]

            st.dataframe(df_tip0, use_container_width=True)

            # Download CSV / Excel
            c1, c2 = st.columns(2)
            with c1:
                csv = df_tip0.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Preuzmi tabelu (CSV)", csv, "tip0_kretanje.csv", "text/csv")
            with c2:
                excel_bytes = io.BytesIO()
                with pd.ExcelWriter(excel_bytes, engine="openpyxl") as writer:
                    df_tip0.to_excel(writer, sheet_name="TIP0", index=False)
                st.download_button("⬇️ Preuzmi tabelu (Excel)", excel_bytes.getvalue(), "tip0_kretanje.xlsx")

    except Exception as e:
        st.error(f"Greška: {e}")

# ---------- Tab 9: Kretanje 4098 kola – TIP 1 ----------
with tab9:
    st.subheader("🚂 Kretanje 4098 kola – samo TIP 1")
    try:
        # Provera postojanja tabela
        tables = run_sql("SELECT TABLE_NAME FROM duckdb_tables()")
        missing = [t for t in [TABLE_NAME, excel_table, "stanice"] if t not in tables["TABLE_NAME"].tolist()]
        if missing:
            st.warning(f"⚠️ Nedostaju tabele: {', '.join(missing)}. Ne mogu da prikažem tab.")
        else:
            df_tip1 = run_sql(f'''
                WITH poslednji AS (
                    SELECT 
                        s.SerijaIpodserija,
                        s.TIP,
                        s.TelegBaza,
                        s.PR,
                        s.NR,
                        k.Serija,
                        k.Stanica,
                        st.Naziv AS NazivStanice,
                        k.DatumVreme,
                        ROW_NUMBER() OVER (
                            PARTITION BY s.SerijaIpodserija
                            ORDER BY k.DatumVreme DESC
                        ) AS rn
                    FROM "{excel_table}" s
                    LEFT JOIN "{TABLE_NAME}" k
                      ON TRIM(k.broj_kola_bez_rezima_i_kb) = TRIM(s.SerijaIpodserija)
                    LEFT JOIN stanice st
                      ON k.Stanica = st.Sifra
                    WHERE s.TIP = 1
                )
                SELECT *
                FROM poslednji
                WHERE rn = 1 OR rn IS NULL
                ORDER BY DatumVreme DESC
            ''')

            if "DatumVreme" in df_tip1.columns:
                df_tip1["BrojDana"] = (pd.Timestamp.now() - pd.to_datetime(df_tip1["DatumVreme"], errors='coerce')).dt.days

            # Filter po seriji i stanici
            series_options = ["Sve serije"] + sorted(df_tip1["Serija"].dropna().unique().tolist())
            selected_series = st.selectbox("🚆 Filtriraj po seriji kola (TIP 1)", series_options, key="tip1_series")
            if selected_series != "Sve serije":
                df_tip1 = df_tip1[df_tip1["Serija"] == selected_series]

            station_options = ["Sve stanice"] + sorted(df_tip1["NazivStanice"].dropna().unique().tolist())
            selected_station = st.selectbox("📍 Filtriraj po stanici (TIP 1)", station_options, key="tip1_station")
            if selected_station != "Sve stanice":
                df_tip1 = df_tip1[df_tip1["NazivStanice"] == selected_station]

            st.dataframe(df_tip1, use_container_width=True)

            # Download CSV / Excel
            c1, c2 = st.columns(2)
            with c1:
                csv = df_tip1.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Preuzmi tabelu (CSV)", csv, "tip1_kretanje.csv", "text/csv")
            with c2:
                excel_bytes = io.BytesIO()
                with pd.ExcelWriter(excel_bytes, engine="openpyxl") as writer:
                    df_tip1.to_excel(writer, sheet_name="TIP1", index=False)
                st.download_button("⬇️ Preuzmi tabelu (Excel)", excel_bytes.getvalue(), "tip1_kretanje.xlsx")

    except Exception as e:
        st.error(f"Greška: {e}")

# ---------- Tab 10: Pivot po serijama i stanicama ----------
with tab10:
    st.subheader("📊 Pivot po seriji i stanicama")
    try:
        tables = run_sql("SELECT TABLE_NAME FROM duckdb_tables()")
        missing = [t for t in [TABLE_NAME, excel_table] if t not in tables["TABLE_NAME"].tolist()]
        if missing:
            st.warning(f"⚠️ Nedostaju tabele: {', '.join(missing)}. Ne mogu da prikažem tab.")
        else:
            df_last = run_sql(f'''
                WITH poslednji AS (
                    SELECT k.Serija, k.Stanica, s.TIP, ROW_NUMBER() OVER (
                        PARTITION BY k.broj_kola_bez_rezima_i_kb
                        ORDER BY k.DatumVreme DESC
                    ) AS rn
                    FROM "{TABLE_NAME}" k
                    JOIN "{excel_table}" s
                      ON k.broj_kola_bez_rezima_i_kb = s.SerijaIpodserija
                )
                SELECT *
                FROM poslednji WHERE rn=1
            ''')

            # Pivot tabela po seriji
            df_pivot = df_last.pivot_table(index='Serija', columns='TIP', aggfunc='size', fill_value=0).reset_index()
            df_pivot = df_pivot.rename(columns={0: "tip0", 1: "tip1"})
            df_pivot["Ukupno"] = df_pivot.get("tip0",0) + df_pivot.get("tip1",0)

            # Red Σ
            total_row = pd.DataFrame([{
                "Serija": "Σ",
                "tip0": df_pivot["tip0"].sum(),
                "tip1": df_pivot["tip1"].sum(),
                "Ukupno": df_pivot["Ukupno"].sum()
            }])
            df_pivot = pd.concat([df_pivot, total_row], ignore_index=True)

            left, right = st.columns([1,1])
            with left:
                st.markdown("### 📋 Ukupan broj kola po serijama")
                st.dataframe(df_pivot, use_container_width=True)

            with right:
                series_list = df_pivot[df_pivot["Serija"]!="Σ"]["Serija"].tolist()
                selected_series = st.selectbox("Izaberi seriju za detalje", ["Nijedna"] + series_list, key="pivot_series")
                if selected_series != "Nijedna":
                    df_detail = df_last[df_last["Serija"]==selected_series].pivot_table(
                        index='Stanica', columns='TIP', aggfunc='size', fill_value=0
                    ).reset_index().rename(columns={0:"tip0",1:"tip1"})
                    df_detail["Ukupno"] = df_detail.get("tip0",0) + df_detail.get("tip1",0)
                    total = pd.DataFrame([{
                        "Stanica": "Σ",
                        "tip0": df_detail["tip0"].sum(),
                        "tip1": df_detail["tip1"].sum(),
                        "Ukupno": df_detail["Ukupno"].sum()
                    }])
                    df_detail = pd.concat([df_detail,total], ignore_index=True)
                    st.dataframe(df_detail, use_container_width=True)

    except Exception as e:
        st.error(f"Greška: {e}")


