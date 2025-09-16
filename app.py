import os
import time
import duckdb
import glob
import pandas as pd
import streamlit as st
import io

# ---------- Konstante ----------
DB_FILE = r"C:\Teretna kola\kola_sk.db"
TABLE_NAME = "kola"

# ---------- Funkcija za dodavanje pojedinačnog fajla ----------
def add_txt_file(uploaded_file, table_name=TABLE_NAME):
    if uploaded_file is None:
        st.warning("⚠️ Niste izabrali fajl.")
        return

    try:
        df = pd.read_csv(uploaded_file, sep="\t")
        df["source_file"] = uploaded_file.name

        con = duckdb.connect(DB_FILE)

        # Ako tabela ne postoji, kreiraj je
        if table_name not in [r[0] for r in con.execute("SHOW TABLES").fetchall()]:
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        else:
            con.execute(f"INSERT INTO {table_name} SELECT * FROM df")

        con.close()
        st.success(f"✅ Fajl '{uploaded_file.name}' je dodat u bazu ({len(df)} redova).")
    except Exception as e:
        st.error(f"❌ Greška pri dodavanju fajla: {e}")

# ---------- Funkcija za update baze iz foldera ----------
def update_database(folder_path, table_name=TABLE_NAME):
    txt_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
    if not txt_files:
        st.warning("⚠️ Nema TXT fajlova u folderu.")
        return

    con = duckdb.connect(DB_FILE)
    try:
        loaded_files = set()
        if table_name in [r[0] for r in con.execute("SHOW TABLES").fetchall()]:
            try:
                loaded_files = set(
                    con.execute(f"SELECT DISTINCT source_file FROM {table_name}").fetchdf()["source_file"].tolist()
                )
            except Exception:
                pass

        for f in txt_files:
            fname = os.path.basename(f)
            if fname in loaded_files:
                continue

            df = pd.read_csv(f, sep="\t")
            df["source_file"] = fname
            if table_name not in [r[0] for r in con.execute("SHOW TABLES").fetchall()]:
                con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
            else:
                con.execute(f"INSERT INTO {table_name} SELECT * FROM df")
            loaded_files.add(fname)

        st.success("✅ Update baze završen.")
    finally:
        con.close()

# ---------- Streamlit UI ----------
st.title("🚃 Teretna kola — DuckDB")

# --- Dodavanje pojedinačnog fajla ---
st.subheader("➕ Dodaj TXT fajl u bazu")
uploaded_file = st.file_uploader("Izaberite TXT fajl", type=["txt"])
if st.button("Dodaj fajl u bazu"):
    add_txt_file(uploaded_file)

# --- Sidebar za update baze ---
st.sidebar.title("⚙️ Podešavanja")

# Folder sa TXT fajlovima
folder_path = st.sidebar.text_input("Folder sa TXT fajlovima", value=r"C:\Teretna kola", key="folder_path")

# Dugme za update baze iz foldera
if st.sidebar.button("➕ Update baze iz foldera", key="update_button"):
    update_database(folder_path)

# Excel upload
st.sidebar.subheader("📂 Uvoz Excela (Stanje SK)")
uploaded_excel = st.sidebar.file_uploader("Izaberi Excel fajl (.xlsx)", type=["xlsx"], key="excel_uploader")
if uploaded_excel and st.sidebar.button("📥 Učitaj u bazu", key="excel_button"):
    try:
        df_stanje = pd.read_excel(uploaded_excel)
        con = duckdb.connect(DB_FILE)
        con.register("df_stanje", df_stanje)
        con.execute("CREATE OR REPLACE TABLE stanje AS SELECT * FROM df_stanje")
        con.unregister("df_stanje")
        con.close()
        st.success(f"✅ Excel učitan u tabelu 'stanje' ({len(df_stanje)} redova).")
    except Exception as e:
        st.error(f"❌ Greška pri uvozu Excela: {e}")

# ---------- Akcije ----------
if st.sidebar.button("➕ Update baze iz foldera"):
    update_database(folder_path)

# ---------- Streamlit dashboard ----------
st.title("🚃 Teretna kola SK — kontrolna tabla")
try:
    df_cnt = run_sql(f'SELECT COUNT(*) AS broj_redova FROM {TABLE_NAME}')
    st.metric("Ukupan broj redova", f"{int(df_cnt['broj_redova'][0]):,}".replace(",", "."))
except Exception as e:
    st.error(f"Ne mogu da pročitam bazu: {e}")
# ---------- Streamlit dashboard ----------
st.title("🚃 Teretna kola SK — kontrolna tabla")

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
        df_cnt = run_sql(f'SELECT COUNT(*) AS broj_redova FROM main.{table_name}')
        col_a.metric("Ukupan broj redova", f"{int(df_cnt['broj_redova'][0]):,}".replace(",", "."))

        df_files = run_sql(f'SELECT COUNT(DISTINCT source_file) AS fajlova FROM main.{table_name}')
        col_b.metric("Učitanih fajlova", int(df_files["fajlova"][0]))

        df_range = run_sql(f'SELECT MIN(DatumVreme) AS min_dt, MAX(DatumVreme) AS max_dt FROM main.{table_name}')
        min_dt = str(df_range["min_dt"][0]) if df_range["min_dt"][0] is not None else "—"
        max_dt = str(df_range["max_dt"][0]) if df_range["max_dt"][0] is not None else "—"
        col_c.metric("Najraniji datum", min_dt)
        col_d.metric("Najkasniji datum", max_dt)

        st.divider()
        st.subheader("Učitanih redova po fajlu (top 20)")
        df_by_file = run_sql(f'''
            SELECT source_file, COUNT(*) AS broj
            main.{table_name}
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
            main.{table_name}
            WHERE DatumVreme IS NOT NULL
            GROUP BY 1 ORDER BY 1
        ''')
        st.line_chart(df_month.set_index("mesec")["ukupno_tona"])

        st.subheader("Top 20 stanica po broju vagona")
        df_sta = run_sql(f'''
            SELECT "Stanica", COUNT(*) AS broj
            FROM "{table_name}"
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
    default_sql = f'SELECT * FROM "{table_name}" LIMIT 100'
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
            df_preview = run_sql(f'SELECT {cols_sql} FROM "{table_name}" LIMIT {limit}')
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
                JOIN "{table_name}" k
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
                FROM "{table_name}"
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
                FROM "{table_name}" k
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
        tables = run_sql("SELECT table_name FROM duckdb_tables()")
        needed_tables = [table_name, excel_table, "stanice"]
        missing = [t for t in needed_tables if t not in tables["table_name"].tolist()]
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
                    LEFT JOIN "{table_name}" k
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
        tables = run_sql("SELECT table_name FROM duckdb_tables()")
        missing = [t for t in [table_name, excel_table, "stanice"] if t not in tables["table_name"].tolist()]
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
                    LEFT JOIN "{table_name}" k
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
        tables = run_sql("SELECT table_name FROM duckdb_tables()")
        missing = [t for t in [table_name, excel_table] if t not in tables["table_name"].tolist()]
        if missing:
            st.warning(f"⚠️ Nedostaju tabele: {', '.join(missing)}. Ne mogu da prikažem tab.")
        else:
            df_last = run_sql(f'''
                WITH poslednji AS (
                    SELECT k.Serija, k.Stanica, s.TIP, ROW_NUMBER() OVER (
                        PARTITION BY k.broj_kola_bez_rezima_i_kb
                        ORDER BY k.DatumVreme DESC
                    ) AS rn
                    FROM "{table_name}" k
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


