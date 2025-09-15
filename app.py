import os
import time
import duckdb
import glob
import io
import pandas as pd
import streamlit as st
import requests

# --- Fiksna glavna baza (Google Drive) ---
DB_FILE = "kola_sk.db" STATE_FILE = "loaded_files.json"

# ---------- Putanja do baze ----------
DB_PATH = r"C:\Teretna kola\kola.duckdb"
# ---------- Helperi ----------
@st.cache_data(show_spinner=False)
def get_tables(db_path: str):
    con = duckdb.connect(db_path)
    try:
        return [r[0] for r in con.execute(
            "SELECT table_name FROM duckdb_tables() WHERE database_name IS NULL"
        ).fetchall()]
    finally:
        con.close()

def run_sql(sql: str) -> pd.DataFrame:
    """Izvrši upit nad UNION bazom (glavna + update)."""
    con = duckdb.connect()
    con.execute(f"ATTACH '{MAIN_DB}' AS main")
    if os.path.exists(UPDATE_DB):
        con.execute(f"ATTACH '{UPDATE_DB}' AS upd")

    # kreiramo view da uvek imamo objedinjene podatke
    q_union = """
        CREATE OR REPLACE VIEW kola_union AS
        SELECT * FROM main.kola
        UNION ALL
        SELECT * FROM upd.kola_update
    """
    if "kola" in get_tables(MAIN_DB):
        if os.path.exists(UPDATE_DB) and "kola_update" in get_tables(UPDATE_DB):
            con.execute(q_union)
        else:
            con.execute("CREATE OR REPLACE VIEW kola_union AS SELECT * FROM main.kola")

    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()
# ---------- Sidebar ----------
st.sidebar.title("⚙️ Podešavanja")
db_path = st.sidebar.text_input(
    "Putanja do DuckDB baze (.db)",
    value = os.path.abspath(MAIN_DB)
)

folder_path = st.sidebar.text_input(
    "Folder sa TXT fajlovima (npr. C:\\Teretna kola)",
    value=r"C:\Teretna kola"
)

table_name = st.sidebar.text_input("Ime tabele", value="kola")

st.sidebar.markdown("---")
st.sidebar.caption("Napomena: `init_database` pokreće **prvo punjenje**. Posle toga koristi `update`.")

init_clicked = st.sidebar.button("🚀 Init database (prvo punjenje)")
update_clicked = st.sidebar.button("➕ Update (dodaj nove fajlove)")

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Ponovno učitavanje fajlova")

# Učitaj sve txt fajlove iz foldera
txt_files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))

if txt_files:
    reload_files = st.sidebar.multiselect(
        "Izaberi fajlove za reload",
        options=txt_files,
        format_func=lambda x: os.path.basename(x)
    )
    if st.sidebar.button("🔁 Reload označenih fajlova"):
        for f in reload_files:
            try:
                reload_file(f, table_name=table_name)
                st.success(f"🔄 Fajl {os.path.basename(f)} uspešno ponovo učitan!")
            except Exception as e:
                st.error(f"⚠️ Greška pri reload-u {os.path.basename(f)}: {e}")
else:
    st.sidebar.warning("⚠️ Nema txt fajlova u izabranom folderu.")

st.sidebar.markdown("---")
st.sidebar.caption("Baza i fajlovi ostaju na disku; aplikacija samo izvršava SQL i poziva operacije iz database.py.")
# ---------- Sidebar: Upload Excela ----------
st.sidebar.subheader("📂 Uvoz Excela (Stanje SK)")

uploaded_excel = st.sidebar.file_uploader(
    "Izaberi Excel fajl (.xlsx)",
    type=["xlsx"]
)

if uploaded_excel is not None:
    if st.sidebar.button("📥 Učitaj u bazu"):
        try:
            # 1. Učitaj Excel u pandas
            df_stanje = pd.read_excel(uploaded_excel)

            # 2. Poveži se na DuckDB
            con = duckdb.connect(DB_FILE)

            # 3. Snimi kao tabelu 'stanje'
            con.register("df_stanje", df_stanje)
            con.execute("CREATE OR REPLACE TABLE stanje AS SELECT * FROM df_stanje")
            con.unregister("df_stanje")

            # 4. Provera kolona
            cols = con.execute("PRAGMA table_info(stanje)").fetchdf()
            con.close()

            st.success(f"✅ Excel učitan u tabelu 'stanje' ({len(df_stanje)} redova).")
            st.dataframe(cols, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Greška pri uvozu Excela: {e}")

# ---------- Akcije ----------
if init_clicked:
    def _init():
        init_database(folder_path, table_name=table_name)
        default_abs = os.path.abspath(DEFAULT_DB)
        if os.path.abspath(db_path) != default_abs and os.path.exists(default_abs):
            try:
                if os.path.exists(db_path):
                    os.remove(db_path)
                os.replace(default_abs, db_path)
            except Exception as e:
                raise RuntimeError(f"Nisam uspeo da premestim {default_abs} → {db_path}: {e}")
    safe_execute(_init, "✅ Inicijalno punjenje završeno.")

if update_clicked:
    def _update():
        update_database(folder_path, table_name=table_name)
    safe_execute(_update, "✅ Update završen.")

st.title("🚃 Teretna kola SK — kontrolna tabla")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Pregled", "📈 Izveštaji", "🔎 SQL upiti", "🔬 Pregled podataka", "📌 Poslednji unosi", "🔍 Pretraga kola", "📊 Kola po stanicima", "🚂 Kretanje 4098 kola–TIP 0", "🚂 Kretanje 4098 kola–TIP 1", "📊 Kola po serijama"])

# ---------- Tab 1: Pregled ----------
with tab1:
    col_a, col_b, col_c, col_d = st.columns(4)
    try:
        df_cnt = run_sql(db_path, f'SELECT COUNT(*) AS broj_redova FROM "{table_name}"')
        col_a.metric("Ukupan broj redova", f"{int(df_cnt['broj_redova'][0]):,}".replace(",", "."))

        df_files = run_sql(db_path, f'SELECT COUNT(DISTINCT source_file) AS fajlova FROM "{table_name}"')
        col_b.metric("Učitanih fajlova", int(df_files["fajlova"][0]))

        df_range = run_sql(
            db_path,
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
            db_path,
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
    df_month = run_sql(db_path, q_month)
    st.line_chart(df_month.set_index("mesec")["ukupno_tona"])

    st.subheader("Top 20 stanica po broju vagona")
    q_sta = f"""
        SELECT "Stanica", COUNT(*) AS broj
        FROM "{table_name}"
        GROUP BY "Stanica"
        ORDER BY broj DESC
        LIMIT 20
    """
    df_sta = run_sql(db_path, q_sta)
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
        df_tip = run_sql(db_path, q_tip)
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
        df_tara = run_sql(db_path, q_tara)
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
            df_user = run_sql(db_path, user_sql)
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
        df_preview = run_sql(db_path, f'SELECT {cols_sql} FROM "{table_name}" LIMIT {int(limit)}')
        st.dataframe(df_preview, use_container_width=True)
    except Exception as e:
        st.error(f"Greška pri čitanju: {e}")
# ---------- Tab 5: Poslednje stanje kola ----------

# ---------- Tab 5: Poslednji unosi ----------
with tab5:
    st.subheader("📌 Poslednji unos za 4098 kola iz Excel tabele")

    if st.button("🔎 Prikaži poslednje unose"):
        try:
            q_last = f"""
                SELECT s.SerijaIpodserija, k.*
                FROM stanje s
                JOIN "{table_name}" k
                  ON CAST(s.SerijaIpodserija AS TEXT) = REPLACE(k.broj_kola_bez_rezima_i_kb, ' ', '')
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY s.SerijaIpodserija
                    ORDER BY k.DatumVreme DESC
                ) = 1
            """
            df_last = run_sql(db_path, q_last)
            st.success(f"✅ Pronađeno {len(df_last)} poslednjih unosa za kola iz Excel tabele.")
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
            df_search = run_sql(db_path, q_search)

            if df_search.empty:
                st.warning("⚠️ Nema podataka za zadate kriterijume.")
            else:
                st.success(f"✅ Pronađeno {len(df_search)} redova.")
                st.dataframe(df_search, use_container_width=True)

        except Exception as e:
            st.error(f"Greška u upitu: {e}")
with tab7:
    st.subheader("📊 Kola po stanicima")

    try:
        q = """
        WITH poslednji AS (
            SELECT 
                k.Serija,
                k.Stanica,
                st.Naziv AS NazivStanice,
                s.TIP,
                ROW_NUMBER() OVER (
                    PARTITION BY k.broj_kola_bez_rezima_i_kb
                    ORDER BY k.DatumVreme DESC
                ) AS rn
            FROM kola k
            JOIN stanje s 
              ON TRIM(CAST(k.broj_kola_bez_rezima_i_kb AS VARCHAR)) = TRIM(CAST(s.SerijaIpodserija AS VARCHAR))
            LEFT JOIN stanice st
              ON TRIM(CAST(k.Stanica AS VARCHAR)) = TRIM(CAST(st.Sifra AS VARCHAR))
        )
        SELECT *
        FROM poslednji
        WHERE rn = 1
        """
        df_last = run_sql(db_path, q)

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
        st.error(f"Greška: {e}")


with tab8:
    st.subheader("🚂 Kretanje 4098 kola – samo TIP 0")

    try:
        q = """
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
            FROM stanje s
            LEFT JOIN kola k 
              ON TRIM(CAST(s.SerijaIpodserija AS VARCHAR)) = TRIM(CAST(k.broj_kola_bez_rezima_i_kb AS VARCHAR))
            LEFT JOIN stanice st
              ON TRIM(CAST(k.Stanica AS VARCHAR)) = TRIM(CAST(st.Sifra AS VARCHAR))
            WHERE s.TIP = 0
        )
        SELECT 
            SerijaIpodserija,
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
        df_tip0 = run_sql(db_path, q)

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

        # Prikaz tabele
        st.dataframe(df_tip0, use_container_width=True)

        # Export CSV / Excel
        c1, c2 = st.columns(2)
        with c1:
            csv = df_tip0.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Preuzmi tabelu (CSV)", csv, "tip0_kretanje.csv", "text/csv")
        with c2:
            import io
            import pandas as pd
            excel_bytes = io.BytesIO()
            with pd.ExcelWriter(excel_bytes, engine="openpyxl") as writer:
                df_tip0.to_excel(writer, sheet_name="TIP0", index=False)
            st.download_button("⬇️ Preuzmi tabelu (Excel)", excel_bytes.getvalue(), "tip0_kretanje.xlsx")

    except Exception as e:
        st.error(f"Greška: {e}")
with tab9:
    st.subheader("🚂 Kretanje 4098 kola–samo TIP 1")

    try:
        q = """
        WITH poslednji AS (
            SELECT 
                s.SerijaIpodserija,
                s.TIP,
                s.TelegBaza,
                k.Serija,
                s.PR,
                s.NR,
                k.Stanica,
                st.Naziv AS NazivStanice,
                k.DatumVreme,
                ROW_NUMBER() OVER (
                    PARTITION BY s.SerijaIpodserija
                    ORDER BY k.DatumVreme DESC
                ) AS rn
            FROM stanje s
            LEFT JOIN kola k 
              ON TRIM(CAST(s.SerijaIpodserija AS VARCHAR)) = TRIM(CAST(k.broj_kola_bez_rezima_i_kb AS VARCHAR))
            LEFT JOIN stanice st
              ON TRIM(CAST(k.Stanica AS VARCHAR)) = TRIM(CAST(st.Sifra AS VARCHAR))
            WHERE s.TIP = 1
        )
        SELECT 
            SerijaIpodserija,
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
        df_tip1 = run_sql(db_path, q)

        # Ako nema DatumVreme → BrojDana će biti NaN
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

        st.dataframe(df_tip1, use_container_width=True)

        # Export CSV / Excel
        c1, c2 = st.columns(2)
        with c1:
            csv = df_tip1.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Preuzmi tabelu (CSV)", csv, "tip1_kretanje.csv", "text/csv")
        with c2:
            import io
            import pandas as pd
            excel_bytes = io.BytesIO()
            with pd.ExcelWriter(excel_bytes, engine="openpyxl") as writer:
                df_tip1.to_excel(writer, sheet_name="TIP1", index=False)
            st.download_button("⬇️ Preuzmi tabelu (Excel)", excel_bytes.getvalue(), "tip1_kretanje.xlsx")

    except Exception as e:
        st.error(f"Greška: {e}")
with tab10:
    st.subheader("📊 Pivot po seriji i stanicama")

    try:
        q = """
        WITH poslednji AS (
            SELECT 
                k.Serija,
                k.Stanica,
                st.Naziv AS NazivStanice,
                s.TIP,
                ROW_NUMBER() OVER (
                    PARTITION BY k.broj_kola_bez_rezima_i_kb
                    ORDER BY k.DatumVreme DESC
                ) AS rn
            FROM kola k
            JOIN stanje s 
              ON TRIM(CAST(k.broj_kola_bez_rezima_i_kb AS VARCHAR)) = TRIM(CAST(s.SerijaIpodserija AS VARCHAR))
            LEFT JOIN stanice st
              ON TRIM(CAST(k.Stanica AS VARCHAR)) = TRIM(CAST(st.Sifra AS VARCHAR))
        )
        SELECT *
        FROM poslednji
        WHERE rn = 1
        """
        df_last = run_sql(db_path, q)

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

        # Dodaj red Σ
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

                # Dodaj red Σ
                total = {
                    "Stanica": "Σ",
                    "NazivStanice": "Ukupno",
                    "tip0": df_detail["tip0"].sum(),
                    "tip1": df_detail["tip1"].sum(),
                    "Ukupno": df_detail["Ukupno"].sum()
                }
                df_detail = pd.concat([df_detail, pd.DataFrame([total])], ignore_index=True)

                st.dataframe(df_detail, use_container_width=True)

    except Exception as e:
        st.error(f"Greška: {e}")

