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

# =========================
# Spajanje delova u kola_sk.db
# =========================
DB_PATH = "kola_sk.db"
FOLDER_ID = "1q__8P3gY-JMzqD5cpt8avm_7VAY-fHWI"

# =========================
# Fallback download sa pydrive2
# =========================
def download_with_pydrive2(folder_id: str):
    """Ako gdown ne uspe, koristi PyDrive2."""
    gauth = GoogleAuth()
    gauth.CommandLineAuth()
    drive = GoogleDrive(gauth)

    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    st.write(f"📂 Pydrive2 našao {len(file_list)} fajlova u folderu")
    for f in file_list:
        fname = f["title"]
        st.write(f"⬇️ Preuzimam {fname}...")
        f.GetContentFile(fname)

# =========================
# Merge delova u jednu bazu
# =========================
def merge_parts():
    part_files = []
    for root, dirs, files in os.walk("."):
        for f in files:
            if re.match(r"(Copy of )?kola_sk\.db\.part\d+$", f):
                full_path = os.path.join(root, f)
                # ako nije u root folderu, prebaci ga u root
                if root != ".":
                    dest_path = os.path.join(".", f)
                    shutil.move(full_path, dest_path)
                    full_path = dest_path
                part_files.append(full_path)

    # Sortiranje po broju dela
    part_files = sorted(part_files, key=lambda x: int(re.search(r"part(\d+)", x).group(1)))

    if len(part_files) == 48:
        with open(DB_PATH, "wb") as outfile:
            for fname in part_files:
                st.write(f"➡️ Dodajem {fname}")
                with open(fname, "rb") as infile:
                    outfile.write(infile.read())
        st.success(f"✅ Spojeno {len(part_files)} delova → {DB_PATH}")
    else:
        st.error(f"❌ Nađeno {len(part_files)} fajlova, očekivano 48")
        st.write("📂 Fajlovi koje sam našao:", part_files)
# =========================
# Preuzimanje i spajanje
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

    # ✅ Tek posle downloada spajaj
    merge_parts()

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
# Glavne promenljive sa fallback logikom
if table_exists("main", "kola_view"):
    table_name = "kola_view"
elif table_exists("main", "kola"):
    table_name = "kola"
else:
    st.error("❌ Nema ni 'kola_view' ni 'kola' u bazi.")
    st.stop()

db_path = DB_PATH

st.title("🚃 Teretna kola SK — kontrolna tabla")

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
        
# ---------- Tab 5: Poslednje stanje kola ----------

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
        JOIN "{table_name}" k
          ON CAST(s.SerijaIpodserija AS TEXT) = REPLACE(k.broj_kola_bez_rezima_i_kb, ' ', '')
    )
    SELECT *
    FROM ranked
    WHERE rn = 1
"""
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
        df_last = run_sql(DB_PATH, q)

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
        df_tip0 = run_sql(DB_PATH, q)

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
        df_tip1 = run_sql(DB_PATH, q)

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
        df_last = run_sql(DB_PATH, q)

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

