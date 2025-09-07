import os
import time
import io
import requests
import sqlite3
import pandas as pd
import streamlit as st

# =========================
#  Konstante / Baze
# =========================
DB_URL = "https://drive.google.com/uc?export=download&id=1SbaxHotQ0BlNxts5f7tawLIQoWNu-hCG"
MAIN_DB = "kola_sk.db"            # glavna baza (SQLite fajl)
UPDATE_DB = "kola_sk_update.db"   # opciona lokalna "update" baza (isto SQLite)

# =========================
#  Preuzimanje glavne baze
# =========================
if not os.path.exists(MAIN_DB):
    with st.spinner("⬇ Preuzimam glavnu bazu sa Google Drive-a..."):
        r = requests.get(DB_URL, stream=True)
        r.raise_for_status()
        with open(MAIN_DB, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    st.success("✅ Glavna baza uspešno preuzeta sa Google Drive-a!")

# =========================
#  Helper funkcije
# =========================
def run_sql(sql: str) -> pd.DataFrame:
    """
    Izvršava SQL upit nad SQLite bazom.
    Ako postoji i UPDATE_DB, pravi UNION (view).
    """
    try:
        con = sqlite3.connect(MAIN_DB)

        # Ako postoji update baza, spoji je
        if os.path.exists(UPDATE_DB):
            con.execute(f"ATTACH DATABASE '{UPDATE_DB}' AS upd")

            # kreiraj pogled kola_view
            con.execute("""
                CREATE TEMP VIEW IF NOT EXISTS kola_view AS
                SELECT * FROM main.kola
                UNION ALL
                SELECT * FROM upd.kola
            """)
        else:
            con.execute("""
                CREATE TEMP VIEW IF NOT EXISTS kola_view AS
                SELECT * FROM main.kola
            """)

        df = pd.read_sql_query(sql, con)
        con.close()
        return df
    except Exception as e:
        st.error(f"Ne mogu da pročitam bazu: {e}")
        return pd.DataFrame()

def create_or_replace_table_from_df(db_file: str, table_name: str, df: pd.DataFrame):
    con = sqlite3.connect(db_file)
    try:
        df.to_sql(table_name, con, if_exists="replace", index=False)
    finally:
        con.close()

# =========================
#  SIDEBAR
# =========================
st.sidebar.title("⚙️ Podešavanja")
st.sidebar.caption("Glavna baza: kola_sk.db (SQLite). Opciona lokalna baza: kola_sk_update.db")

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Uvoz Excela → tabela 'stanje'")
uploaded_excel_stanje = st.sidebar.file_uploader("Izaberi Excel (.xlsx) za tabelu 'stanje'", type=["xlsx"], key="stanje_up")
if uploaded_excel_stanje is not None:
    if st.sidebar.button("📥 Učitaj u bazu kao 'stanje'"):
        try:
            df_stanje = pd.read_excel(uploaded_excel_stanje)
            create_or_replace_table_from_df(MAIN_DB, "stanje", df_stanje)
            st.sidebar.success(f"✅ 'stanje' učitano ({len(df_stanje)} redova).")
        except Exception as e:
            st.sidebar.error(f"❌ Greška pri uvozu 'stanje': {e}")

st.sidebar.subheader("🗺️ Uvoz mape stanica → tabela 'stanice'")
uploaded_excel_stanice = st.sidebar.file_uploader("Izaberi Excel (.xlsx) za tabelu 'stanice'", type=["xlsx"], key="stanice_up")
if uploaded_excel_stanice is not None:
    if st.sidebar.button("📥 Učitaj u bazu kao 'stanice'"):
        try:
            df_st = pd.read_excel(uploaded_excel_stanice)
            create_or_replace_table_from_df(MAIN_DB, "stanice", df_st)
            st.sidebar.success(f"✅ 'stanice' učitano ({len(df_st)} redova).")
        except Exception as e:
            st.sidebar.error(f"❌ Greška pri uvozu 'stanice': {e}")

st.sidebar.markdown("---")
st.sidebar.caption("Sve tabele možete koristiti u SQL upitima. Glavni podaci su u pogledu 'kola_view'.")

# =========================
#  Glavni naslov i Tabovi
# =========================
st.title("🚃 Teretna kola SK — kontrolna tabla")

tab1, tab2, tab3 = st.tabs(["📊 Pregled", "📈 Izveštaji", "🔎 SQL upiti"])

# =========================
#  Tab 1: Pregled
# =========================
with tab1:
    col_a, col_b, col_c, col_d = st.columns(4)
    try:
        df_cnt = run_sql("SELECT COUNT(*) AS broj_redova FROM kola_view")
        col_a.metric("Ukupan broj redova", f"{int(df_cnt['broj_redova'][0]):,}".replace(",", "."))

        df_files = run_sql("SELECT COUNT(DISTINCT source_file) AS fajlova FROM kola_view")
        col_b.metric("Učitanih fajlova", int(df_files["fajlova"][0]))

        df_range = run_sql("""
            SELECT MIN(DatumVreme) AS min_dt, MAX(DatumVreme) AS max_dt
            FROM kola_view WHERE DatumVreme IS NOT NULL
        """)
        min_dt = str(df_range["min_dt"][0]) if df_range["min_dt"][0] is not None else "—"
        max_dt = str(df_range["max_dt"][0]) if df_range["max_dt"][0] is not None else "—"
        col_c.metric("Najraniji datum", min_dt)
        col_d.metric("Najkasniji datum", max_dt)

        st.divider()
        st.subheader("Učitanih redova po fajlu (top 20)")
        df_by_file = run_sql("""
            SELECT source_file, COUNT(*) AS broj
            FROM kola_view
            GROUP BY source_file
            ORDER BY broj DESC
            LIMIT 20
        """)
        st.dataframe(df_by_file, use_container_width=True)
    except Exception as e:
        st.error(f"Ne mogu da pročitam bazu: {e}")
        st.stop()

# =========================
#  Tab 2: Izveštaji
# =========================
with tab2:
    st.subheader("Suma NetoTone po mesecu")
    df_month = run_sql("""
        SELECT strftime('%Y-%m', DatumVreme) AS mesec,
               SUM(COALESCE("NetoTone", 0)) AS ukupno_tona
        FROM kola_view
        WHERE DatumVreme IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """)
    if not df_month.empty:
        st.line_chart(df_month.set_index("mesec")["ukupno_tona"])

# =========================
#  Tab 3: SQL upiti
# =========================
with tab3:
    st.subheader("Piši svoj SQL (koristi npr. kola_view, stanje, stanice)")
    default_sql = "SELECT * FROM kola_view LIMIT 100"
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
# =========================
#  Tab 4: Pregled podataka
# =========================
with tab4:
    st.subheader("Brzi pregled")
    limit = st.slider("Broj redova (LIMIT)", 10, 2000, 200)
    cols = st.multiselect(
        "Kolone",
        [
            "Režim","Vlasnik","Serija","Inv br","KB","Tip kola","Voz br","Stanica","Status",
            "Datum","Vreme","Roba","Reon","tara","NetoTone","Broj vagona","Broj kola",
            "source_file","DatumVreme","broj_kola_bez_rezima_i_kb"
        ],
        default=["DatumVreme","Stanica","Tip kola","NetoTone","tara","source_file"]
    )
    try:
        cols_sql = ", ".join([f'"{c}"' if c not in ("DatumVreme",) else c for c in cols])
        df_preview = run_sql(db_path, f"SELECT {cols_sql} FROM kola_view LIMIT {int(limit)}")
        st.dataframe(df_preview, use_container_width=True)
    except Exception as e:
        st.error(f"Greška pri čitanju: {e}")

# =========================
#  Tab 5: Poslednji unosi (za 4098 iz 'stanje')
# =========================
with tab5:
    st.subheader("📌 Poslednji unos za 4098 kola iz Excel tabele (stanje)")
    if st.button("🔎 Prikaži poslednje unose"):
        try:
            q_last = """
                WITH poslednji AS (
                    SELECT 
                        s.SerijaIpodserija,
                        k.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY s.SerijaIpodserija
                            ORDER BY k.DatumVreme DESC
                        ) AS rn
                    FROM stanje s
                    JOIN kola_view k
                      ON TRIM(CAST(s.SerijaIpodserija AS VARCHAR)) = TRIM(CAST(k.broj_kola_bez_rezima_i_kb AS VARCHAR))
                )
                SELECT * FROM poslednji WHERE rn = 1
            """
            df_last = run_sql(db_path, q_last)
            st.success(f"✅ Pronađeno {len(df_last)} poslednjih unosa.")
            st.dataframe(df_last, use_container_width=True)
        except Exception as e:
            st.error(f"Greška u upitu: {e}")

# =========================
#  Tab 6: Pretraga kola
# =========================
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
                FROM kola_view
                WHERE "Broj kola" LIKE '%{broj_kola_input}%'
                  AND DatumVreme BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY DatumVreme DESC
            """
            df_search = run_sql(db_path, q_search)
            if df_search.empty:
                st.warning("⚠️ Nema podataka za zadate kriterijume.")
            else:
                st.success(f"✅ Pronađeno {len(df_search)} redova.")
                st.dataframe(df_search, use_container_width=True)
        except Exception as e:
            st.error(f"Greška u upitu: {e}")

# =========================
#  Tab 7: Kola po stanicama (pivot + detalji)
# =========================
with tab7:
    st.subheader("📊 Kola po stanicama")
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
            FROM kola_view k
            JOIN stanje s
              ON TRIM(CAST(k.broj_kola_bez_rezima_i_kb AS VARCHAR)) = TRIM(CAST(s.SerijaIpodserija AS VARCHAR))
            LEFT JOIN stanice st
              ON TRIM(CAST(k.Stanica AS VARCHAR)) = TRIM(CAST(st.Sifra AS VARCHAR))
        )
        SELECT * FROM poslednji WHERE rn = 1
        """
        df_last = run_sql(db_path, q)

        # Pivot po stanici (TIP 0/1)
        df_pivot = (
            df_last.groupby(["Stanica", "NazivStanice", "TIP"])
            .size().unstack(fill_value=0).reset_index()
        )
        if 0 not in df_pivot.columns: df_pivot[0] = 0
        if 1 not in df_pivot.columns: df_pivot[1] = 0
        df_pivot = df_pivot.rename(columns={0: "tip0", 1: "tip1"})
        df_pivot["Ukupno"] = df_pivot["tip0"] + df_pivot["tip1"]

        # Σ red
        total_row = {
            "Stanica": "Σ",
            "NazivStanice": "Ukupno",
            "tip0": int(df_pivot["tip0"].sum()),
            "tip1": int(df_pivot["tip1"].sum()),
            "Ukupno": int(df_pivot["Ukupno"].sum()),
        }
        df_pivot = pd.concat([df_pivot, pd.DataFrame([total_row])], ignore_index=True)

        left, right = st.columns(2)
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
                    .groupby(["Serija", "TIP"]).size().unstack(fill_value=0).reset_index()
                )
                if 0 not in df_detail.columns: df_detail[0] = 0
                if 1 not in df_detail.columns: df_detail[1] = 0
                df_detail = df_detail.rename(columns={0: "tip0", 1: "tip1"})
                df_detail["Ukupno"] = df_detail["tip0"] + df_detail["tip1"]
                total = {
                    "Serija": "Σ",
                    "tip0": int(df_detail["tip0"].sum()),
                    "tip1": int(df_detail["tip1"].sum()),
                    "Ukupno": int(df_detail["Ukupno"].sum()),
                }
                df_detail = pd.concat([df_detail, pd.DataFrame([total])], ignore_index=True)
                st.dataframe(df_detail, use_container_width=True)
    except Exception as e:
        st.error(f"Greška: {e}")

# =========================
#  Tab 8: TIP 0 (4098) – od najnovijeg ka najstarijem
# =========================
with tab8:
    st.subheader("🚂 Kretanje 4098 kola – samo TIP 0 (od najnovijeg ka najstarijem)")

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
            LEFT JOIN kola_view k 
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

        # Filteri
        series_options = ["Sve serije"] + sorted(df_tip0["Serija"].dropna().unique().tolist())
        selected_series = st.selectbox("🚆 Filtriraj po seriji kola", series_options, key="tip0_series")
        if selected_series != "Sve serije":
            df_tip0 = df_tip0[df_tip0["Serija"] == selected_series]

        station_options = ["Sve stanice"] + sorted(df_tip0["NazivStanice"].dropna().unique().tolist())
        selected_station = st.selectbox("📍 Filtriraj po stanici", station_options, key="tip0_station")
        if selected_station != "Sve stanice":
            df_tip0 = df_tip0[df_tip0["NazivStanice"] == selected_station]

        st.dataframe(df_tip0, use_container_width=True)

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

# =========================
#  Tab 9: TIP 1 (4098) – od najnovijeg ka najstarijem
# =========================
with tab9:
    st.subheader("🚂 Kretanje 4098 kola – samo TIP 1 (od najnovijeg ka najstarijem)")
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
            LEFT JOIN kola_view k 
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
        ORDER BY BrojDana ASC
        """
        df_tip1 = run_sql(db_path, q)

        if "BrojDana" in df_tip1.columns:
            df_tip1["BrojDana"] = df_tip1["BrojDana"].astype("Int64")

        # Filteri
        series_options = ["Sve serije"] + sorted(df_tip1["Serija"].dropna().unique().tolist())
        selected_series = st.selectbox("🚆 Filtriraj po seriji kola", series_options, key="tip1_series")
        if selected_series != "Sve serije":
            df_tip1 = df_tip1[df_tip1["Serija"] == selected_series]

        station_options = ["Sve stanice"] + sorted(df_tip1["NazivStanice"].dropna().unique().tolist())
        selected_station = st.selectbox("📍 Filtriraj po stanici", station_options, key="tip1_station")
        if selected_station != "Sve stanice":
            df_tip1 = df_tip1[df_tip1["NazivStanice"] == selected_station]

        st.dataframe(df_tip1, use_container_width=True)

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

# =========================
#  Tab 10: Pivot po serijama (sa detaljima)
# =========================
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
            FROM kola_view k
            JOIN stanje s
              ON TRIM(CAST(k.broj_kola_bez_rezima_i_kb AS VARCHAR)) = TRIM(CAST(s.SerijaIpodserija AS VARCHAR))
            LEFT JOIN stanice st
              ON TRIM(CAST(k.Stanica AS VARCHAR)) = TRIM(CAST(st.Sifra AS VARCHAR))
        )
        SELECT * FROM poslednji WHERE rn = 1
        """
        df_last = run_sql(db_path, q)

        df_pivot = (
            df_last.groupby(["Serija", "TIP"]).size().unstack(fill_value=0).reset_index()
        )
        if 0 not in df_pivot.columns: df_pivot[0] = 0
        if 1 not in df_pivot.columns: df_pivot[1] = 0
        df_pivot = df_pivot.rename(columns={0: "tip0", 1: "tip1"})
        df_pivot["Ukupno"] = df_pivot["tip0"] + df_pivot["tip1"]

        total_row = {
            "Serija": "Σ",
            "tip0": int(df_pivot["tip0"].sum()),
            "tip1": int(df_pivot["tip1"].sum()),
            "Ukupno": int(df_pivot["Ukupno"].sum())
        }
        df_pivot = pd.concat([df_pivot, pd.DataFrame([total_row])], ignore_index=True)

        left, right = st.columns(2)
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
                    .size().unstack(fill_value=0).reset_index()
                )
                if 0 not in df_detail.columns: df_detail[0] = 0
                if 1 not in df_detail.columns: df_detail[1] = 0
                df_detail = df_detail.rename(columns={0: "tip0", 1: "tip1"})
                df_detail["Ukupno"] = df_detail["tip0"] + df_detail["tip1"]

                total = {
                    "Stanica": "Σ",
                    "NazivStanice": "Ukupno",
                    "tip0": int(df_detail["tip0"].sum()),
                    "tip1": int(df_detail["tip1"].sum()),
                    "Ukupno": int(df_detail["Ukupno"].sum())
                }
                df_detail = pd.concat([df_detail, pd.DataFrame([total])], ignore_index=True)
                st.dataframe(df_detail, use_container_width=True)
    except Exception as e:
        st.error(f"Greška: {e}")
