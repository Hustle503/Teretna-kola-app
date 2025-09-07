import duckdb
import polars as pl
import glob
import os
import json

DB_FILE = "kola_sk.db"
STATE_FILE = "loaded_files.json"


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
                "Broj kola": line[1:11].strip(),
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

        # ✅ dodat flag za validnost datuma
        pl.col("Datum").str.strptime(pl.Date, "%Y%m%d", strict=False).is_not_null().alias("Datum_validan")
    ])

    # Brojevi u int
    df = df.with_columns([
        pl.col("tara").cast(pl.Int32, strict=False),
        pl.col("NetoTone").cast(pl.Int32, strict=False),
        pl.col("Inv br").cast(pl.Int32, strict=False),
    ])

    return df

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_state(processed_files):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(list(processed_files), f, indent=2)


def init_database(folder: str, table_name: str = "kola"):
    files = glob.glob(os.path.join(folder, "*.txt"))
    if not files:
        raise FileNotFoundError(f"Nema txt fajlova u folderu: {folder}")

    all_dfs = [parse_txt(f) for f in files]
    df = pl.concat(all_dfs)

    con = duckdb.connect(DB_FILE)
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.register("df", df)
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    con.unregister("df")

    save_state(set(files))
    print(f"✅ Inicijalno učitano {len(df)} redova iz {len(files)} fajlova")
    return con


def update_database(folder: str, table_name: str = "kola"):
    processed = load_state()
    files = set(glob.glob(os.path.join(folder, "*.txt")))

    new_files = files - processed
    if not new_files:
        print("ℹ️ Nema novih fajlova za unos.")
        return

    con = duckdb.connect(DB_FILE)
    for f in sorted(new_files):
        df_new = parse_txt(f)
        con.register("df_new", df_new)
        con.execute(f"INSERT INTO {table_name} SELECT * FROM df_new")
        con.unregister("df_new")
        print(f"➕ Ubačeno {len(df_new)} redova iz {os.path.basename(f)}")
        processed.add(f)

    save_state(processed)
    print("✅ Update završen.")


def reload_file(path: str, table_name: str = "kola"):
    """Ponovo učitaj fajl – obriši stare redove i unesi nove."""
    fname = os.path.basename(path)

    con = duckdb.connect(DB_FILE)

    # 1. Obrisi stare redove tog fajla
    con.execute(f"DELETE FROM {table_name} WHERE source_file = ?", [fname])

    # 2. Učitaj nove podatke
    df_new = parse_txt(path)
    con.register("df_new", df_new)
    con.execute(f"INSERT INTO {table_name} SELECT * FROM df_new")
    con.unregister("df_new")

    # 3. Osveži state fajl
    processed = load_state()
    processed.add(path)
    save_state(processed)

    print(f"🔄 Fajl {fname} ponovo učitan ({len(df_new)} redova)")