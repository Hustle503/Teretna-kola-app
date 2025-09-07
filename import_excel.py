import duckdb
import pandas as pd

# 1. Putanja do Excela (može i apsolutna npr. r"C:\Users\PC\Stanje SK.xlsx")
excel_path = "Stanje SK.xlsx"

# 2. Učitaj Excel u pandas DataFrame
df_stanje = pd.read_excel(excel_path)

print("✅ Excel učitan, broj redova:", len(df_stanje))

# 3. Poveži se na DuckDB bazu
con = duckdb.connect("kola_sk.db")

# 4. Snimi podatke iz Excela kao novu tabelu 'stanje'
con.register("df_stanje", df_stanje)
con.execute("CREATE OR REPLACE TABLE stanje AS SELECT * FROM df_stanje")
con.unregister("df_stanje")

# 5. Proveri koje kolone sada ima tabela 'stanje'
print("📋 Kolone u tabeli stanje:")
print(con.execute("PRAGMA table_info(stanje)").fetchdf())

con.close()
