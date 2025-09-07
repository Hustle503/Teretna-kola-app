import duckdb
import pandas as pd

# putanja do Excela sa stanicama
EXCEL_FILE = r"C:\Teretna kola\stanice1.xlsx"

# 1. Učitaj Excel
df_stanice = pd.read_excel(EXCEL_FILE)

# 2. Poveži se na postojeću bazu
con = duckdb.connect(r"C:\Teretna kola\kola_sk.db")

# 3. Ubaci podatke u novu tabelu
con.register("df_stanice", df_stanice)
con.execute("CREATE OR REPLACE TABLE stanice AS SELECT * FROM df_stanice")
con.unregister("df_stanice")

# 4. Proveri strukturu
print(con.execute("PRAGMA table_info(stanice)").fetchdf())
