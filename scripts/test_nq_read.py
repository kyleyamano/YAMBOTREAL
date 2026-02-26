import pandas as pd

print("Reading NQ parquet...")

df = pd.read_parquet("data/raw/NQ/nq_1m_2010_2026.parquet")

print("SUCCESS")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("Date range:", df.index.min(), "→", df.index.max())
