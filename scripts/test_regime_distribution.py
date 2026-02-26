import pandas as pd

df = pd.read_parquet("data/processed/mnq_1m_with_regimes.parquet")

print("Regime Distribution:")
print(df["regime"].value_counts())

print("\nNull Regimes:", df["regime"].isna().sum())
print("Date Range:", df.index.min(), "→", df.index.max())
