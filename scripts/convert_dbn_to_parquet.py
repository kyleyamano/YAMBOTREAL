import databento as db
import pandas as pd

# Path to your downloaded file
INPUT_PATH = "data/raw/databento/mnq_1m_full.dbn.zst"

# Output parquet path
OUTPUT_PATH = "data/raw/mnq_1m_full.parquet"

print("Reading DBN file...")

# Load DBN file
store = db.DBNStore.from_file(INPUT_PATH)

df = store.to_df()

print("Rows loaded:", len(df))

# Keep only needed columns
df = df.reset_index()

df = df[["ts_event", "open", "high", "low", "close", "volume"]]

df.rename(columns={"ts_event": "datetime"}, inplace=True)

# Convert timestamp
df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

print("Saving to Parquet...")

df.to_parquet(OUTPUT_PATH, index=False)

print("Done.")
