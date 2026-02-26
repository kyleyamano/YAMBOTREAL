from pathlib import Path
import pandas as pd
import databento as db

# 🔧 Adjust RAW path if needed
RAW = Path("data/raw/MNQ/mnq_1m_full.dbn.zst")
OUT = Path("data/processed/mnq_1m_clean.parquet")


def main():
    print("Loading DBN file...")
    store = db.DBNStore.from_file(str(RAW))
    df = store.to_df()

    # Lowercase columns
    df.columns = [c.lower() for c in df.columns]

    # Ensure timestamp is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex from Databento ohlcv-1m")

    # Force UTC + sort
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    # Keep only canonical OHLCV
    df = df[["open", "high", "low", "close", "volume"]]

    print("Initial rows:", len(df))

    # 🔎 Handle zero / non-positive price bars
    price_cols = ["open", "high", "low", "close"]
    zero_mask = (df[price_cols] <= 0).any(axis=1)

    zero_count = int(zero_mask.sum())
    if zero_count > 0:
        print(f"Found {zero_count} zero/non-positive price bars. Converting to NaN.")
        df.loc[zero_mask, price_cols] = pd.NA

    # Volume must never be negative
    if (df["volume"] < 0).any():
        raise ValueError("Negative volume found.")

    # Remove duplicate timestamps safely
    if df.index.has_duplicates:
        dup_count = df.index.duplicated().sum()
        print(f"Found {dup_count} duplicate timestamps. Keeping last.")
        df = df[~df.index.duplicated(keep="last")]

    # 🔧 Enforce full 1-minute grid
    print("Enforcing 1-minute grid...")
    full_idx = pd.date_range(
        df.index.min(),
        df.index.max(),
        freq="1min",
        tz="UTC"
    )

    df = df.reindex(full_idx)

    # Forward fill prices
    df[price_cols] = df[price_cols].ffill()

    # Missing volume = 0
    df["volume"] = df["volume"].fillna(0)

    # Final structural validation
    if df.isna().any().any():
        raise ValueError("NaNs remain after cleaning.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Index not monotonic increasing.")

    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps remain.")

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)

    print("\nCLEAN DATA WRITTEN")
    print("Rows:", len(df))
    print("Start:", df.index.min())
    print("End:", df.index.max())


if __name__ == "__main__":
    main()
