from pathlib import Path
import pandas as pd
import databento as db

RAW = Path("data/raw/MNQ/mnq_1m_full.dbn.zst")
OUT = Path("data/processed/mnq_continuous_1m.parquet")


def main():
    print("Loading DBN...")
    store = db.DBNStore.from_file(str(RAW))
    df = store.to_df()
    df.columns = [c.lower() for c in df.columns]

    # Ensure datetime index
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    # Remove spread symbols
    df = df[~df["symbol"].str.contains("-")]

    print("Outright contracts:", df["symbol"].nunique())

    # Keep relevant columns
    df = df[["symbol", "open", "high", "low", "close", "volume"]]

    # Handle zero price bars
    price_cols = ["open", "high", "low", "close"]
    zero_mask = (df[price_cols] <= 0).any(axis=1)
    df.loc[zero_mask, price_cols] = pd.NA

    # Add date column (UTC date)
    df["date"] = df.index.date

    # Compute daily volume per contract
    daily_vol = (
        df.groupby(["date", "symbol"])["volume"]
        .sum()
        .reset_index()
    )

    # Highest volume contract per day
    idx = daily_vol.groupby("date")["volume"].idxmax()
    front_month = daily_vol.loc[idx][["date", "symbol"]]

    # Merge to keep only selected contract per day
    df = df.reset_index()
    df = df.merge(front_month, on=["date", "symbol"], how="inner")

    # Restore datetime index (first column after reset_index)
    datetime_col = df.columns[0]
    df = df.set_index(datetime_col)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    df = df[["open", "high", "low", "close", "volume"]]

    # Enforce full 1-minute grid
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="1min", tz="UTC")
    df = df.reindex(full_idx)

    df[price_cols] = df[price_cols].ffill()
    df["volume"] = df["volume"].fillna(0)

    # Final validation
    if df.isna().any().any():
        raise ValueError("NaNs remain.")
    if df.index.has_duplicates:
        raise ValueError("Duplicates remain.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index not sorted.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)

    print("\nCONTINUOUS CONTRACT BUILT")
    print("Rows:", len(df))
    print("Start:", df.index.min())
    print("End:", df.index.max())


if __name__ == "__main__":
    main()
