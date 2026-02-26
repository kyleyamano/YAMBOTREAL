from pathlib import Path
import pandas as pd
import databento as db

RAW = Path("data/raw/NQ/nq_1m_2010_2026.parquet.zst")
OUT = Path("data/processed/nq_continuous_1m.parquet")


def main():
    print("Loading NQ DBN...")
    store = db.DBNStore.from_file(str(RAW))
    df = store.to_df()

    df.columns = [c.lower() for c in df.columns]

    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    # Remove spreads
    df = df[~df["symbol"].str.contains("-", na=False)]

    print("Outright contracts:", df["symbol"].nunique())

    df = df[["symbol", "open", "high", "low", "close", "volume"]]

    price_cols = ["open", "high", "low", "close"]

    # Handle zero prices
    zero_mask = (df[price_cols] <= 0).any(axis=1)
    if int(zero_mask.sum()) > 0:
        print(f"Found {int(zero_mask.sum())} zero-price bars. Converting to NaN.")
        df.loc[zero_mask, price_cols] = pd.NA

    if (df["volume"] < 0).any():
        raise ValueError("Negative volume found.")

    # Daily front month selection
    df["date"] = df.index.date

    daily_vol = (
        df.groupby(["date", "symbol"])["volume"]
        .sum()
        .reset_index()
    )

    idx = daily_vol.groupby("date")["volume"].idxmax()
    front_month = daily_vol.loc[idx][["date", "symbol"]]

    df = df.reset_index()
    df = df.merge(front_month, on=["date", "symbol"], how="inner")

    dt_col = df.columns[0]
    df = df.set_index(dt_col)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    df = df[["open", "high", "low", "close", "volume"]]

    # Enforce full 1-minute grid
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="1min", tz="UTC")
    df = df.reindex(full_idx)

    df[price_cols] = df[price_cols].ffill()
    df["volume"] = df["volume"].fillna(0)

    if df.isna().any().any():
        raise ValueError("NaNs remain.")
    if df.index.has_duplicates:
        raise ValueError("Duplicates remain.")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index not sorted.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)

    print("\nNQ CONTINUOUS BUILT")
    print("Rows:", len(df))
    print("Start:", df.index.min())
    print("End:", df.index.max())


if __name__ == "__main__":
    main()
