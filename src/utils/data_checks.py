import pandas as pd

REQUIRED_COLS = {"open", "high", "low", "close"}

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # If datetime is a column, set it as index
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.set_index("datetime")

    # Standardize common alternates
    rename_map = {}
    for a, b in [("o","open"), ("h","high"), ("l","low"), ("c","close"), ("vol","volume")]:
        if a in df.columns and b not in df.columns:
            rename_map[a] = b
    if rename_map:
        df = df.rename(columns=rename_map)

    return df

def validate_market_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.name is None:
        raise ValueError("DataFrame must have a datetime index.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex.")

    df = df.sort_index()

    # Drop rows with invalid timestamps
    if df.index.isna().any():
        df = df[~df.index.isna()]

    # Dedupe deterministically (keep last)
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Have: {list(df.columns)}")

    # Basic sanity checks
    bad = (df["high"] < df["low"]) | (df["close"] <= 0) | (df["open"] <= 0)
    if bad.any():
        n = int(bad.sum())
        raise ValueError(f"Found {n} invalid bars (high<low or non-positive prices).")

    return df
