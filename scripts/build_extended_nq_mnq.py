from pathlib import Path
import pandas as pd

NQ = Path("data/processed/nq_continuous_1m.parquet")
MNQ = Path("data/processed/mnq_continuous_1m.parquet")
OUT = Path("data/processed/nq_pre2019_mnq_post2019_1m.parquet")


def main():
    nq = pd.read_parquet(NQ)
    mnq = pd.read_parquet(MNQ)

    # Ensure UTC index
    nq.index = pd.to_datetime(nq.index, utc=True)
    mnq.index = pd.to_datetime(mnq.index, utc=True)

    nq = nq.sort_index()
    mnq = mnq.sort_index()

    # Cut NQ strictly before MNQ begins (no overlap)
    cut = mnq.index.min()
    nq = nq[nq.index < cut]

    # Concatenate
    df = pd.concat([nq, mnq], axis=0)

    # Validate continuity (no missing minutes, no dupes)
    if df.index.has_duplicates:
        raise ValueError("Duplicates found after concat.")
    expected = len(pd.date_range(df.index.min(), df.index.max(), freq="1min", tz="UTC"))
    missing = expected - len(df)
    if missing != 0:
        raise ValueError(f"Missing minutes after concat: {missing}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)

    print("\nEXTENDED SERIES BUILT")
    print("Rows:", len(df))
    print("Start:", df.index.min())
    print("Cutover:", cut)
    print("End:", df.index.max())


if __name__ == "__main__":
    main()
