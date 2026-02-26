from pathlib import Path
import pandas as pd

NQ = Path("data/processed/nq_continuous_1m.parquet")
MNQ = Path("data/processed/mnq_continuous_1m.parquet")
OUT = Path("data/processed/extended_nq_mnq_1m.parquet")


def main():
    print("Loading datasets...")
    nq = pd.read_parquet(NQ)
    mnq = pd.read_parquet(MNQ)

    nq.index = pd.to_datetime(nq.index, utc=True)
    mnq.index = pd.to_datetime(mnq.index, utc=True)

    nq = nq.sort_index()
    mnq = mnq.sort_index()

    cut = mnq.index.min()

    print("Cutover timestamp:", cut)

    nq_pre = nq[nq.index < cut]

    print("NQ pre rows:", len(nq_pre))
    print("MNQ rows:", len(mnq))

    df = pd.concat([nq_pre, mnq], axis=0)

    # Structural validation
    if df.index.has_duplicates:
        raise ValueError("Duplicates after concat.")

    expected = len(pd.date_range(df.index.min(), df.index.max(), freq="1min", tz="UTC"))
    missing = expected - len(df)

    if missing != 0:
        raise ValueError(f"Missing minutes after concat: {missing}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)

    print("\nEXTENDED DATASET BUILT")
    print("Rows:", len(df))
    print("Start:", df.index.min())
    print("End:", df.index.max())


if __name__ == "__main__":
    main()
