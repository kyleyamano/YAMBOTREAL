import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.data.loader import load_parquet


def main():

    print("Loading MNQ clean dataset...")
    mnq = load_parquet("data/processed/mnq_continuous_1m.parquet")

    print("Loading NQ regime master...")
    nq_regimes = pd.read_parquet("data/processed/nq_regimes_master.parquet")

    print("Aligning NQ regimes to MNQ timestamps...")
    mnq = mnq.merge(
        nq_regimes[["regime"]],
        left_index=True,
        right_index=True,
        how="left"
    )

    print("Forward-filling any missing regime values...")
    mnq["regime"] = mnq["regime"].ffill()

    print("Saving MNQ with NQ-derived regimes...")
    mnq.to_parquet("data/processed/mnq_1m_with_regimes.parquet")

    print("Done.")


if __name__ == "__main__":
    main()
