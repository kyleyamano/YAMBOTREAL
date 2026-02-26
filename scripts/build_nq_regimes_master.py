import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.data.loader import load_parquet
from src.research.regimes import add_regime_labels


def main():

    print("Loading NQ full history...")
    nq = load_parquet("data/processed/nq_continuous_1m.parquet")

    print("Computing regimes from NQ...")
    nq = add_regime_labels(nq)

    print("Saving NQ regime master...")
    nq[["regime"]].to_parquet("data/processed/nq_regimes_master.parquet")

    print("Done.")


if __name__ == "__main__":
    main()
