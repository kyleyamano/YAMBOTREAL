from pathlib import Path
import zstandard as zstd

RAW_ZST = Path("data/raw/NQ/nq_1m_2010_2026.parquet.zst")
OUT_PARQUET = Path("data/raw/NQ/nq_1m_2010_2026.parquet")

def main():
    print("Decompressing NQ ZST file...")

    dctx = zstd.ZstdDecompressor()

    with open(RAW_ZST, "rb") as compressed:
        with open(OUT_PARQUET, "wb") as destination:
            dctx.copy_stream(compressed, destination)

    print("Done.")
    print("Output:", OUT_PARQUET)

if __name__ == "__main__":
    main()
