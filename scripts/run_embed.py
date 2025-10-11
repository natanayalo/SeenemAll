import os
from etl.compute_embeddings import run

if __name__ == "__main__":
    # Optional env overrides:
    #   EMBED_BATCH=64 EMBED_MODEL=all-MiniLM-L6-v2
    #   MAX_ITEMS=1000
    max_items = os.getenv("MAX_ITEMS")
    run(
        batch=int(os.getenv("EMBED_BATCH", "64")),
        max_items=int(max_items) if max_items else None,
    )
