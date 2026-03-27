"""CLI helper to ingest product data from a JSON file into Pinecone."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from stylesync.rag.schema import GarmentMetadata
from stylesync.rag.vector_store import ProductVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_from_json(path: str | Path) -> None:
    """Load a JSON file of products and upsert into Pinecone.

    Expected JSON format:
    {
        "product-id-1": { ...GarmentMetadata fields... },
        "product-id-2": { ...GarmentMetadata fields... }
    }
    """
    path = Path(path)
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)

    raw = json.loads(path.read_text())
    products = {pid: GarmentMetadata(**data) for pid, data in raw.items()}
    logger.info("Parsed %d products from %s", len(products), path)

    store = ProductVectorStore()
    store.upsert_batch(products)
    logger.info("Ingestion complete")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m stylesync.rag.ingest <products.json>")
        sys.exit(1)
    ingest_from_json(sys.argv[1])
