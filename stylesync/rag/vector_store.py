"""Pinecone vector store wrapper for product metadata RAG."""

from __future__ import annotations

import logging
from typing import Optional

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from stylesync.rag.schema import GarmentMetadata, StyleGuideline
from stylesync.utils.config import settings

logger = logging.getLogger(__name__)

NAMESPACE = "products"


class ProductVectorStore:
    """Manages the Pinecone index for garment product data and style guidelines."""

    def __init__(self) -> None:
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._encoder = SentenceTransformer(settings.embedding_model)
        self._index = self._get_or_create_index()

    def _get_or_create_index(self):
        index_name = settings.pinecone_index
        existing = [idx.name for idx in self._pc.list_indexes()]
        if index_name not in existing:
            logger.info("Creating Pinecone index '%s' ...", index_name)
            self._pc.create_index(
                name=index_name,
                dimension=settings.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=settings.pinecone_environment),
            )
        return self._pc.Index(index_name)

    # ── Write ──────────────────────────────────────────────────────────────────

    def upsert_product(self, product_id: str, metadata: GarmentMetadata) -> None:
        """Embed and upsert a single product into Pinecone."""
        text = metadata.to_embedding_text()
        embedding = self._encoder.encode(text).tolist()
        self._index.upsert(
            vectors=[
                {
                    "id": product_id,
                    "values": embedding,
                    "metadata": metadata.model_dump(),
                }
            ],
            namespace=NAMESPACE,
        )
        logger.info("Upserted product %s", product_id)

    def upsert_batch(self, products: dict[str, GarmentMetadata], batch_size: int = 100) -> None:
        """Upsert multiple products in batches."""
        items = list(products.items())
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            vectors = []
            for pid, meta in batch:
                text = meta.to_embedding_text()
                emb = self._encoder.encode(text).tolist()
                vectors.append({"id": pid, "values": emb, "metadata": meta.model_dump()})
            self._index.upsert(vectors=vectors, namespace=NAMESPACE)
            logger.info("Upserted batch %d–%d", i, i + len(batch))

    # ── Read / Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query_text: str,
        top_k: int = 5,
        brand_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
    ) -> list[dict]:
        """Semantic search over products. Returns raw Pinecone matches."""
        embedding = self._encoder.encode(query_text).tolist()

        filters = {}
        if brand_filter:
            filters["brand"] = {"$eq": brand_filter}
        if category_filter:
            filters["category"] = {"$eq": category_filter}

        results = self._index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=NAMESPACE,
            filter=filters if filters else None,
        )
        return results.get("matches", [])

    def get_style_guideline(
        self,
        query_text: str,
        brand: Optional[str] = None,
        category: Optional[str] = None,
    ) -> StyleGuideline:
        """High-level helper: search and return the best-matching StyleGuideline."""
        matches = self.search(query_text, top_k=1, brand_filter=brand, category_filter=category)

        if not matches:
            logger.warning("No matching products found — returning defaults")
            return StyleGuideline(
                brand=brand or "unknown",
                pose="front",
                background="white",
                lighting="studio-soft",
                style_notes="",
                has_logo=False,
                logo_position=None,
                pattern="solid",
                fit="regular",
            )

        meta = matches[0]["metadata"]
        negative_hints = []
        if meta.get("has_logo"):
            negative_hints.append("do not alter, distort, or remove the logo/graphic")
        if meta.get("pattern") == "striped":
            negative_hints.append("preserve stripe alignment and spacing")

        return StyleGuideline(
            brand=meta.get("brand", "unknown"),
            pose=meta.get("model_pose", "front"),
            background=meta.get("background", "white"),
            lighting=meta.get("lighting", "studio-soft"),
            style_notes=meta.get("style_notes", ""),
            has_logo=meta.get("has_logo", False),
            logo_position=meta.get("logo_position"),
            pattern=meta.get("pattern", "solid"),
            fit=meta.get("fit", "regular"),
            negative_prompt_hints=negative_hints,
        )

    # ── Seed data helper ───────────────────────────────────────────────────────

    def seed_demo_data(self) -> None:
        """Populate the index with sample products for testing."""
        samples = {
            "SKU-NIKE-001": GarmentMetadata(
                brand="Nike",
                category="t-shirt",
                sub_category="crew-neck",
                color_primary="black",
                pattern="logo",
                fabric="cotton",
                fit="regular",
                gender="mens",
                has_logo=True,
                logo_position="center-chest",
                style_notes=(
                    "Nike brand guidelines: Swoosh logo must remain sharp and unaltered. "
                    "Use clean white or light-grey studio background. Model should face "
                    "camera with relaxed posture, hands at sides."
                ),
                model_pose="front",
                background="white",
                lighting="studio-soft",
                sku="SKU-NIKE-001",
            ),
            "SKU-ZARA-042": GarmentMetadata(
                brand="Zara",
                category="t-shirt",
                sub_category="v-neck",
                color_primary="white",
                pattern="solid",
                fabric="cotton-blend",
                fit="slim",
                gender="womens",
                has_logo=False,
                style_notes=(
                    "Zara editorial style: minimalist background, neutral tones. "
                    "Model should have a natural, editorial pose. Slight shadow for depth."
                ),
                model_pose="3/4",
                background="gradient",
                lighting="natural",
                sku="SKU-ZARA-042",
            ),
            "SKU-SUPREME-007": GarmentMetadata(
                brand="Supreme",
                category="t-shirt",
                sub_category="crew-neck",
                color_primary="red",
                pattern="graphic",
                fabric="cotton",
                fit="oversized",
                gender="unisex",
                has_logo=True,
                logo_position="center-chest",
                style_notes=(
                    "Supreme box logo must be pixel-perfect — no warping or color shift. "
                    "Streetwear aesthetic: urban background acceptable. "
                    "Model can have a relaxed, slightly angled stance."
                ),
                model_pose="3/4",
                background="lifestyle",
                lighting="dramatic",
                sku="SKU-SUPREME-007",
            ),
        }
        self.upsert_batch(samples)
        logger.info("Seeded %d demo products", len(samples))
