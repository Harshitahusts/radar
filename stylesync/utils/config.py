"""Centralized configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # LLM
    openai_api_key: str = field(default_factory=lambda: os.environ["OPENAI_API_KEY"])
    llm_model: str = "gpt-4o"

    # Pinecone
    pinecone_api_key: str = field(default_factory=lambda: os.environ["PINECONE_API_KEY"])
    pinecone_index: str = field(
        default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "stylesync-products")
    )
    pinecone_environment: str = field(
        default_factory=lambda: os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    )
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Image models
    sdxl_model_id: str = field(
        default_factory=lambda: os.getenv(
            "SDXL_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0"
        )
    )
    controlnet_model_id: str = field(
        default_factory=lambda: os.getenv(
            "CONTROLNET_MODEL_ID", "lllyasviel/control_v11p_sd15_openpose"
        )
    )
    tryon_model_id: str = field(
        default_factory=lambda: os.getenv("TRYON_MODEL_ID", "yisol/IDM-VTON")
    )

    # Paths
    upload_dir: Path = field(
        default_factory=lambda: Path(os.getenv("UPLOAD_DIR", "./uploads"))
    )
    output_dir: Path = field(
        default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "./outputs"))
    )
    max_upload_size_mb: int = 20

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
