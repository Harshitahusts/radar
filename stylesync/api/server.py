"""FastAPI server — exposes the StyleSync pipeline as a REST API."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from stylesync.agents.graph import stylesync_graph
from stylesync.agents.state import AgentState
from stylesync.utils.config import settings

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="StyleSync AI", description="Upload a flat-lay garment image → get professional model-on product shots", version="0.1.0")

settings.ensure_dirs()
app.mount("/outputs", StaticFiles(directory=str(settings.output_dir)), name="outputs")


class GenerationResponse(BaseModel):
    job_id: str
    status: str
    output_paths: list[str]
    style_context: str
    error: str | None = None


@app.post("/generate", response_model=GenerationResponse)
async def generate_tryon(
    garment_image: UploadFile = File(..., description="Flat-lay garment photo"),
    brand: str = Form(default=None, description="Brand name hint (optional)"),
    category: str = Form(default=None, description="Category hint: t-shirt, hoodie, etc."),
    num_images: int = Form(default=1, ge=1, le=4, description="Number of variants"),
    reference_model: UploadFile | None = File(default=None, description="Optional reference model photo for pose"),
):
    """Upload a garment flat-lay and generate model-on studio images."""
    job_id = uuid.uuid4().hex[:12]
    garment_path = settings.upload_dir / f"{job_id}_garment{Path(garment_image.filename).suffix}"
    garment_path.write_bytes(await garment_image.read())

    ref_model_path = None
    if reference_model:
        ref_model_path = settings.upload_dir / f"{job_id}_model{Path(reference_model.filename).suffix}"
        ref_model_path.write_bytes(await reference_model.read())

    initial_state: AgentState = {
        "garment_image_path": str(garment_path),
        "brand_hint": brand,
        "category_hint": category,
        "reference_model_path": str(ref_model_path) if ref_model_path else None,
        "num_images": num_images,
    }

    try:
        result = stylesync_graph.invoke(initial_state)
    except Exception as e:
        logger.exception("Pipeline failed for job %s", job_id)
        return JSONResponse(status_code=500, content=GenerationResponse(job_id=job_id, status="error", output_paths=[], style_context="", error=str(e)).model_dump())

    if result.get("error"):
        return GenerationResponse(job_id=job_id, status="error", output_paths=[], style_context=result.get("rag_context", ""), error=result["error"])

    return GenerationResponse(job_id=job_id, status="completed", output_paths=result.get("output_paths", []), style_context=result.get("rag_context", ""))


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
