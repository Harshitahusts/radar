# StyleSync AI

> Upload a flat-lay garment image → get professional HD studio photos of a model wearing that exact garment, with logo and texture integrity preserved.

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Agent 1    │     │     Agent 2      │     │     Agent 3      │
│  Garment     │────▶│  RAG Retriever   │────▶│  Image           │
│  Analyzer    │     │  (Pinecone)      │     │  Generator       │
│  (GPT-4o)    │     │                  │     │  (SDXL+ControlNet)│
└──────────────┘     └──────────────────┘     └──────────────────┘
      │                      │                        │
  Detects brand,       Fetches style           Generates HD
  logo, pattern,       guidelines: pose,       model-on images
  colors, fabric       bg, lighting,           with logo
  from flat-lay        logo protection         preservation
```

**Orchestrator**: LangGraph `StateGraph` — sequential pipeline with error routing.

## Tech Stack

| Component | Technology |
|---|---|
| Multi-Agent Framework | LangGraph (LangChain) |
| Vector Database (RAG) | Pinecone (Serverless) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vision Analysis | GPT-4o (via LangChain) |
| Image Generation | SDXL + ControlNet (OpenPose + Canny) |
| Garment Transfer | IP-Adapter Plus |
| Virtual Try-On | IDM-VTON (alternative pipeline) |
| Background Removal | rembg (U2-Net) |
| API | FastAPI |

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd radar
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Seed demo data into Pinecone
python -c "from stylesync.rag.vector_store import ProductVectorStore; ProductVectorStore().seed_demo_data()"

# 4. Run the API server
uvicorn stylesync.api.server:app --host 0.0.0.0 --port 8000

# 5. Upload a garment
curl -X POST http://localhost:8000/generate \
  -F "garment_image=@tshirt.png" \
  -F "brand=Nike" \
  -F "num_images=2"
```

## Project Structure

```
stylesync/
├── agents/
│   ├── state.py              # Shared AgentState TypedDict
│   ├── garment_analyzer.py   # Agent 1: GPT-4o vision analysis
│   ├── rag_retriever.py      # Agent 2: Pinecone style lookup
│   ├── image_generator.py    # Agent 3: SDXL + ControlNet generation
│   └── graph.py              # LangGraph workflow definition
├── rag/
│   ├── schema.py             # Pydantic models + Pinecone schema
│   ├── vector_store.py       # Pinecone CRUD operations
│   └── ingest.py             # Bulk data ingestion CLI
├── imaging/
│   ├── garment_processor.py  # BG removal, masking, logo extraction
│   ├── tryon_pipeline.py     # Diffusers SDXL + ControlNet pipeline
│   └── pose_estimator.py     # OpenPose + default skeleton generation
├── api/
│   └── server.py             # FastAPI REST endpoints
└── utils/
    └── config.py             # Environment-based settings

configs/
└── comfyui_workflow.json     # ComfyUI node graph for local generation

tests/
├── test_schema.py
├── test_garment_processor.py
├── test_tryon_pipeline.py
└── test_pose_estimator.py
```

## Logo & Texture Preservation Strategy

The system uses a **multi-layer protection** approach:

1. **Detection**: GPT-4o identifies logo presence, position, and type
2. **Masking**: A binary inpainting mask protects the logo bounding box
3. **Canny ControlNet**: Preserves edge boundaries of the garment pattern
4. **IP-Adapter**: Transfers texture/color from the original flat-lay image
5. **Compositing**: After generation, the original logo pixels are composited back
6. **Refinement**: A low-denoise (0.3) inpainting pass blends seams without altering the logo

## RAG Schema

Each product in Pinecone carries metadata that drives generation decisions:

| Field | Type | Purpose |
|---|---|---|
| `brand` | string | Brand name → style guideline lookup |
| `category` | string | Garment type (t-shirt, hoodie, etc.) |
| `has_logo` | bool | Triggers logo protection pipeline |
| `logo_position` | string | Drives ROI extraction for masking |
| `model_pose` | string | Controls ControlNet pose skeleton |
| `background` | string | Sets the scene description in prompt |
| `lighting` | string | Controls lighting in prompt |
| `style_notes` | string | Free-text brand-specific instructions |

## Running Tests

```bash
pytest tests/ -v
```
