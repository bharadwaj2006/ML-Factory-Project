"""
=============================================================
LEXISCAN AUTO — STEP 4: FastAPI REST API
=============================================================
PURPOSE:
    Expose the full pipeline (OCR → NER → Validation) as a
    production-ready REST API endpoint.

    Endpoint:  POST /extract
    Input:     PDF file (multipart upload) OR raw text (JSON)
    Output:    Structured JSON with validated entities

HOW TO RUN LOCALLY:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --reload --port 8000

    Then open:  http://localhost:8000/docs
    (FastAPI auto-generates interactive Swagger UI)

HOW TO TEST:
    curl -X POST "http://localhost:8000/extract" \
         -F "file=@contract.pdf"

    OR with raw text:
    curl -X POST "http://localhost:8000/extract/text" \
         -H "Content-Type: application/json" \
         -d '{"text": "Agreement dated March 15 2024 between Apex LLC and Beta Corp for $500,000."}'

API ENDPOINTS:
    GET  /              → Health check
    GET  /health        → Detailed health status
    POST /extract       → Upload PDF → get entities
    POST /extract/text  → Send raw text → get entities
    GET  /docs          → Swagger UI (auto-generated)
=============================================================
"""

import os
import sys
import json
import uuid
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Import our pipeline modules ──
# Add parent directory to path so we can import step modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from step1_ocr.ocr_pipeline import extract_text_from_pdf
from step3_bert_validation.validator import validate_entities

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
# Paths to trained models (set via environment variable or default)
SPACY_MODEL_PATH = os.getenv("SPACY_MODEL_PATH", "models/ner_spacy_model")
BERT_MODEL_PATH  = os.getenv("BERT_MODEL_PATH",  "models/bert_ner_model")

# Which NER model to use: "spacy" or "bert"
# BERT is more accurate; spaCy is faster
NER_BACKEND = os.getenv("NER_BACKEND", "spacy")

# Temp directory for uploaded PDFs
UPLOAD_DIR = "/tmp/lexiscan_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Maximum PDF size: 50 MB
MAX_FILE_SIZE_MB = 50


# ─────────────────────────────────────────────
# LOAD NER MODEL AT STARTUP
# ─────────────────────────────────────────────
# Models are loaded once into memory (not on every request)
_ner_model = None
_ner_tokenizer = None

def load_ner_model():
    """Load the NER model into memory at server startup."""
    global _ner_model, _ner_tokenizer

    if NER_BACKEND == "spacy":
        import spacy
        if os.path.exists(SPACY_MODEL_PATH):
            logger.info(f"Loading spaCy model from: {SPACY_MODEL_PATH}")
            _ner_model = spacy.load(SPACY_MODEL_PATH)
            logger.info("✓ spaCy NER model loaded.")
        else:
            logger.warning(f"spaCy model not found at {SPACY_MODEL_PATH}. Run step2 first.")

    elif NER_BACKEND == "bert":
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        if os.path.exists(BERT_MODEL_PATH):
            logger.info(f"Loading BERT model from: {BERT_MODEL_PATH}")
            _ner_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
            _ner_model     = AutoModelForTokenClassification.from_pretrained(BERT_MODEL_PATH)
            _ner_model.eval()
            logger.info("✓ BERT NER model loaded.")
        else:
            logger.warning(f"BERT model not found at {BERT_MODEL_PATH}. Run step3 first.")


def run_ner(text: str) -> list:
    """
    Run NER on text using whichever model is loaded.
    Returns list of raw entity dicts: [{text, label, start, end}, ...]
    """
    if _ner_model is None:
        # Fallback: use a simple regex-based extractor for demo purposes
        return _regex_fallback_ner(text)

    if NER_BACKEND == "spacy":
        doc = _ner_model(text)
        return [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

    elif NER_BACKEND == "bert":
        from step3_bert_validation.bert_finetune import bert_predict
        return bert_predict(BERT_MODEL_PATH, text)

    return []


def _regex_fallback_ner(text: str) -> list:
    """
    Simple regex-based NER used when no trained model is available.
    Less accurate than the trained models — for demo/testing only.
    """
    import re
    entities = []

    # Dates
    date_pattern = r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b"
    for m in re.finditer(date_pattern, text, re.IGNORECASE):
        entities.append({"text": m.group(), "label": "DATE", "start": m.start(), "end": m.end()})

    # Dollar amounts
    amount_pattern = r"\$[\d,]+(?:\.\d{2})?|USD\s*[\d,]+"
    for m in re.finditer(amount_pattern, text, re.IGNORECASE):
        entities.append({"text": m.group(), "label": "DOLLAR_AMOUNT", "start": m.start(), "end": m.end()})

    # Termination clauses (simple keyword search)
    term_pattern = r"[^.]*(?:terminat|cancel|withdraw)[^.]*\."
    for m in re.finditer(term_pattern, text, re.IGNORECASE):
        entities.append({"text": m.group().strip(), "label": "TERMINATION_CLAUSE", "start": m.start(), "end": m.end()})

    return entities


# ─────────────────────────────────────────────
# PYDANTIC MODELS (Request / Response shapes)
# ─────────────────────────────────────────────

class TextRequest(BaseModel):
    """Request body for the /extract/text endpoint."""
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Agreement dated March 15 2024 between Apex LLC and Beta Corp for $500,000. Either party may terminate with 30 days notice."
            }
        }


class EntityResponse(BaseModel):
    """Full response from the /extract endpoints."""
    document_id:         str
    filename:            str
    status:              str
    extraction_method:   str           # "digital", "ocr", or "text_input"
    page_count:          int
    processing_time_sec: float
    entities: dict                     # validated entities
    validation_summary:  dict
    timestamp:           str


class HealthResponse(BaseModel):
    status:      str
    ner_backend: str
    model_loaded: bool
    version:     str


# ─────────────────────────────────────────────
# FastAPI APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="LexiScan Auto API",
    description="Intelligent Legal Contract Entity Extractor (NER). Extracts Dates, Parties, Dollar Amounts, and Termination Clauses from PDF contracts.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow cross-origin requests (needed if a web frontend calls this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# STARTUP EVENT — load models once
# ─────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("LexiScan Auto API starting up...")
    load_ner_model()
    logger.info("API ready.")


# ─────────────────────────────────────────────
# MIDDLEWARE — log every request
# ─────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration:.2f}s)")
    return response


# ─────────────────────────────────────────────
# INTERNAL: Run full pipeline on text
# ─────────────────────────────────────────────
def process_text(text: str, filename: str, method: str, page_count: int) -> dict:
    """
    Core pipeline:
        text → NER model → validate entities → structured JSON
    """
    start_time = time.time()
    doc_id = str(uuid.uuid4())[:8]

    # Step 1: Run NER
    raw_entities = run_ner(text)
    logger.info(f"  NER extracted {len(raw_entities)} raw entities")

    # Step 2: Validate + normalise entities
    validated = validate_entities(raw_entities)
    logger.info(f"  Validation: {validated['validation_summary']['passed']} passed, "
                f"{validated['validation_summary']['rejected']} rejected")

    processing_time = round(time.time() - start_time, 3)

    return {
        "document_id":         doc_id,
        "filename":            filename,
        "status":              "success",
        "extraction_method":   method,
        "page_count":          page_count,
        "processing_time_sec": processing_time,
        "entities": {
            "dates":               validated["dates"],
            "parties":             validated["parties"],
            "amounts":             validated["amounts"],
            "termination_clauses": validated["termination_clauses"],
        },
        "validation_summary":  validated["validation_summary"],
        "timestamp":           datetime.utcnow().isoformat() + "Z",
    }


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", summary="Root — welcome message")
async def root():
    return {"message": "LexiScan Auto API v1.0 — POST /extract to process a PDF."}


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health():
    """Check if the API and NER model are running correctly."""
    return {
        "status":       "healthy",
        "ner_backend":  NER_BACKEND,
        "model_loaded": _ner_model is not None,
        "version":      "1.0.0",
    }


@app.post("/extract", response_model=EntityResponse, summary="Extract entities from a PDF file")
async def extract_from_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF contract and extract legal entities.

    - Accepts: PDF files (native digital or scanned)
    - Returns: Structured JSON with Dates, Parties, Amounts, Termination Clauses
    - Max file size: 50 MB
    """
    # ── Validate file type ──
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # ── Read and size-check ──
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB."
        )

    logger.info(f"Received PDF: {file.filename} ({size_mb:.2f} MB)")

    # ── Save to temp file ──
    tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    try:
        with open(tmp_path, "wb") as f:
            f.write(contents)

        # ── Step 1: OCR extraction ──
        ocr_result = extract_text_from_pdf(tmp_path)
        text        = ocr_result["text"]
        method      = ocr_result["method"]
        page_count  = ocr_result["page_count"]

        if len(text.strip()) < 50:
            raise HTTPException(
                status_code=422,
                detail="Could not extract readable text from PDF. The document may be corrupted or image-only without OCR support."
            )

        # ── Steps 2+3: NER + Validation ──
        result = process_text(text, file.filename, method, page_count)
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")
    finally:
        # Always clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/extract/text", response_model=EntityResponse, summary="Extract entities from raw text")
async def extract_from_text(request: TextRequest):
    """
    Send raw contract text (already extracted) for entity recognition.
    Useful when you have your own OCR or are processing copy-pasted text.
    """
    text = request.text.strip()

    if len(text) < 20:
        raise HTTPException(
            status_code=400,
            detail="Text too short. Please provide at least 20 characters of contract text."
        )

    if len(text) > 500_000:
        raise HTTPException(
            status_code=413,
            detail="Text too long (> 500,000 characters). Please split into smaller chunks."
        )

    logger.info(f"Received text input: {len(text)} chars")
    result = process_text(text, filename="text_input", method="text_input", page_count=1)
    return JSONResponse(content=result)


# ─────────────────────────────────────────────
# RUN THE SERVER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",    # listen on all interfaces (needed in Docker)
        port=8000,
        reload=False,       # set True for development hot-reload
        workers=1,          # 1 worker per container (scale by adding containers)
        log_level="info",
    )
