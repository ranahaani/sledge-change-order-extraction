"""Small HTTP API over the same pipeline as the CLI.

Run with:
    uvicorn change_order_extract.api:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from change_order_extract import __version__
from change_order_extract.pipeline import extract_bytes, extract_text
from change_order_extract.schema import ChangeOrderExtraction

app = FastAPI(
    title="Change Order Extraction",
    version=__version__,
    summary="Extract a construction change order into validated JSON with confidence scores.",
)


class TextRequest(BaseModel):
    text: str = Field(min_length=1)


class Health(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", version=__version__)


@app.post("/extract", response_model=ChangeOrderExtraction)
def extract_from_text(body: TextRequest) -> ChangeOrderExtraction:
    return extract_text(body.text, source_name="request")


@app.post("/extract/file", response_model=ChangeOrderExtraction)
async def extract_from_file(file: UploadFile = File(...)) -> ChangeOrderExtraction:
    name = file.filename or "upload.txt"
    suffix = Path(name).suffix.lower()
    if suffix not in {".txt", ".md", ".pdf"}:
        raise HTTPException(status_code=415, detail="upload a .txt, .md, or .pdf file")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    try:
        return extract_bytes(data, suffix, source_name=Path(name).name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
