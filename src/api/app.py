"""
FastAPI REST API Service
Exposes multimodal deepfake detection and semantic fact-checking endpoints.
"""

from typing import Optional
import os
import shutil
import tempfile
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

from src.pipeline import MultimodalVerificationPipeline

app = FastAPI(
    title="Multimodal Deepfake Forensics & Fact-Checking API",
    description="Dual-branch hybrid verification API for synthetic video and contextual cheapfakes.",
    version="1.0.0"
)

# Global pipeline instance (lazy loaded)
pipeline: Optional[MultimodalVerificationPipeline] = None


def get_pipeline() -> MultimodalVerificationPipeline:
    global pipeline
    if pipeline is None:
        pipeline = MultimodalVerificationPipeline()
    return pipeline


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    device: str
    cuda_available: bool


class ClaimCheckRequest(BaseModel):
    claim_text: str
    top_k: int = 3


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns runtime health and hardware acceleration info."""
    return HealthResponse(
        status="healthy",
        service="Multimodal Deepfake & Fact-Checking Engine",
        version="1.0.0",
        device="cuda" if torch.cuda.is_available() else "cpu",
        cuda_available=torch.cuda.is_available()
    )


@app.post("/api/v1/verify/video")
async def verify_video(
    file: UploadFile = File(...),
    claim_caption: Optional[str] = Form(None)
):
    """
    Upload a video file (.mp4, .avi, .mov) for complete multimodal analysis.
    Optional claim_caption allows testing cheapfakes / recontextualized titles.
    """
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload an MP4, AVI, MOV, or WEBM video."
        )

    pipe = get_pipeline()

    # Save to temporary file for processing
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        report = pipe.analyze_media(tmp_path, override_claim_text=claim_caption)
        return report.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.post("/api/v1/claims/check")
def check_claim(request: ClaimCheckRequest):
    """
    Rapid text-only semantic fact-check against indexed knowledge bases.
    """
    pipe = get_pipeline()
    evidence = pipe.retriever.vector_store.query(request.claim_text, top_k=request.top_k)
    return {
        "query_claim": request.claim_text,
        "evidence_matches": [
            {
                "chunk_id": e.chunk_id,
                "claim_reference": e.claim_reference,
                "ground_truth_verdict": e.ground_truth_verdict,
                "source": e.source,
                "similarity": e.similarity_score,
                "evidence_text": e.evidence_text
            }
            for e in evidence
        ]
    }
