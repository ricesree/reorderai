"""Async (queued) invoice PDF parsing.

Same job pattern as POST /api/products/enrich: upload returns a job_id
immediately, poll status for the result. Parsing runs invoice_agent's own
two-phase OpenAI pipeline (unmodified), which can take a while per PDF -
queuing keeps that off the request thread instead of blocking the upload.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.services import invoice_parse_queue as queue_service

router = APIRouter(prefix="/api/invoice-agent", tags=["invoice-agent"])


@router.post("/parse-invoice")
async def queue_parse_invoice(
    file: UploadFile = File(...),
    tenant_id: str | None = Form(default=None),
) -> dict:
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only application/pdf uploads are supported")
    pdf_bytes = await file.read()
    job_id = queue_service.enqueue(tenant_id, file.filename or "invoice.pdf", pdf_bytes)
    return {"status": "queued", "job_id": job_id, "tenant_id": tenant_id}


@router.get("/parse-invoice/status/{job_id}")
def parse_invoice_status(job_id: str) -> dict:
    status = queue_service.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return status
