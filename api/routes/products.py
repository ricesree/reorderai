"""Fill products.product_name / brand / size from products.name via an LLM."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.product_enrichment import ProductEnrichRequest
from api.services import enrich_queue
from api.services import product_enrichment_service as service

router = APIRouter(prefix="/api/products", tags=["products"])


@router.post("/enrich")
def enrich_products(body: ProductEnrichRequest) -> dict:
    if body.id is not None:
        try:
            return service.enrich_one(body.id, tenant_id=body.tenant_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"LLM enrichment failed: {exc}") from exc
    job_id, is_new = enrich_queue.enqueue(body.tenant_id, body.limit)
    if not is_new:
        raise HTTPException(
            status_code=409,
            detail={"status": "already_running", "job_id": job_id},
        )
    return {"status": "queued", "job_id": job_id}


@router.get("/enrich/status/{job_id}")
def enrich_status(job_id: str) -> dict:
    status = enrich_queue.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return status
