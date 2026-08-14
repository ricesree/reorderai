"""Fill products.product_name / brand / size from products.name via an LLM."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas.product_enrichment import ProductEnrichBatchResponse, ProductEnrichRequest
from api.services import product_enrichment_service as service

router = APIRouter(prefix="/api/products", tags=["products"])


@router.post("/enrich")
def enrich_products(body: ProductEnrichRequest) -> dict:
    if body.id is not None:
        try:
            return service.enrich_one(body.id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"LLM enrichment failed: {exc}") from exc
    try:
        result = service.enrich_missing(limit=body.limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Enrichment batch failed: {exc}") from exc
    return ProductEnrichBatchResponse(**result).model_dump()
